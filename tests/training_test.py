# Copyright 2026 Joren Brunekreef. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the training steps of ``dlux.training``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from dlux.config.cohort import Cohort, ContractField, Objective, Splits, SplitStrategy, Storage
from dlux.data import layout
from dlux.data.build_db import build_db
from dlux.eval.aggregate import _read_meta
from dlux.eval.trained_folds import fold_target_stats
from dlux.tasks.fit_state import FitState
from dlux.training import (
    RunContext,
    _gene_panel,
    _patient_test_metric,
    _resolve_split_version,
    check_experiment_choice,
    check_target_choice,
    fit_and_test,
    load_best_checkpoint,
    run_metadata,
    scored_test_coverage,
)


def _trainer(callback: Any) -> Any:
    return cast(Any, SimpleNamespace(checkpoint_callback=callback))


def _checkpoint(best_model_path: str) -> ModelCheckpoint:
    callback = ModelCheckpoint()
    callback.best_model_path = best_model_path
    return callback


def test_load_best_checkpoint_loads_the_best_weights(tmp_path: Path):
    trained, fresh = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    best = tmp_path / "best.ckpt"
    torch.save({"state_dict": trained.state_dict()}, best)

    path = load_best_checkpoint(_trainer(_checkpoint(str(best))), fresh)

    assert path == str(best)
    assert torch.equal(fresh.weight, trained.weight)


def test_load_best_checkpoint_refuses_a_trainer_without_a_checkpoint_callback():
    with pytest.raises(RuntimeError, match="no ModelCheckpoint callback"):
        load_best_checkpoint(_trainer(None), torch.nn.Linear(2, 2))


@pytest.mark.parametrize("best_model_path", ["", "does/not/exist.ckpt"])
def test_load_best_checkpoint_refuses_a_fit_without_a_best_checkpoint(best_model_path: str):
    with pytest.raises(RuntimeError, match="no best checkpoint"):
        load_best_checkpoint(_trainer(_checkpoint(best_model_path)), torch.nn.Linear(2, 2))


# -- which fold this is -------------------------------------------------------
def _fold_cfg(**kwargs: Any) -> Any:
    return OmegaConf.create(
        {"fold": None, "split_version": None, "study": {"name": "s", "splits": {"n_outer": 2, "n_inner": 3}}, **kwargs}
    )


def test_a_flat_fold_index_names_its_outer_and_inner_fold():
    assert _resolve_split_version(_fold_cfg(fold=4), "er") == "er_cv_o1_i1"  # divmod(4, n_inner=3)
    assert _resolve_split_version(_fold_cfg(split_version="er_cv_o0_i2"), "er") == "er_cv_o0_i2"


@pytest.mark.parametrize("kwargs", [{"fold": 6}, {"fold": -1}, {}, {"fold": 0, "split_version": "er_cv_o0_i0"}])
def test_a_fold_outside_the_grid_or_not_exactly_one_name_is_refused(kwargs: dict):
    with pytest.raises(SystemExit):
        _resolve_split_version(_fold_cfg(**kwargs), "er")


def test_the_target_must_match_the_split_and_be_a_study_target():
    check_target_choice("er", "er_cv_o0_i0", ["er"])
    check_target_choice("er", "all_test_er", ["er"])  # a non-CV split names no field to cross-check
    with pytest.raises(SystemExit, match="holds the folds for 'grade'"):
        check_target_choice("er", "grade_cv_o0_i0", ["er", "grade"])
    with pytest.raises(SystemExit, match="is not one of study.targets"):
        check_target_choice("er", "er_cv_o0_i0", ["grade"])


def test_the_experiment_config_must_match_its_name_and_study():
    check_experiment_choice("s/baseline", "baseline", "s")
    with pytest.raises(SystemExit, match="chosen experiment leaf"):
        check_experiment_choice("s/baseline", "fusion", "s")
    with pytest.raises(SystemExit, match="nested under a different study"):
        check_experiment_choice("t/baseline", "baseline", "s")


# -- what a fold records ------------------------------------------------------
def _cohort_db(tmp_path: Path) -> str:
    """A real cohort DB from ``build_db``: 8 patients, a 2 x 2 nested CV on ``cancer_type``."""
    codes = [f"brca{i}" for i in range(4)] + [f"coad{i}" for i in range(4)]
    patients, slides = tmp_path / "patients.csv", tmp_path / "slides.csv"
    patients.write_text("patient_id,cancer_type\n" + "".join(f"{c},{c[:4].upper()}\n" for c in codes))
    slides.write_text("patient_id,image_path\n" + "".join(f"{c},{c}.svs\n" for c in codes))
    db = tmp_path / "db"
    build_db(
        Cohort(
            name="c",
            storage=Storage(image_dir=str(tmp_path)),
            contract={"cancer_type": ContractField.model_validate({"type": "binary", "map": {"BRCA": 0, "COAD": 1}})},
        ),
        Splits(strategy=SplitStrategy.nested_cv, n_outer=2, n_inner=2),
        db,
        patients_csv=patients,
        slides_csv=slides,
        probe="none",
        analyze=False,
    )
    return f"sqlite:///{layout.manifest_db(db)}"


def test_coverage_names_the_declared_test_patients_that_got_no_prediction(tmp_path: Path):
    ctx = cast(Any, SimpleNamespace(cohort_name="c", split_version="cancer_type_cv_o0_i0"))
    setup = cast(Any, SimpleNamespace(database_uri=_cohort_db(tmp_path)))
    declared = scored_test_coverage(ctx, setup, set())["lost"]
    assert declared and set(declared) <= {f"brca{i}" for i in range(4)} | {f"coad{i}" for i in range(4)}

    coverage = scored_test_coverage(ctx, setup, set(declared[1:]))
    assert coverage == {"declared": len(declared), "scored": len(declared) - 1, "lost": [declared[0]]}


def _context(tmp_path: Path) -> RunContext:
    return RunContext(
        output_dir=tmp_path,
        run_dir=tmp_path / "run",
        study_name="s",
        cohort_name="c",
        experiment_name="baseline",
        split_version="er_cv_o0_i0",
        exp_choice="s/baseline",
        target="er",
        target_key="patient.er",
        target_normalize="zscore",
        force=False,
        random_baseline=False,
        train_start_iso="2026-01-01T00:00:00+00:00",
    )


def test_the_metadata_a_fold_writes_is_what_aggregate_and_external_scoring_read(tmp_path: Path):
    task = SimpleNamespace(
        num_classes=1, fit_state=lambda: FitState(pred_mean=np.float64(2.5), pred_std=np.float64(0.5))
    )
    setup = cast(
        Any,
        SimpleNamespace(
            task=task,
            study=SimpleNamespace(splits=SimpleNamespace(n_outer=5, n_inner=5)),
            objective=Objective.regression,
            model_name="uni2",
            model_hash="abc",
        ),
    )
    meta = run_metadata(
        OmegaConf.create({"task_name": "wsi_regression"}),
        _context(tmp_path),
        setup,
        final_epoch=3,
        best_ckpt_path=str(tmp_path / "epoch_003.ckpt"),
        test_metrics={"test/r2": 0.4},
        test_coverage={"declared": 10, "scored": 9, "lost": ["p1"]},
    )
    (tmp_path / "metadata.json").write_text(json.dumps(meta))  # every value must be JSON-serialisable

    assert _read_meta(tmp_path)["n_outer"] == 5
    assert fold_target_stats(tmp_path) == (2.5, 0.5)
    assert meta["best_ckpt_filename"] == "epoch_003.ckpt" and meta["endpoint_type"] == "regression"


def test_the_gene_panel_column_reads_an_input_panel_before_the_target_panel():
    fusion = OmegaConf.create(
        {"task": {"inputs": {"tiles": {}, "rna": {"gene_panel": "hvg2000"}}, "target": {"gene_panel": "full"}}}
    )
    expression = OmegaConf.create({"task": {"inputs": {"tiles": {}}, "target": {"gene_panel": "hallmark"}}})
    assert _gene_panel(fusion) == "hvg2000"
    assert _gene_panel(expression) == "hallmark"
    assert _gene_panel(OmegaConf.create({"task": {"inputs": {"tiles": {}}, "target": {"field": "er"}}})) == "none"


def test_the_patient_metric_is_keyed_beside_the_streamed_one():
    preds = {"patient_codes": ["p1", "p1", "p2"], "probs": np.array([0.9, 0.7, 0.1]), "labels": np.array([1, 1, 0])}
    assert _patient_test_metric(preds, Objective.binary) == {"test/auroc_patient": 1.0}
    assert _patient_test_metric(preds, Objective.regression_vector) == {}


# -- fit and test -------------------------------------------------------------
class _Regressor(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(2, 1)

    def _loss(self, batch: Any) -> torch.Tensor:
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x).squeeze(-1), y)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        return self._loss(batch)

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self.log("validate/loss", self._loss(batch))

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self.log("test/loss", self._loss(batch), logger=False)

    def configure_optimizers(self) -> Any:
        return torch.optim.SGD(self.parameters(), lr=0.1)


class _Data(LightningDataModule):
    def _loader(self) -> DataLoader:
        x = torch.linspace(-1, 1, 16).reshape(8, 2)
        return DataLoader(TensorDataset(x, x.sum(dim=1)), batch_size=4)

    def train_dataloader(self) -> DataLoader:
        return self._loader()

    def val_dataloader(self) -> DataLoader:
        return self._loader()

    def test_dataloader(self) -> DataLoader:
        return self._loader()


def _fit_cfg(tmp_path: Path) -> Any:
    checkpoint = {
        "_target_": "lightning.pytorch.callbacks.ModelCheckpoint",
        "dirpath": str(tmp_path / "ckpt"),
        "monitor": "validate/loss",
        "mode": "min",
        "save_top_k": 1,
    }
    return OmegaConf.create(
        {
            "trainer": {"max_epochs": 2, "accelerator": "cpu", "devices": 1, "enable_model_summary": False},
            "callbacks": {"model_checkpoint": checkpoint},
        }
    )


def test_fit_and_test_tests_the_best_checkpoint(tmp_path: Path):
    module = _Regressor()
    epoch, best, metrics = fit_and_test(
        _fit_cfg(tmp_path),
        SimpleNamespace(sample_unit="slide"),
        module,
        _Data(),
        run_logger=False,
        random_baseline=False,
        output_dir=tmp_path,
    )
    assert epoch == 2 and Path(best).parent == tmp_path / "ckpt"
    loaded = torch.load(best, weights_only=False)["state_dict"]
    assert torch.equal(module.layer.weight, loaded["layer.weight"])  # what was tested is the best checkpoint
    assert set(metrics) == {"test/loss"}


def test_a_random_baseline_tests_and_saves_the_untrained_weights(tmp_path: Path):
    module = _Regressor()
    initial = module.layer.weight.detach().clone()
    epoch, best, metrics = fit_and_test(
        _fit_cfg(tmp_path),
        SimpleNamespace(sample_unit="slide"),
        module,
        _Data(),
        run_logger=False,
        random_baseline=True,
        output_dir=tmp_path,
    )
    assert epoch == 0 and best == str(tmp_path / "checkpoints" / "random.ckpt")
    assert torch.equal(torch.load(best, weights_only=False)["state_dict"]["layer.weight"], initial)
    assert torch.equal(module.layer.weight, initial) and set(metrics) == {"test/loss"}
