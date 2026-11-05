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
"""Tests for rebuilding a development sweep's folds from their run records (dlux.eval.trained_folds)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dlux.config.cohort import Cohort, ContractField, Objective, Role, SplitParams, Storage, Study
from dlux.eval import trained_folds
from dlux.eval.trained_folds import (
    assert_disjoint_from_development,
    discover,
    fold_target_stats,
    load,
    load_run_cfg,
    plan,
    read_fold_states,
    rebuild_task,
    recorded_tiling,
    require_complete,
)
from dlux.tasks import build as task_build
from dlux.tasks.fit_state import FitState


def _fold(exp_dir: Path, name: str, *, ckpt: bool = True, metadata: dict | None = None) -> Path:
    run_dir = exp_dir / name
    run_dir.mkdir(parents=True)
    if metadata is not None:
        (run_dir / "metadata.json").write_text(json.dumps(metadata))
    if ckpt:
        (run_dir / "best.ckpt").write_bytes(b"")
    return run_dir


def test_discover_dev_folds_keeps_complete_cv_folds_only(tmp_path):
    _fold(tmp_path, "er_cv_o0_i0", metadata={})
    _fold(tmp_path, "er_cv_o1_i0", metadata={})
    _fold(tmp_path, "her2_cv_o0_i0", metadata={})
    _fold(tmp_path, "er_cv_o2_i0", ckpt=False, metadata={})  # no checkpoint
    _fold(tmp_path, "er_cv_o3_i0")  # no metadata
    _fold(tmp_path, "scratch", metadata={})  # not a cv split
    folds = discover(tmp_path)
    assert sorted(folds) == ["er", "her2"]
    assert [(o, i) for o, i, _ in folds["er"]] == [(0, 0), (1, 0)]
    assert folds["er"][0][2] == tmp_path / "er_cv_o0_i0" / "best.ckpt"


def test_load_run_cfg_swaps_in_the_callers_paths(tmp_path):
    OmegaConf.save(
        {"paths": {"root_dir": "/sweep/site"}, "task": {"_target_": "x.Task", "inputs": {"tiles": {}}}},
        tmp_path / "hparams.yaml",
    )
    run_cfg = load_run_cfg(tmp_path, OmegaConf.create({"root_dir": "/here"}))
    assert run_cfg.paths.root_dir == "/here"


def test_load_run_cfg_refuses_a_record_without_task_inputs(tmp_path):
    OmegaConf.save({"task": {"_target_": "x.Task"}}, tmp_path / "hparams.yaml")
    with pytest.raises(ValueError, match="no task _target_ / inputs"):
        load_run_cfg(tmp_path, OmegaConf.create({}))


def test_recorded_tiling_resolves_the_runs_tiling():
    run_cfg = OmegaConf.create({"mpp": 1.0, "tiling": {"mpp": "${mpp}", "tile_size": [512, 512]}})
    assert recorded_tiling(run_cfg) == {"mpp": 1.0, "tile_size": [512, 512]}


def test_fold_target_stats_is_the_recorded_scalar_scale(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"target_mean": 2.5, "target_std": 0.5}))
    assert fold_target_stats(tmp_path) == (2.5, 0.5)


def test_fold_target_stats_refuses_a_run_without_a_scalar_scale(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"target_mean": None, "target_std": None}))
    with pytest.raises(ValueError, match="records no scalar target_mean/target_std"):
        fold_target_stats(tmp_path)


def test_read_fold_states_refuses_a_fold_without_the_others_state(tmp_path, monkeypatch):
    monkeypatch.setattr(trained_folds, "read_stream_state", lambda run_dir: {"rna": {}} if "o0" in run_dir.name else {})
    folds = [
        (0, 0, _fold(tmp_path, "er_cv_o0_i0", metadata={}) / "best.ckpt"),
        (1, 0, _fold(tmp_path, "er_cv_o1_i0", metadata={}) / "best.ckpt"),
    ]
    with pytest.raises(ValueError, match="inconsistently"):
        read_fold_states("er", folds)


def test_read_fold_states_accepts_folds_that_record_nothing(tmp_path):
    folds = [(0, 0, _fold(tmp_path, "er_cv_o0_i0", metadata={}) / "best.ckpt")]
    assert read_fold_states("er", folds) == {(0, 0): {}}


def _patients(by_db: dict[str, set[str]]):
    return lambda db_uri, manifest_name, split_version, categories: by_db[db_uri]


def test_assert_disjoint_from_development_passes_disjoint_cohorts(monkeypatch):
    monkeypatch.setattr(trained_folds, "split_patients", _patients({"dev": {"p1", "p2"}, "val": {"q1", "q2"}}))
    assert_disjoint_from_development("er", "dev", "dev_cohort", "val", "val_cohort", "all_test_er")


def test_assert_disjoint_from_development_refuses_a_shared_patient(monkeypatch):
    monkeypatch.setattr(trained_folds, "split_patients", _patients({"dev": {"p1", "p2"}, "val": {"p2", "q1"}}))
    with pytest.raises(ValueError, match=r"1 of the 2 patients .* \['p2'\]"):
        assert_disjoint_from_development("er", "dev", "dev_cohort", "val", "val_cohort", "all_test_er")


def test_assert_disjoint_from_development_refuses_an_empty_development_split(monkeypatch):
    monkeypatch.setattr(trained_folds, "split_patients", _patients({"dev": set(), "val": {"q1"}}))
    with pytest.raises(ValueError, match="development split .* has no patients"):
        assert_disjoint_from_development("er", "dev", "dev_cohort", "val", "val_cohort", "all_test_er")


def test_assert_disjoint_from_development_refuses_an_empty_scored_split(monkeypatch):
    monkeypatch.setattr(trained_folds, "split_patients", _patients({"dev": {"p1"}, "val": set()}))
    with pytest.raises(ValueError, match="scored split .* has no patients"):
        assert_disjoint_from_development("er", "dev", "dev_cohort", "val", "val_cohort", "all_test_er")


def _grid(n_outer: int, n_inner: int, drop: tuple[int, int] | None = None):
    return [(o, i, Path(f"o{o}_i{i}/best.ckpt")) for o in range(n_outer) for i in range(n_inner) if (o, i) != drop]


def test_require_complete_accepts_the_full_sweep():
    require_complete("er", _grid(2, 2), 2, 2, strict=True)


def test_require_complete_refuses_a_missing_model():
    with pytest.raises(ValueError, match=r"1 of 4 development models are missing \(\['o1_i0'\]\)"):
        require_complete("er", _grid(2, 2, drop=(1, 0)), 2, 2, strict=True)


def test_require_complete_lets_a_partial_sweep_through_when_not_strict():
    require_complete("er", _grid(2, 2, drop=(1, 0)), 2, 2, strict=False)


class _ExpressionTask:
    """The part of an expression task that ``load`` reads: its streams and its fit state."""

    streams: list = []

    def fit_state(self) -> FitState:
        return FitState(pred_mean=np.array([1.0, 2.0]), pred_std=np.array([0.5, 0.25]))


def _fold_with_checkpoint(tmp_path: Path, module: torch.nn.Module, metadata: dict) -> Path:
    run_dir = _fold(tmp_path, "er_cv_o0_i0", metadata=metadata)
    torch.save({"state_dict": module.state_dict()}, run_dir / "best.ckpt")
    return run_dir / "best.ckpt"


def test_load_fold_loads_the_weights_and_gives_the_per_gene_statistics_for_expression(tmp_path):
    trained, fresh = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    ckpt = _fold_with_checkpoint(tmp_path, trained, {"target_mean": None, "target_std": None})

    fit = load(fresh, _ExpressionTask(), ckpt, {}, Objective.regression_vector, torch.device("cpu"))

    assert torch.equal(fresh.weight, trained.weight)
    np.testing.assert_array_equal(fit.pred_mean, [1.0, 2.0])
    np.testing.assert_array_equal(fit.pred_std, [0.5, 0.25])


def test_load_fold_gives_the_recorded_scalar_statistics_for_a_scalar_target(tmp_path):
    ckpt = _fold_with_checkpoint(tmp_path, torch.nn.Linear(2, 2), {"target_mean": 3.0, "target_std": 2.0})
    fit = load(torch.nn.Linear(2, 2), _ExpressionTask(), ckpt, {}, Objective.regression, torch.device("cpu"))
    assert (fit.pred_mean, fit.pred_std, fit.time_edges) == (3.0, 2.0, None)


def test_load_fold_gives_the_recorded_time_bins_for_a_survival_target(tmp_path):
    ckpt = _fold_with_checkpoint(
        tmp_path, torch.nn.Linear(2, 2), {"target_mean": 0.0, "target_std": 1.0, "time_edges": [3.0, 7.0]}
    )
    fit = load(torch.nn.Linear(2, 2), _ExpressionTask(), ckpt, {}, Objective.survival, torch.device("cpu"))
    np.testing.assert_array_equal(fit.time_edges, [3.0, 7.0])


class _RecordingTask:
    def __init__(self, target, contract_field, data_description, inputs, target_stats=None):
        self.target_stats = target_stats

    def register_adapter_augmentations(self, augmentations):
        pass


def test_rebuild_task_cuts_a_binarize_target_at_the_scored_cohorts_own_statistic(tmp_path, monkeypatch):
    monkeypatch.setattr(trained_folds, "build_inputs", lambda run_cfg, context: {})
    monkeypatch.setattr(task_build, "get_class", lambda path: _RecordingTask)
    cohort = Cohort(
        name="val",
        storage=Storage(image_dir="/i"),
        contract={
            "high": ContractField.model_validate(
                {"source": {"column": "x", "transform": {"binarize": {"method": "median"}}}, "objective": "binary"}
            )
        },
    )
    stats = tmp_path / "val_contract_stats.json"
    stats.write_text(json.dumps({"high": {"transform": "binarize", "column": "x", "threshold": 0.3, "n": 9}}))
    run_cfg = OmegaConf.create(
        {"task": {"_target_": "x.Task", "inputs": {}, "target": {}}, "feature_extractor": {}, "augmentations": {}}
    )

    task = rebuild_task("high", run_cfg, cohort, None, cohorts_dir=tmp_path, contract_stats=stats, recorded_state={})

    assert task.target_stats["threshold"] == 0.3


def test_a_scalar_scale_is_recorded_and_a_per_gene_scale_is_not() -> None:
    scalar = FitState(pred_mean=2.0, pred_std=0.5, time_edges=np.array([3.0, 7.0])).recorded()
    assert (scalar["target_mean"], scalar["target_std"], scalar["time_edges"]) == (2.0, 0.5, [3.0, 7.0])
    per_gene = FitState(pred_mean=np.array([1.0, 2.0]), pred_std=np.array([0.5, 0.25])).recorded()
    assert (per_gene["target_mean"], per_gene["target_std"]) == (None, None)


# -- plan --------------------------------------------------------------------
def _study(**cohorts: Role) -> Study:
    return Study(
        name="s",
        cohorts=cohorts,
        splits=SplitParams(n_outer=1, n_inner=2),
        targets=["er", "os", "pr"],
        require_modalities=[],
    )


def _val_cohort(*fields: str) -> Cohort:
    kinds = {
        "er": {"type": "binary", "map": {"neg": 0, "pos": 1}},
        "os": {"type": "survival", "event": "dead", "time": "days"},
        "pr": {"type": "binary", "map": {"neg": 0, "pos": 1}},
    }
    return Cohort(
        name="val",
        storage=Storage(image_dir="/i"),
        contract={f: ContractField.model_validate(kinds[f]) for f in fields},
    )


def _sweep(studies_dir: Path, *fields: str) -> None:
    exp_dir = studies_dir / "s" / "runs" / "exp" / "dev"
    for field in fields:
        for inner in range(2):
            _fold(exp_dir, f"{field}_cv_o0_i{inner}", metadata={})


def _plan(studies_dir: Path, study: Study | None = None, cohort: Cohort | None = None, **kwargs):
    return plan(
        study or _study(dev=Role.development, val=Role.validation),
        cohort or _val_cohort("er", "os", "pr"),
        role=kwargs.pop("role", Role.validation),
        flow="evaluate_external",
        experiment="exp",
        studies_dir=studies_dir,
        strict=True,
        **kwargs,
    )


def test_plan_runs_the_endpoints_the_study_the_sweep_and_the_cohort_share(tmp_path):
    _sweep(tmp_path, "pr", "er", "os")
    planned = _plan(tmp_path, cohort=_val_cohort("os", "pr"))
    assert planned.fields == ["os", "pr"]  # er: not in the cohort's contract
    assert planned.dev_cohort == "dev"
    assert [(o, i) for o, i, _ in planned.folds["pr"]] == [(0, 0), (0, 1)]
    assert planned.db_uri.endswith("/s/db/val/manifest.db") and planned.dev_db_uri.endswith("/s/db/dev/manifest.db")


def test_plan_leaves_out_an_endpoint_the_sweep_did_not_train(tmp_path):
    _sweep(tmp_path, "er")
    assert _plan(tmp_path).fields == ["er"]


def test_plan_leaves_out_an_objective_the_flow_cannot_report(tmp_path):
    _sweep(tmp_path, "er", "os")
    assert _plan(tmp_path, reportable=["binary"]).fields == ["er"]


def test_plan_narrows_to_only_fields_and_refuses_an_unknown_one(tmp_path):
    _sweep(tmp_path, "er", "os")
    assert _plan(tmp_path, only_fields=["os"]).fields == ["os"]
    with pytest.raises(ValueError, match=r"only_fields=\['pr'\] are not endpoints"):
        _plan(tmp_path, only_fields=["pr"])


def test_plan_refuses_a_cohort_without_the_flows_role(tmp_path):
    _sweep(tmp_path, "er")
    with pytest.raises(ValueError, match=r"'val' is not a 'predict' cohort of study 's' \(role=validation\)"):
        _plan(tmp_path, role=Role.predict)


def test_plan_needs_exactly_one_development_cohort(tmp_path):
    _sweep(tmp_path, "er")
    two = _study(dev=Role.development, dev2=Role.development, val=Role.validation)
    with pytest.raises(ValueError, match=r"exactly one development cohort .* \['dev', 'dev2'\]"):
        _plan(tmp_path, study=two)


def test_plan_refuses_an_incomplete_sweep(tmp_path):
    _fold(tmp_path / "s" / "runs" / "exp" / "dev", "er_cv_o0_i0", metadata={})
    with pytest.raises(ValueError, match="1 of 2 development models are missing"):
        _plan(tmp_path)


def test_plan_refuses_a_sweep_with_no_shared_endpoint(tmp_path):
    _sweep(tmp_path, "er")
    with pytest.raises(ValueError, match="no endpoint is shared"):
        _plan(tmp_path, cohort=_val_cohort("os"))


def test_rebuild_task_gives_a_survival_task_its_recorded_options(tmp_path, monkeypatch):
    monkeypatch.setattr(trained_folds, "build_inputs", lambda run_cfg, context: {})
    cohort = _val_cohort("os")
    run_cfg = OmegaConf.create(
        {
            "task": {"_target_": "dlux.tasks.wsi_survival.WSISurvivalTask", "target": {"field": "os"}, "n_bins": 6},
            "feature_extractor": {},
            "augmentations": {},
        }
    )

    task = rebuild_task(
        "os", run_cfg, cohort, None, cohorts_dir=tmp_path, contract_stats=tmp_path / "none.json", recorded_state={}
    )

    assert task.num_classes == 6
    assert task.fit_state().time_edges is None  # each fold's bins come with that fold's own record


# -- the cohort flows ------------------------------------------------------------
def _fake_predictions(per_fold: dict):
    """Stands in for ``trained_folds.predictions``: one per-slide prediction dict per fold."""

    def _predictions(rebuilt, folds, fold_states, *, stage, trainer_cfg, device, require_labels=True):
        for outer, inner, _ckpt in folds:
            yield outer, inner, per_fold[(outer, inner)]

    return _predictions


def _slides(probs, labels=None, attention=None):
    return {
        "patient_codes": np.asarray(["p0", "p0", "p1"]),
        "probs": np.asarray(probs, dtype=np.float64),
        "labels": None if labels is None else np.asarray(labels, dtype=np.float64),
        "attention": attention,
    }


def test_cohort_predictions_rolls_slides_to_patients_and_casts_the_labels(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        trained_folds,
        "predictions",
        _fake_predictions(
            {(0, 0): _slides([0.2, 0.4, 0.9], [0, 0, 1], attention=["a"]), (0, 1): _slides([0.6, 0.6, 0.5], [0, 0, 1])}
        ),
    )
    rebuilt = SimpleNamespace(objective=Objective.binary, field="er")
    folds = [(0, 0, Path("x")), (0, 1, Path("y"))]

    per_model, labels, attention = trained_folds.cohort_predictions(
        rebuilt, folds, {}, stage="test", trainer_cfg=None, device=None
    )

    assert per_model[(0, 0)] == pytest.approx({"p0": 0.3, "p1": 0.9})
    assert labels == {"p0": 0, "p1": 1} and all(isinstance(v, int) for v in labels.values())
    assert attention == ["a"]


def test_predict_field_grand_ensembles_every_fold(tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(trained_folds, "compile_data_description", lambda *a, **k: None)
    monkeypatch.setattr(trained_folds, "read_fold_states", lambda field, folds: {(0, 0): {}, (1, 0): {}})
    monkeypatch.setattr(
        trained_folds,
        "rebuild",
        lambda *a, **k: SimpleNamespace(objective=Objective.binary, field="er", task=SimpleNamespace(num_classes=1)),
    )
    monkeypatch.setattr(
        trained_folds,
        "predictions",
        _fake_predictions({(0, 0): _slides([0.2, 0.4, 0.8]), (1, 0): _slides([0.4, 0.6, 0.6])}),
    )
    plan = SimpleNamespace(
        folds={"er": [(0, 0, Path("a")), (1, 0, Path("b"))]}, db_uri="", contract_stats=tmp_path, device=None
    )

    prediction = trained_folds.predict_field(
        "er", plan, None, OmegaConf.create({"tiling": {}}), cohorts_dir=tmp_path, trainer_cfg=None, num_workers=0
    )

    assert prediction.patients == ["p0", "p1"]
    np.testing.assert_allclose(prediction.predictions, [0.4, 0.7])  # mean over folds of each fold's slide mean
    trained_folds.write_cohort_predictions([prediction], "exp", tmp_path / "out")
    with np.load(tmp_path / "out" / "per_patient_er.npz", allow_pickle=False) as table:
        assert table["patient_codes"].tolist() == ["p0", "p1"] and str(table["endpoint_type"]) == "binary"
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "per_patient_er.csv",
        "per_patient_er.npz",
        "summary.md",
    ]


def test_predict_field_refuses_folds_that_predict_other_patients(tmp_path, monkeypatch):
    from types import SimpleNamespace

    other = {
        "patient_codes": np.asarray(["p0", "p2"]),
        "probs": np.asarray([0.1, 0.2]),
        "labels": None,
        "attention": None,
    }
    monkeypatch.setattr(trained_folds, "compile_data_description", lambda *a, **k: None)
    monkeypatch.setattr(trained_folds, "read_fold_states", lambda field, folds: {(0, 0): {}, (1, 0): {}})
    monkeypatch.setattr(
        trained_folds,
        "rebuild",
        lambda *a, **k: SimpleNamespace(objective=Objective.binary, field="er", task=SimpleNamespace(num_classes=1)),
    )
    monkeypatch.setattr(
        trained_folds, "predictions", _fake_predictions({(0, 0): _slides([0.2, 0.4, 0.8]), (1, 0): other})
    )
    plan = SimpleNamespace(
        folds={"er": [(0, 0, Path("a")), (1, 0, Path("b"))]}, db_uri="", contract_stats=tmp_path, device=None
    )
    with pytest.raises(ValueError, match="predict other patients"):
        trained_folds.predict_field(
            "er", plan, None, OmegaConf.create({"tiling": {}}), cohorts_dir=tmp_path, trainer_cfg=None, num_workers=0
        )
