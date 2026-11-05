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
"""Tests for build_task: the task config's ``_target_`` naming the class, and ``resolve_fit_extras``'
per-combo fit-time kwargs. The DB/parquet-touching ``_fit_*`` helpers are monkeypatched to sentinels,
so class selection and extras selection are exercised in isolation (no cache, no manifest, no RNA
matrix)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hydra.utils import get_class
from omegaconf import OmegaConf
from pydantic import ValidationError

from dlux.config.cohort import Cohort, ContractField, Objective, Splits, SplitStrategy, Storage
from dlux.data import layout
from dlux.data.build_db import build_db
from dlux.tasks import build
from dlux.tasks.tile_classification import TileClassificationTask
from dlux.tasks.wsi_bulk_rna import WSIBulkRNATask
from dlux.tasks.wsi_classification import WSIClassificationTask
from dlux.tasks.wsi_regression import WSIRegressionTask
from dlux.tasks.wsi_survival import WSISurvivalTask

# Sentinels returned by the monkeypatched fit helpers (so extras values are identifiable).
_GENE_IDS = ["g0", "g1"]
_EDGES = np.array([1.0, 2.0, 3.0], dtype=np.float32)
_FIT_INDICES = [0] * 7 + [1] * 1 + [2] * 2  # 7/1/2 -> pos_weight 7.0, balanced 3-class weights
_TSTATS = (11.0, 22.0)
_GENE_KEYS = {"matrix_path", "gene_means", "gene_stds", "gene_ids", "gene_panel"}

_FIELDS = {
    Objective.binary: {"type": "binary", "map": {"a": 0, "b": 1}},
    Objective.multiclass: {"type": "multiclass", "map": {"a": 0, "b": 1, "c": 2}},
    Objective.regression: {"type": "continuous"},
    Objective.regression_vector: {"type": "expression"},
    Objective.survival: {"type": "survival", "event": "dead", "time": "days"},
}


@pytest.fixture(autouse=True)
def _stub_fit(monkeypatch):
    """Replace every DB/parquet fit helper with a sentinel-returning stub."""
    monkeypatch.setattr(build, "_fit_gene_stats", lambda *a, **k: (_GENE_IDS, np.zeros(2), np.ones(2)))
    monkeypatch.setattr(build, "_fit_class_indices", lambda *a, **k: _FIT_INDICES)
    monkeypatch.setattr(build, "_fit_target_stats", lambda *a, **k: _TSTATS)
    monkeypatch.setattr(build, "_fit_time_bins", lambda *a, **k: _EDGES)


def _extras(objective, target_normalize="none", weight="none"):
    return build.resolve_fit_extras(
        objective,
        cfg=OmegaConf.create({"task": {"n_bins": 4}}),
        manifest_name="c",
        contract_field=ContractField.model_validate(_FIELDS[objective]),
        loss_weight=weight,
        target_normalize=target_normalize,
        database_uri="db",
        split_version="v",
        matrix_path=Path("/tmp/none.parquet"),
        gene_panel="hvg2000",
    )


def _task_config_dir() -> Path:
    """The shipped ``config/task`` dir, next to ``tests/`` (under Bazel, the runfiles copy from this
    target's ``data`` dep)."""
    candidate = Path(__file__).parents[1] / "config" / "task"
    if not candidate.is_dir():
        raise AssertionError(f"config/task not found at {candidate}")
    return candidate


def test_every_shipped_task_config_declares_a_resolvable_target():
    """Every task config must name its class, and that class must import — the task is chosen by
    ``_target_``, so a typo there is only discoverable at run time otherwise."""
    configs = sorted(_task_config_dir().glob("*.yaml"))
    assert configs, "no task configs found"
    expected = {
        "wsi_classification": WSIClassificationTask,
        "tile_classification": TileClassificationTask,
        "wsi_bulk_rna": WSIBulkRNATask,
        "wsi_regression": WSIRegressionTask,
        "wsi_survival": WSISurvivalTask,
    }
    assert {c.stem for c in configs} == set(expected), "a task class without a config, or vice versa"
    for config in configs:
        target = OmegaConf.load(config).get("_target_")
        assert target, f"{config.name} declares no _target_"
        assert get_class(str(target)) is expected[config.stem], f"{config.name} -> {target}"


def test_build_task_requires_a_target(monkeypatch):
    """A task config without ``_target_`` fails with a message naming the missing key, not an
    AttributeError deep inside construction."""
    cfg = OmegaConf.create({"task": {"target": {"field": "x"}}})
    field = ContractField.model_validate(_FIELDS[Objective.binary])
    with pytest.raises(ValueError, match=r"_target_"):
        build.build_task(cfg, None, field, None, None, "x_cv_o0_i0")


def test_task_class_rejects_a_mismatched_objective():
    """The objective guard lives in each task class, so no objective-to-task table is needed. A
    classification task handed a survival field must raise."""
    with pytest.raises(ValueError, match="binary/multiclass"):
        WSIClassificationTask(
            target="x",
            contract_field=ContractField(objective=Objective.survival, source={"column": "x", "time_column": "t"}),
            inputs={},
            data_description=None,
        )


def test_the_tile_task_refuses_a_non_binary_endpoint():
    with pytest.raises(ValueError, match="TileClassificationTask handles binary objectives only"):
        TileClassificationTask(
            target="x",
            contract_field=ContractField.model_validate(_FIELDS[Objective.multiclass]),
            inputs={},
            data_description=None,
        )


def test_extras_are_target_side_only():
    """Fit-time extras belong to the ENDPOINT. An input stream resolves its own statistics in
    Modality.from_spec, so no combination of inputs changes what appears here."""
    assert set(_extras(Objective.survival)) == {"time_edges"}
    assert set(_extras(Objective.regression)) == {"target_mean", "target_std"}
    assert set(_extras(Objective.regression_vector)) == _GENE_KEYS  # the expression TARGET's own stats
    assert set(_extras(Objective.binary)) == {"pos_weight", "class_weights"}
    assert set(_extras(Objective.multiclass)) == {"pos_weight", "class_weights"}


def test_target_zscore_regression_only_and_requested():
    assert _extras(Objective.regression, "zscore") == {"target_mean": 11.0, "target_std": 22.0}
    assert _extras(Objective.regression, "none") == {"target_mean": 0.0, "target_std": 1.0}


def test_loss_weighting_is_scalar_for_binary_and_a_vector_for_multiclass():
    """The two objectives take DIFFERENT weighting kwargs, so each must resolve only its own and
    leave the other None — the task passes both straight to the loss."""
    binary = _extras(Objective.binary, weight="balanced")
    assert binary["pos_weight"] == pytest.approx(7.0)  # 7 neg / 1 pos in _FIT_INDICES
    assert binary["class_weights"] is None

    multi = _extras(Objective.multiclass, weight="balanced")
    assert multi["pos_weight"] is None
    # n / (K * n_c) over 7/1/2 supports
    assert multi["class_weights"] == pytest.approx([10 / 21, 10 / 3, 10 / 6])


def test_loss_weighting_off_by_default():
    for objective in (Objective.binary, Objective.multiclass):
        assert _extras(objective) == {"pos_weight": None, "class_weights": None}


def test_loss_weighting_is_a_task_knob_not_a_contract_facet():
    """It is resolved from what the RUN asks for, so two arms on one cohort can differ. A contract
    field carrying `loss:` is rejected rather than quietly ignored."""
    assert _extras(Objective.binary, weight="balanced")["pos_weight"] == pytest.approx(7.0)
    with pytest.raises(ValidationError):
        ContractField.model_validate({**_FIELDS[Objective.binary], "loss": {"weight": "balanced"}})


def test_manual_multiclass_weights_fill_omitted_classes():
    assert _extras(Objective.multiclass, weight={"0": 2.0, "2": 4.0})["class_weights"] == [2.0, 1.0, 4.0]


def test_an_unknown_task_key_is_refused():
    """A key that neither task construction nor the task class reads would otherwise pass without effect."""
    cfg = OmegaConf.create(
        {
            "task": {
                "target": {"field": "x"},
                "fuse_rna": True,
                "_target_": "dlux.tasks.wsi_classification.WSIClassificationTask",
            }
        }
    )
    field = ContractField.model_validate(_FIELDS[Objective.binary])
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['fuse_rna'\]"):
        build.build_task(cfg, None, field, None, None, "x_cv_o0_i0")


def test_scalar_target_is_rejected():
    """`target` is a block (`target: {field: os}`). A scalar `target: os` is refused with that form in the
    message, not an obscure failure."""
    cfg = OmegaConf.create({"task": {"_target_": "dlux.tasks.wsi_survival.WSISurvivalTask", "target": "os"}})
    field = ContractField.model_validate(_FIELDS[Objective.binary])
    with pytest.raises(ValueError, match=r"target: \{field: os\}"):
        build.build_task(cfg, None, field, None, None, "os_cv_o0_i0")


def test_survival_bins_and_edges():
    extras = _extras(Objective.survival)
    assert np.allclose(extras["time_edges"], _EDGES)


class _ForkedTask:
    def __init__(self, target, contract_field, data_description, inputs, target_stats=None, sigma=0.7):
        self.sigma = sigma


def test_a_task_takes_its_own_options_from_the_task_config(monkeypatch):
    monkeypatch.setattr(build, "get_class", lambda path: _ForkedTask)
    task_cfg = OmegaConf.create({"_target_": "p.Forked", "target": {"field": "x"}, "loss": {}, "sigma": 0.5})

    task = build.construct_task(
        task_cfg,
        target="x",
        contract_field=None,
        data_description=None,
        inputs={},
        target_stats=None,
        fit_extras={},
    )

    assert task.sigma == 0.5
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['sigam'\]"):
        build.task_class_and_options(OmegaConf.create({"_target_": "p.Forked", "sigam": 0.5}))


def _cohort_db(tmp_path: Path) -> tuple[str, str]:
    """A real cohort DB from ``build_db``: 8 patients, a 2 x 2 nested CV on ``cancer_type``."""
    codes = [f"brca{i}" for i in range(4)] + [f"coad{i}" for i in range(4)]
    patients, slides = tmp_path / "patients.csv", tmp_path / "slides.csv"
    patients.write_text("patient_id,cancer_type\n" + "".join(f"{c},{c[:4].upper()}\n" for c in codes))
    slides.write_text("patient_id,image_path\n" + "".join(f"{c},{c}.svs\n" for c in codes))
    contract = {"cancer_type": ContractField.model_validate({"type": "binary", "map": {"BRCA": 0, "COAD": 1}})}
    db = tmp_path / "db"
    build_db(
        Cohort(name="c", storage=Storage(image_dir=str(tmp_path)), contract=contract),
        Splits(strategy=SplitStrategy.nested_cv, n_outer=2, n_inner=2),
        db,
        patients_csv=patients,
        slides_csv=slides,
        probe="none",
        analyze=False,
    )
    return f"sqlite:///{layout.manifest_db(db)}", "c"


def _codes(records):
    return sorted(str(r.patient_code) for r in records)


def test_fit_records_by_codes_matches_the_stored_split_for_the_same_patients(tmp_path):
    """Naming the fit patients directly selects exactly what the stored-split query does, so a fit-time
    statistic does not depend on which route named the set."""
    uri, cohort = _cohort_db(tmp_path)
    stored = _codes(build._fit_records(uri, cohort, "cancer_type_cv_o0_i0"))
    assert 0 < len(stored) < 8
    assert _codes(build._fit_records(uri, cohort, None, stored)) == stored


def test_fit_records_by_codes_ignores_patients_outside_the_given_set(tmp_path):
    uri, cohort = _cohort_db(tmp_path)
    assert _codes(build._fit_records(uri, cohort, None, ["brca1", "coad3", "absent"])) == ["brca1", "coad3"]


def test_fit_records_rejects_both_ways_of_naming_the_fit_set():
    """Both given is ambiguous about which patients a leakage-sensitive statistic covered."""
    with pytest.raises(ValueError, match="exactly one of split_version or fit_patient_codes"):
        build._fit_records("uri", "m", "v_cv_o0_i0", ["p0"])


def test_fit_records_rejects_neither_way_of_naming_the_fit_set():
    with pytest.raises(ValueError, match="exactly one of split_version or fit_patient_codes"):
        build._fit_records("uri", "m", None)
