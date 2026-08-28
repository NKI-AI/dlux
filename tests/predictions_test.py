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
"""Tests for dlux.eval.predictions: the type-aware collector + NPZ/CSV writers.

Fakes a lit_module (real nn.Module for params/device/eval-train) with a duck-typed
_task, and a list-of-dicts "dataloader" — so the collector is exercised without any
Lightning/ahcore machinery.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from dlux.eval.predictions import collect_predictions, write_predictions_csv, write_predictions_npz


@dataclass
class _FakeSlideView:
    slide_id: str
    patient_code: str


class _FakeTask:
    def select_inputs(self, batch):
        return {"features": batch["features"]}

    def select_targets(self, batch):
        return {"label": batch["label"]}


class _FakeModule(torch.nn.Module):
    """Deterministic 'model': logit(s) computed from the features so tests can recompute."""

    def __init__(self, out_dim: int):
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))  # gives .parameters()/.device
        self._out_dim = out_dim
        self._task = _FakeTask()

    def forward(self, inputs):
        base = inputs["features"].mean(dim=-1, keepdim=True)  # (B, 1)
        if self._out_dim == 1:
            return {"logits": base}  # (B, 1) -> binary
        offsets = torch.arange(self._out_dim, dtype=base.dtype).reshape(1, -1)
        return {"logits": base + offsets}  # (B, K)


def _make_batches(labels_per_batch, feat_dim: int = 4):
    torch.manual_seed(0)
    batches = []
    gid = 0
    for labels in labels_per_batch:
        b = len(labels)
        svs = [_FakeSlideView(slide_id=f"s{gid + i}", patient_code=f"p{gid + i}") for i in range(b)]
        batches.append({"features": torch.randn(b, feat_dim), "label": torch.tensor(labels), "slide_view": svs})
        gid += b
    return batches


def _binary_preds():
    mod = _FakeModule(out_dim=1)
    batches = _make_batches([[0, 1, 0], [1, 1]])  # N=5 across 2 batches
    preds = collect_predictions(mod, batches, endpoint_type="binary", num_classes=1, target_key="patient.cancer_type")
    return mod, batches, preds


def test_binary_collect_shapes_and_order():
    _, _, preds = _binary_preds()
    assert preds["slide_ids"] == ["s0", "s1", "s2", "s3", "s4"]
    assert preds["patient_codes"] == ["p0", "p1", "p2", "p3", "p4"]
    assert preds["logits"].shape == (5,)
    assert preds["probs"].shape == (5,)
    assert preds["labels"].tolist() == [0, 1, 0, 1, 1]
    assert np.allclose(preds["probs"], 1 / (1 + np.exp(-preds["logits"])))
    assert (preds["endpoint_type"], preds["num_classes"], preds["target_key"]) == (
        "binary",
        1,
        "patient.cancer_type",
    )


def test_train_eval_mode_restored():
    mod, batches, _ = _binary_preds()
    # default nn.Module starts training=True -> must be restored True
    assert mod.training is True
    mod.eval()
    collect_predictions(mod, batches, endpoint_type="binary", num_classes=1, target_key="patient.cancer_type")
    assert mod.training is False


class _NoLabelTask:
    """A task whose select_targets RAISES — an unlabeled predict cohort has no labels to select."""

    def select_inputs(self, batch):
        return {"features": batch["features"]}

    def select_targets(self, batch):
        raise AssertionError("select_targets must not be called when require_labels=False")


def test_labels_optional_matches_labeled_and_skips_targets():
    # The invariant bin/predict relies on: unlabeled inference is the labeled pass minus labels. Scoring
    # the same model+batches with require_labels=False must (a) never touch select_targets, (b) return
    # labels=None, and (c) leave every other output identical.
    mod = _FakeModule(out_dim=1)
    batches = _make_batches([[0, 1, 0], [1, 1]])
    labeled = collect_predictions(mod, batches, endpoint_type="binary", num_classes=1, target_key="patient.x")
    mod._task = _NoLabelTask()  # the raising select_targets proves require_labels=False never calls it
    unlabeled = collect_predictions(
        mod, batches, endpoint_type="binary", num_classes=1, target_key="patient.x", require_labels=False
    )
    assert unlabeled["labels"] is None
    assert unlabeled["slide_ids"] == labeled["slide_ids"]
    assert unlabeled["patient_codes"] == labeled["patient_codes"]
    assert np.array_equal(unlabeled["logits"], labeled["logits"])
    assert np.array_equal(unlabeled["probs"], labeled["probs"])


def test_npz_roundtrip(tmp_path):
    _, _, preds = _binary_preds()
    path = tmp_path / "b.npz"
    write_predictions_npz(preds, path)
    with np.load(path, allow_pickle=True) as d:
        assert list(map(str, d["slide_ids"])) == preds["slide_ids"]
        assert list(map(str, d["patient_codes"])) == preds["patient_codes"]
        assert np.allclose(d["probs"], preds["probs"])
        assert d["labels"].tolist() == preds["labels"].tolist()
        assert str(d["endpoint_type"]) == "binary"
        assert int(d["num_classes"]) == 1
        assert str(d["target_key"]) == "patient.cancer_type"


def test_csv_binary(tmp_path):
    _, _, preds = _binary_preds()
    path = tmp_path / "b.csv"
    write_predictions_csv(preds, path)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    assert set(rows[0].keys()) == {"slide_id", "patient_code", "logit", "prob", "label"}
    assert rows[0]["patient_code"] == "p0"
    assert abs(float(rows[0]["prob"]) - preds["probs"][0]) < 1e-4


def test_multiclass_collect_and_npz(tmp_path):
    mod = _FakeModule(out_dim=3)
    preds = collect_predictions(
        mod, _make_batches([[0, 2], [1]]), endpoint_type="multiclass", num_classes=3, target_key="patient.tissue"
    )
    assert preds["logits"].shape == (3, 3)
    assert preds["probs"].shape == (3, 3)
    assert np.allclose(preds["probs"].sum(axis=1), 1.0)
    path = tmp_path / "m.npz"
    write_predictions_npz(preds, path)
    with np.load(path, allow_pickle=True) as d:
        assert d["probs"].shape == (3, 3)
        assert str(d["endpoint_type"]) == "multiclass"


def test_multiclass_csv(tmp_path):
    import csv

    mod = _FakeModule(out_dim=3)
    preds = collect_predictions(
        mod, _make_batches([[0, 1, 2]]), endpoint_type="multiclass", num_classes=3, target_key="patient.tissue"
    )
    path = tmp_path / "m.csv"
    write_predictions_csv(preds, path)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert {"prob_0", "prob_1", "prob_2", "pred", "label"} <= set(rows[0].keys())
    # pred = argmax over the per-class prob columns
    for r in rows:
        probs = [float(r[f"prob_{c}"]) for c in range(3)]
        assert int(r["pred"]) == int(np.argmax(probs))


def test_unsupported_endpoint_raises():
    mod, batches, _ = _binary_preds()
    with pytest.raises(ValueError):
        collect_predictions(mod, batches, endpoint_type="ordinal", num_classes=1, target_key="patient.event")


def test_regression_collect_and_npz(tmp_path):
    mod = _FakeModule(out_dim=1)
    batches = _make_batches([[1.0, 0.0], [3.5]])  # float targets, N=3
    preds = collect_predictions(mod, batches, endpoint_type="regression", num_classes=1, target_key="patient.p16")
    assert preds["logits"].shape == (3,) and preds["probs"].shape == (3,)
    assert np.allclose(preds["probs"], preds["logits"])  # regression: raw value, no activation
    assert preds["labels"].dtype == np.float64 and preds["labels"].tolist() == [1.0, 0.0, 3.5]
    path = tmp_path / "r.npz"
    write_predictions_npz(preds, path)
    with np.load(path, allow_pickle=True) as d:
        assert d["labels"].dtype == np.float64 and str(d["endpoint_type"]) == "regression"


def test_csv_regression(tmp_path):
    mod = _FakeModule(out_dim=1)
    preds = collect_predictions(
        mod, _make_batches([[1.5, 0.0], [3.5]]), endpoint_type="regression", num_classes=1, target_key="patient.p16"
    )
    path = tmp_path / "r.csv"
    write_predictions_csv(preds, path)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert set(rows[0].keys()) == {"slide_id", "patient_code", "pred", "label"}
    assert float(rows[2]["label"]) == 3.5  # float label preserved, not truncated to int


class _FakeSurvivalTask:
    """Duck-types the survival task's target seam: the coupled (time, event) pair, no class label."""

    def select_inputs(self, batch):
        return {"features": batch["features"]}

    def select_targets(self, batch):
        return {"time": batch["label"][:, 0], "event": batch["label"][:, 1]}


def _survival_preds(edges=(183.0, 402.0, 1094.0), n_bins=4):
    mod = _FakeModule(out_dim=n_bins)
    mod._task = _FakeSurvivalTask()
    batches = _make_batches([[[100.0, 1.0], [900.0, 0.0]], [[500.0, 1.0]]])  # [time, event] per slide
    return collect_predictions(
        mod,
        batches,
        endpoint_type="survival",
        num_classes=n_bins,
        target_key="patient.os_days",
        time_edges=None if edges is None else np.asarray(edges),
    )


def test_survival_collect_carries_edges_and_hazard_logits():
    preds = _survival_preds()
    assert preds["logits"].shape == (3, 4)  # one hazard logit per time bin, kept per slide
    assert preds["probs"].shape == (3,)  # the scalar ranking risk
    assert preds["labels"].shape == (3, 2)  # coupled [time, event]
    assert preds["time_edges"].tolist() == [183.0, 402.0, 1094.0]
    # risk = -sum_j S_j, so it lies in [-n_bins, 0] and ranks shorter predicted survival higher.
    assert np.all((preds["probs"] > -4.0) & (preds["probs"] < 0.0))


def test_survival_edge_count_must_match_head_width():
    with pytest.raises(ValueError, match="interior bin edges"):
        _survival_preds(edges=(183.0, 402.0))  # 2 edges for a 4-bin head


def test_survival_npz_roundtrip(tmp_path):
    preds = _survival_preds()
    path = tmp_path / "s.npz"
    write_predictions_npz(preds, path)
    with np.load(path, allow_pickle=True) as d:
        assert d["logits"].shape == (3, 4) and d["probs"].shape == (3,)
        assert d["time_edges"].tolist() == [183.0, 402.0, 1094.0]
        assert str(d["endpoint_type"]) == "survival"


def test_survival_writers_refuse_unlabelled_hazards(tmp_path):
    preds = _survival_preds(edges=None)  # the inference-only construction: no fit split, no edges
    assert preds["time_edges"] is None
    with pytest.raises(ValueError, match="time_edges"):
        write_predictions_npz(preds, tmp_path / "s.npz")
    with pytest.raises(ValueError, match="time_edges"):
        write_predictions_csv(preds, tmp_path / "s.csv")


def test_survival_csv_names_columns_by_interval(tmp_path):
    preds = _survival_preds()
    path = tmp_path / "s.csv"
    write_predictions_csv(preds, path)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    haz = ["h_0-183", "h_183-402", "h_402-1094", "h_1094+"]
    assert set(haz) <= set(rows[0].keys())
    assert {c.replace("h_", "S_", 1) for c in haz} <= set(rows[0].keys())
    assert float(rows[0]["time"]) == 100.0 and int(rows[0]["event"]) == 1
    # S is the running product of (1 - h) over the bins, so it decreases and matches the hazards.
    hazards = [float(rows[0][c]) for c in haz]
    surv = [float(rows[0][c.replace("h_", "S_", 1)]) for c in haz]
    assert surv == sorted(surv, reverse=True)
    assert surv[1] == pytest.approx((1 - hazards[0]) * (1 - hazards[1]), abs=1e-4)
