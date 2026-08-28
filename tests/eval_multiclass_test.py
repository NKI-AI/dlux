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
"""Tests for the multiclass endpoint (dlux.eval.multiclass): the macro/per-class metric fns directly +
the vector-prob aggregation core + confusion/per-class report artifacts through the aggregate driver."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from dlux.eval.aggregate import aggregate_experiment, write_reports
from dlux.eval.predictions import write_predictions_npz


def _write_multiclass_fold(runs_root, field, outer, inner, patient_probs, patient_labels, *, grid=None):
    """One multiclass fold: ``patient_probs`` maps patient_code -> (K,) softmax vector."""
    d = runs_root / f"{field}_cv_o{outer}_i{inner}"
    d.mkdir(parents=True)
    pcs = list(patient_probs)
    probs = np.stack([np.asarray(patient_probs[p], dtype=np.float64) for p in pcs])  # (N, K)
    labels = np.asarray([patient_labels[p] for p in pcs], dtype=np.int64)
    write_predictions_npz(
        {
            "slide_ids": [f"{p}_s0" for p in pcs],
            "patient_codes": pcs,
            "logits": probs,
            "probs": probs,
            "labels": labels,
            "endpoint_type": "multiclass",
            "num_classes": probs.shape[1],
            "target_key": f"patient.{field}",
        },
        d / "test_predictions.npz",
    )
    meta = {"n_outer": grid[0], "n_inner": grid[1]} if grid else {}
    (d / "metadata.json").write_text(json.dumps(meta))


def _mc(true_class, k=3, hi=0.8):
    """A (K,) softmax vector that argmaxes to ``true_class`` (hi on the true class, rest shared)."""
    v = np.full(k, (1.0 - hi) / (k - 1))
    v[true_class] = hi
    return v


def _mc_labels():
    return {"p0": 0, "p1": 1, "p2": 2, "p3": 0, "p4": 1, "p5": 2}


def _perfect_multiclass_sweep(runs_root):
    """2 outer folds x 2 inner replicates, all three classes per fold, perfectly separated -> every
    metric 1.0. Inner replicates differ slightly so the ensemble mean is exercised."""
    labels = _mc_labels()
    for inner, hi in enumerate((0.8, 0.7)):  # two replicates, same argmax
        _write_multiclass_fold(
            runs_root,
            "cancer_type",
            0,
            inner,
            {"p0": _mc(0, hi=hi), "p1": _mc(1, hi=hi), "p2": _mc(2, hi=hi)},
            labels,
            grid=(2, 2),
        )
        _write_multiclass_fold(
            runs_root,
            "cancer_type",
            1,
            inner,
            {"p3": _mc(0, hi=hi), "p4": _mc(1, hi=hi), "p5": _mc(2, hi=hi)},
            labels,
            grid=(2, 2),
        )


def test_multiclass_routes_and_scores(tmp_path):
    _perfect_multiclass_sweep(tmp_path)
    (res,) = aggregate_experiment(tmp_path, expected_fields=["cancer_type"], n_outer=2, n_inner=2)

    assert res.endpoint_type == "multiclass" and res.num_classes == 3
    assert res.pooled_n == 6 and res.pooled_positive == 0
    assert res.inner_coverage == {0: 2, 1: 2}
    # perfect separation -> every macro metric 1.0, per outer fold and pooled
    for key in ("auroc", "accuracy", "balanced_accuracy", "macro_f1"):
        assert res.pooled[key] == pytest.approx(1.0)
        assert {m.outer_fold: m.metrics[key] for m in res.per_outer} == {0: pytest.approx(1.0), 1: pytest.approx(1.0)}
    # per-class pooled table: 3 classes, support 2 each, all metrics 1.0
    assert [r["class"] for r in res.pooled_per_class] == [0, 1, 2]
    assert all(r["support"] == 2 and r["auroc"] == pytest.approx(1.0) for r in res.pooled_per_class)
    # patient predictions carry the ensembled (K,) prob vector = mean of the two inner replicates
    p0 = next(p for p in res.patient_predictions if p.patient_code == "p0")
    assert p0.ensemble_probs.shape == (3,)
    assert p0.ensemble_probs == pytest.approx((_mc(0, hi=0.8) + _mc(0, hi=0.7)) / 2)
    assert int(np.argmax(p0.ensemble_probs)) == 0


def test_multiclass_metrics_imperfect():
    from dlux.eval.multiclass import multiclass_metrics

    labels = np.array([0, 1, 2, 0])
    probs = np.stack([_mc(0), _mc(1), _mc(2), _mc(1)])  # last (true 0) misclassified as 1
    m = multiclass_metrics(labels, probs, num_classes=3)
    assert m["accuracy"] == pytest.approx(0.75)  # 3/4 argmax-correct
    # class 0: 1 of 2 recalled; class 1,2: fully recalled -> balanced acc = mean(0.5, 1, 1)
    assert m["balanced_accuracy"] == pytest.approx((0.5 + 1.0 + 1.0) / 3)
    assert np.isfinite(m["auroc"]) and np.isfinite(m["macro_f1"])


def test_multiclass_metrics_missing_class_auroc_nan():
    from dlux.eval.multiclass import multiclass_per_class

    labels = np.array([0, 0, 1])  # class 2 absent
    probs = np.stack([_mc(0), _mc(0), _mc(1)])
    rows = multiclass_per_class(labels, probs, num_classes=3)
    assert rows[2]["support"] == 0 and np.isnan(rows[2]["auroc"])  # undefined without any class-2 positive


def test_multiclass_write_reports(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _perfect_multiclass_sweep(runs)
    results = aggregate_experiment(runs, expected_fields=["cancer_type"], n_outer=2, n_inner=2)

    out = write_reports(results, "mc", tmp_path / "results")
    assert (out / "figures" / "confusion_cancer_type.png").exists()  # argmax at a fixed threshold
    assert (out / "figures" / "roc_cancer_type.png").exists()  # one-vs-rest, threshold-free
    assert (out / "figures" / "auroc_by_fold.png").exists()  # classification strip (macro AUROC)
    assert (out / "predictions_cancer_type.csv").exists()  # per-patient per-class probs
    with (out / "predictions_cancer_type.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6 and {"prob_0", "prob_1", "prob_2", "pred", "label"} <= set(rows[0].keys())
    md = (out / "summary.md").read_text()
    assert "macro AUROC" in md and "| class | support |" in md and "| pos |" not in md


def test_ovr_roc_figure_survives_an_absent_class(tmp_path):
    """A class with no positives has no ROC. It must still draw — the figure is the whole endpoint's
    view, so one undefined class cannot take the other two down with it."""
    from dlux.eval.multiclass import _ovr_roc_figure

    labels = np.array([0, 0, 1, 1])  # class 2 never occurs
    probs = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.8, 0.1]])
    path = tmp_path / "roc.png"
    _ovr_roc_figure(labels, probs, 3, "t", path)
    assert path.exists()
