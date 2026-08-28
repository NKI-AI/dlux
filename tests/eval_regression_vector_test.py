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
"""Tests for the expression endpoint (dlux.eval.regression_vector, regression_vector): the per-gene aggregation
core (per-gene Pearson, constant-gene exclusion) + its gene-Pearson report artifacts, via the driver."""

from __future__ import annotations

import json

import numpy as np
import pytest
from dlux.eval.aggregate import aggregate_experiment, write_reports
from dlux.eval.predictions import write_predictions_npz


def _write_expr_fold(runs_root, field, outer, inner, patient_vecs, *, grid=None):
    """One expression fold: ``patient_vecs`` maps patient_code -> (preds (G,), labels (G,))."""
    d = runs_root / f"{field}_cv_o{outer}_i{inner}"
    d.mkdir(parents=True)
    pcs = list(patient_vecs)
    preds_arr = np.stack([patient_vecs[p][0] for p in pcs]).astype(np.float64)  # (N, G)
    labels_arr = np.stack([patient_vecs[p][1] for p in pcs]).astype(np.float64)  # (N, G)
    payload = {
        "slide_ids": [f"{p}_s0" for p in pcs],
        "patient_codes": pcs,
        "logits": preds_arr,
        "probs": preds_arr,
        "labels": labels_arr,
        "endpoint_type": "regression_vector",
        "num_classes": preds_arr.shape[1],
        "target_key": f"patient.{field}",
    }
    write_predictions_npz(payload, d / "test_predictions.npz")
    meta = {"n_outer": grid[0], "n_inner": grid[1]} if grid else {}
    (d / "metadata.json").write_text(json.dumps(meta))


def _expr_vecs(patients):
    out = {}
    for i, pc in enumerate(patients):
        label = np.array([float(i), 10.0 - i, 5.0])  # gene0 up, gene1 down, gene2 CONSTANT -> excluded
        pred = np.array([2.0 * i, float(i), 0.0])  # gene0 r=+1, gene1 r=-1, gene2 constant
        out[pc] = (pred, label)
    return out


def test_expression_aggregation_routes_and_scores(tmp_path):
    _write_expr_fold(tmp_path, "expression", 0, 0, _expr_vecs(["p0", "p1", "p2"]), grid=(2, 1))
    _write_expr_fold(tmp_path, "expression", 1, 0, _expr_vecs(["p3", "p4", "p5"]), grid=(2, 1))
    (res,) = aggregate_experiment(tmp_path, expected_fields=["expression"], n_outer=2, n_inner=1)
    assert res.endpoint_type == "regression_vector" and res.pooled_n == 6 and len(res.per_outer) == 2
    r = res.pooled_gene_pearson
    assert r.shape == (3,)
    assert abs(r[0] - 1.0) < 1e-9 and abs(r[1] + 1.0) < 1e-9 and np.isnan(r[2])  # constant gene2 excluded
    assert abs(res.pooled["gene_pearson_mean"]) < 1e-9  # mean of {+1, -1}


def test_expression_reports_run_end_to_end(tmp_path):
    _write_expr_fold(tmp_path, "expression", 0, 0, _expr_vecs(["p0", "p1", "p2"]), grid=(2, 1))
    _write_expr_fold(tmp_path, "expression", 1, 0, _expr_vecs(["p3", "p4", "p5"]), grid=(2, 1))
    out = write_reports(
        aggregate_experiment(tmp_path, expected_fields=["expression"], n_outer=2, n_inner=1),
        "expr",
        tmp_path / "results",
    )
    assert (out / "figures" / "gene_pearson_expression.png").exists()
    assert (out / "figures" / "gene_pearson_by_fold.png").exists()
    assert (out / "per_gene_pearson_expression.csv").exists()
    md = (out / "summary.md").read_text()
    assert "mean gene Pearson" in md and "pooled OOF" in md and "| pos |" not in md  # no positive column


def test_constant_gene_is_undefined_for_spearman_not_a_perfect_rho():
    """A gene constant in both predictions and labels (an unmeasured, all-zero gene) must be EXCLUDED.

    argsort breaks ties by position, so it invents a strictly increasing ranking for a constant column;
    both sides then rank identically and score rho = +1. Panels carry plenty of such genes, so counting
    them inflates the median -- Pearson has always reported them as undefined and Spearman must agree.
    """
    from dlux.eval.gene_metrics import gene_metric_summary, per_gene_pearson, per_gene_spearman

    labels = np.array([[1.0, 5.0, 0.0], [2.0, 4.0, 0.0], [3.0, 3.0, 0.0], [4.0, 2.0, 0.0]])
    preds = np.array([[1.0, 2.0, 7.0], [2.0, 3.0, 7.0], [3.0, 4.0, 7.0], [4.0, 5.0, 7.0]])

    r = per_gene_pearson(preds, labels)
    rho = per_gene_spearman(preds, labels)
    assert np.isnan(r[2]) and np.isnan(rho[2])  # gene 2 is constant on both sides
    assert rho[0] == pytest.approx(1.0) and rho[1] == pytest.approx(-1.0)

    summary = gene_metric_summary(preds, labels, top_k=100)
    assert summary["n_scored"] == 2 and summary["n_genes"] == 3
    assert summary["gene_spearman_median"] == pytest.approx(0.0)  # median(+1, -1), not median(+1, -1, +1)
