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
"""Tests for the regression endpoint (dlux.eval.regression) through the aggregate driver: scalar-target
routing/scoring + its report artifacts. Synthetic folds written with the REAL NPZ writer, sentinel-gated."""

from __future__ import annotations

import json

import numpy as np
import pytest
from dlux.eval.aggregate import aggregate_experiment, write_reports
from dlux.eval.predictions import write_predictions_npz

# Fractional targets in [0,1) on purpose: an int64 cast anywhere in the read/roll path would
# truncate them all to 0 (the leukocyte-fraction bug), which these tests must catch.
_REG_LABELS = {"p0": 0.10, "p1": 0.40, "p2": 0.20, "p3": 0.80}


def _write_regression_fold(runs_root, outer, inner, preds):
    d = runs_root / f"p16_cv_o{outer}_i{inner}"
    d.mkdir(parents=True)
    pcs = list(preds)
    arr = np.asarray([preds[x] for x in pcs], dtype=np.float64)
    lab = np.asarray([_REG_LABELS[x] for x in pcs], dtype=np.float64)
    write_predictions_npz(
        {
            "slide_ids": [f"{x}_s0" for x in pcs],
            "patient_codes": pcs,
            "logits": arr,
            "probs": arr,  # regression: raw predicted value
            "labels": lab,  # float target, NOT class index
            "endpoint_type": "regression",
            "num_classes": 1,
            "target_key": "patient.p16",
        },
        d / "test_predictions.npz",
    )
    (d / "metadata.json").write_text(json.dumps({"n_outer": 2, "n_inner": 1}))


def test_regression_aggregate(tmp_path):
    _write_regression_fold(tmp_path, 0, 0, {"p0": 0.12, "p1": 0.38})
    _write_regression_fold(tmp_path, 1, 0, {"p2": 0.25, "p3": 0.70})
    (results,) = aggregate_experiment(tmp_path, expected_fields=["p16"], n_outer=2, n_inner=1)
    assert results.endpoint_type == "regression"
    assert set(results.per_outer[0].metrics) == {"mae", "rmse", "r2", "pearson", "spearman"}
    # pooled OOF preds [.12,.38,.25,.70] vs labels [.10,.40,.20,.80] -> |err| .02,.02,.05,.10 -> MAE .0475
    assert results.pooled["mae"] == pytest.approx(0.0475)
    # labels must survive as fractions, not be truncated to 0 (would make Spearman NaN, MAE≈mean(pred))
    assert not np.isnan(results.pooled["spearman"])
    assert results.pooled_positive == 0  # not meaningful for regression


def test_regression_write_reports(tmp_path):
    runs = tmp_path / "runs"
    _write_regression_fold(runs, 0, 0, {"p0": 0.12, "p1": 0.38})
    _write_regression_fold(runs, 1, 0, {"p2": 0.25, "p3": 0.70})
    results = aggregate_experiment(runs, expected_fields=["p16"], n_outer=2, n_inner=1)
    out = write_reports(results, "reg", tmp_path / "results")
    assert (out / "figures" / "scatter_p16.png").exists()  # predicted-vs-actual, not ROC
    assert (out / "figures" / "r2_by_fold.png").exists()
    md = (out / "summary.md").read_text()
    assert "R²" in md and "MAE" in md and "pooled OOF" in md and "| pos |" not in md
