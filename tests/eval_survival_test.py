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
"""Tests for the survival endpoint (dlux.eval.survival): the coupled (time, event) aggregation core
scoring Harrell's C + its KM-by-risk report artifacts, through the aggregate driver."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from dlux.eval.aggregate import aggregate_experiment, write_reports
from dlux.eval.predictions import write_predictions_npz


def _write_survival_fold(runs_root, outer, patients, *, grid, inner=0):
    """One survival fold: probs = risk (N,), labels = (N, 2) [time, event]. ``patients`` maps
    patient_code -> (time, event, risk). Written with the REAL NPZ writer, sentinel-gated."""
    d = runs_root / f"pfi_cv_o{outer}_i{inner}"
    d.mkdir(parents=True)
    pcs = list(patients)
    risk = np.asarray([patients[p][2] for p in pcs], dtype=np.float64)
    labels = np.asarray([[patients[p][0], patients[p][1]] for p in pcs], dtype=np.float64)  # (N, 2)
    preds = {
        "slide_ids": [f"{p}_s0" for p in pcs],
        "patient_codes": pcs,
        "logits": np.zeros((len(pcs), 4), dtype=np.float64),
        "probs": risk,
        "labels": labels,
        "endpoint_type": "survival",
        "num_classes": 4,
        "target_key": "patient.pfi",
        "time_edges": np.asarray([0.5, 1.5, 2.5]),  # 4 bins -> 3 interior cut points
    }
    write_predictions_npz(preds, d / "test_predictions.npz")
    (d / "metadata.json").write_text(json.dumps({"n_outer": grid[0], "n_inner": grid[1]}))


def test_survival_routing_and_cindex(tmp_path):
    runs = tmp_path / "runs"
    # risk ranks survival perfectly (shorter time -> higher risk) -> C = 1.0 per fold + pooled.
    _write_survival_fold(runs, 0, {"p0": (1.0, 1, 0.9), "p1": (2.0, 1, 0.1)}, grid=(2, 1))
    _write_survival_fold(runs, 1, {"p2": (1.0, 1, 0.8), "p3": (2.0, 1, 0.2)}, grid=(2, 1))
    (res,) = aggregate_experiment(runs, expected_fields=["pfi"], n_outer=2, n_inner=1)
    assert res.endpoint_type == "survival"
    assert res.pooled["c_index"] == pytest.approx(1.0)
    assert res.pooled_n == 4 and res.pooled_positive == 4  # pos = observed events
    assert all(m.metrics["c_index"] == pytest.approx(1.0) for m in res.per_outer)


def test_survival_write_reports(tmp_path):
    runs = tmp_path / "runs"
    _write_survival_fold(runs, 0, {"p0": (1.0, 1, 0.9), "p1": (2.0, 0, 0.1)}, grid=(2, 1))
    _write_survival_fold(runs, 1, {"p2": (1.0, 1, 0.8), "p3": (2.0, 1, 0.2)}, grid=(2, 1))
    results = aggregate_experiment(runs, expected_fields=["pfi"], n_outer=2, n_inner=1)

    out = write_reports(results, "surv", tmp_path / "results")
    assert (out / "figures" / "km_pfi.png").exists()  # KM by risk, not ROC/scatter
    assert (out / "figures" / "c_index_by_fold.png").exists()  # survival per-fold strip
    with (out / "predictions_pfi.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 and {"risk", "time", "event"} <= set(rows[0].keys())
    md = (out / "summary.md").read_text()
    assert "C-index" in md and "(survival)" in md
