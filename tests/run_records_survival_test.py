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
"""Tests for the viewer's survival reading: the per-slide curve, its step semantics, and the
cross-replicate ensemble that has to happen on a common time grid."""

from __future__ import annotations

import numpy as np
import pytest
from dlux.eval.predictions import write_predictions_npz
from dlux.eval.run_records import (
    SurvivalCurve,
    _average_predictions,
    _combine_curves,
    _read_predictions,
    _survival_context,
    survival_at,
)


def _fold_npz(path, edges, logits, labels, slide_ids):
    write_predictions_npz(
        {
            "slide_ids": slide_ids,
            # One patient per slide unless a caller repeats a code, so the patient rollup is exercised.
            "patient_codes": [s.rsplit("/", 1)[0] for s in slide_ids],
            "logits": np.asarray(logits, dtype=np.float64),
            # risk = -sum_j S_j, i.e. the cumulative product of (1 - hazard), as survival_risk computes it
            "probs": -np.cumprod(1.0 - 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64))), axis=1).sum(axis=1),
            "labels": np.asarray(labels, dtype=np.float64),
            "endpoint_type": "survival",
            "num_classes": np.asarray(logits).shape[1],
            "target_key": "patient.os",
            "time_edges": np.asarray(edges, dtype=np.float64),
        },
        path,
    )


def test_curve_reads_hazards_onto_the_persisted_edges(tmp_path):
    path = tmp_path / "p.npz"
    _fold_npz(path, [100.0, 200.0, 300.0], [[0.0, 0.0, 0.0, 0.0]], [[250.0, 1.0]], ["s_0"])
    curve = _read_predictions(path, "os_cv_o0")["s_0"].survival
    assert curve.edges == [100.0, 200.0, 300.0]
    # sigmoid(0) = 0.5 per bin, so S steps 0.5, 0.25, 0.125 and the open last bin lands at 0.0625.
    assert curve.surv == pytest.approx([0.5, 0.25, 0.125])
    assert curve.tail == pytest.approx(0.0625)
    assert len(curve.hazards) == 4 and curve.hazards == pytest.approx([0.5] * 4)


def test_survival_without_edges_is_refused(tmp_path):
    # A pre-edges artifact: hazards with no time axis to place them on.
    path = tmp_path / "old.npz"
    np.savez_compressed(
        path,
        slide_ids=np.array(["s_0"], dtype=object),
        patient_codes=np.array(["s"], dtype=object),
        logits=np.zeros((1, 4)),
        probs=np.array([-1.0]),
        labels=np.array([[10.0, 1.0]]),
        endpoint_type=np.array("survival", dtype=object),
        num_classes=np.array(4),
        target_key=np.array("patient.os", dtype=object),
    )
    with pytest.raises(ValueError, match="without `time_edges`"):
        _read_predictions(path, "os_cv_o0")


def test_survival_at_is_a_right_continuous_step():
    curve = SurvivalCurve(edges=[100.0, 200.0], surv=[0.8, 0.5], hazards=[], tail=0.3)
    # Before the first edge nothing has happened; the step lands ON each edge, not after it.
    assert survival_at(curve, [0.0, 99.9, 100.0, 150.0, 200.0, 1e6]).tolist() == [
        1.0,
        1.0,
        0.8,
        0.8,
        0.5,
        0.5,
    ]


def test_ensemble_averages_on_the_union_grid_not_per_bin():
    # Two replicates with DIFFERENT edges: bin 0 spans 0-100 in one and 0-300 in the other, so their
    # hazards are not comparable index-by-index. On the union grid they are.
    a = SurvivalCurve(edges=[100.0, 200.0], surv=[0.9, 0.8], hazards=[], tail=0.7)
    b = SurvivalCurve(edges=[300.0, 400.0], surv=[0.5, 0.4], hazards=[], tail=0.3)
    combined = _combine_curves([a, b])
    assert combined.edges == [100.0, 200.0, 300.0, 400.0]
    # At t=100: a says 0.9, b still says 1.0 -> 0.95. At t=300: a says 0.8, b says 0.5 -> 0.65.
    assert combined.surv == pytest.approx([0.95, 0.9, 0.65, 0.6])
    assert combined.tail == pytest.approx(0.5)
    # Per-bin hazards are dropped rather than averaged across incomparable bins.
    assert combined.hazards == []


def test_ensemble_of_identical_curves_is_that_curve():
    a = SurvivalCurve(edges=[100.0, 200.0], surv=[0.9, 0.8], hazards=[0.1, 0.2, 0.3, 0.4], tail=0.7)
    combined = _combine_curves([a, a])
    assert combined.edges == a.edges and combined.surv == pytest.approx(a.surv)
    assert combined.tail == pytest.approx(a.tail)


def test_average_predictions_carries_the_ensembled_curve(tmp_path):
    left = tmp_path / "a.npz"
    right = tmp_path / "b.npz"
    _fold_npz(left, [100.0, 200.0, 300.0], [[0.0] * 4], [[250.0, 1.0]], ["s_0"])
    _fold_npz(right, [150.0, 250.0, 350.0], [[0.0] * 4], [[250.0, 1.0]], ["s_0"])
    entries = [_read_predictions(left, "o0")["s_0"], _read_predictions(right, "o0")["s_0"]]
    merged = _average_predictions(entries, "o0")
    assert merged.survival is not None
    assert merged.survival.edges == [100.0, 150.0, 200.0, 250.0, 300.0, 350.0]
    assert merged.survival.surv[0] == pytest.approx(0.75)  # 0.5 from one replicate, 1.0 from the other


def test_survival_context_ranks_risk_and_counts_events(tmp_path):
    path = tmp_path / "p.npz"
    _fold_npz(
        path,
        [100.0, 200.0, 300.0],
        [[-2.0] * 4, [0.0] * 4, [2.0] * 4],
        [[250.0, 1.0], [400.0, 0.0], [50.0, 1.0]],
        ["pA/s0", "pB/s0", "pC/s0"],
    )
    predictions = _read_predictions(path, "o0")
    summary, percentile = _survival_context(predictions)
    assert summary.n_slides == 3 and summary.n_events == 2
    assert summary.median_follow_up == pytest.approx(250.0)
    # Highest hazards -> least survival mass -> highest risk -> out-risks everything else.
    assert percentile["pC/s0"] == pytest.approx(1.0)
    assert percentile["pA/s0"] == pytest.approx(0.0)


def test_risk_rank_is_over_patients_not_slides(tmp_path):
    path = tmp_path / "p.npz"
    # pA has THREE slides at the lowest hazard, pB one at the highest. Ranking slides would put pA's
    # risk below three quarters of the set; ranking patients puts it below the one other patient.
    _fold_npz(
        path,
        [100.0, 200.0, 300.0],
        [[-2.0] * 4, [-2.0] * 4, [-2.0] * 4, [2.0] * 4],
        [[250.0, 1.0]] * 3 + [[50.0, 1.0]],
        ["pA/s0", "pA/s1", "pA/s2", "pB/s0"],
    )
    summary, percentile = _survival_context(_read_predictions(path, "o0"))
    assert summary.n_slides == 4 and summary.n_patients == 2
    # Two patients: the low-risk one out-risks 0%, the high-risk one 100%.
    assert percentile["pA/s0"] == pytest.approx(0.0)
    assert percentile["pB/s0"] == pytest.approx(1.0)
    # Every slide of a patient reports that patient's rank, so the panel cannot disagree with itself.
    assert percentile["pA/s1"] == percentile["pA/s0"] == percentile["pA/s2"]


def test_km_reference_counts_patients_not_slides(tmp_path):
    path = tmp_path / "p.npz"
    # Three slides, TWO patients: pA contributes twice with one (time, event) pair.
    _fold_npz(
        path,
        [100.0, 200.0, 300.0],
        [[0.0] * 4, [0.0] * 4, [1.0] * 4],
        [[250.0, 1.0], [250.0, 1.0], [50.0, 1.0]],
        ["pA/s0", "pA/s1", "pB/s0"],
    )
    summary, _ = _survival_context(_read_predictions(path, "o0"))
    assert summary.n_slides == 3 and summary.n_patients == 2
    # KM over the two distinct patients, non-increasing and starting at 1.
    assert summary.km_surv[0] == pytest.approx(1.0)
    assert summary.km_surv == sorted(summary.km_surv, reverse=True)
    assert summary.km_surv[-1] == pytest.approx(0.0)  # both patients had the event
    assert len(summary.km_times) == len(summary.km_surv)
