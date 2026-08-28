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
"""Tests for arm comparison: the join, the refusals that keep two arms comparable, and the report."""

from __future__ import annotations

import csv

import numpy as np
import pytest
from dlux.data.errors import BuildDbError
from dlux.eval.compare import binary_auroc, compare_arms, default_metric_key, resolve_metric, write_comparison


def _arm(results_dir, arm, cohort, rows):
    """Write one arm's per_patient_ensemble.csv. ``rows`` is (field, fold, patient, pred, label)."""
    out = results_dir / arm / cohort
    out.mkdir(parents=True)
    with (out / "per_patient_ensemble.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "outer_fold", "patient_code", "ensemble_prediction", "label", "n_replicates"])
        for field, fold, patient, pred, label in rows:
            writer.writerow([field, fold, patient, pred, label, 5])


def _rows(preds, labels, field="er", fold=0):
    return [(field, fold, f"p{i}", p, y) for i, (p, y) in enumerate(zip(preds, labels))]


LABELS = [0, 0, 1, 1]


def test_perfect_and_inverted_arms_score_as_expected(tmp_path):
    _arm(tmp_path, "good", "c", _rows([0.1, 0.2, 0.8, 0.9], LABELS))
    _arm(tmp_path, "bad", "c", _rows([0.9, 0.8, 0.2, 0.1], LABELS))
    (comparison,) = compare_arms(tmp_path, "c", ["good", "bad"], reference="good")
    scores = {s.arm: s.pooled for s in comparison.scores}
    assert scores["good"] == pytest.approx(1.0)
    assert scores["bad"] == pytest.approx(0.0)
    assert comparison.pooled_deltas["bad"] == pytest.approx(-1.0)


def test_delta_is_paired_per_fold(tmp_path):
    rows_a = _rows([0.1, 0.2, 0.8, 0.9], LABELS, fold=0) + _rows([0.1, 0.2, 0.8, 0.9], LABELS, fold=1)
    rows_b = _rows([0.1, 0.2, 0.8, 0.9], LABELS, fold=0) + _rows([0.9, 0.8, 0.2, 0.1], LABELS, fold=1)
    # distinct patient ids per fold
    rows_a = [(f, fo, f"{p}_{fo}", pr, y) for f, fo, p, pr, y in rows_a]
    rows_b = [(f, fo, f"{p}_{fo}", pr, y) for f, fo, p, pr, y in rows_b]
    _arm(tmp_path, "a", "c", rows_a)
    _arm(tmp_path, "b", "c", rows_b)
    (comparison,) = compare_arms(tmp_path, "c", ["a", "b"], reference="a")
    # Fold 0 identical, fold 1 inverted: the delta is per fold, not an average of marginals.
    assert comparison.deltas["b"] == {0: pytest.approx(0.0), 1: pytest.approx(-1.0)}


def test_different_patient_sets_are_intersected_and_coverage_reported(tmp_path):
    """An arm can legitimately score patients another cannot (a finer grid clearing min_tiles). The
    metric is held to the shared set so it describes one population, and the gap is reported."""
    _arm(tmp_path, "a", "c", _rows([0.1, 0.2, 0.8, 0.9], LABELS))
    _arm(tmp_path, "b", "c", _rows([0.1, 0.2, 0.8], LABELS[:3]))
    (comparison,) = compare_arms(tmp_path, "c", ["a", "b"], reference="a")
    assert comparison.n_patients == 3  # scored on the shared set, not either arm's own
    assert comparison.coverage == {"a": 4, "b": 3}
    assert comparison.coverage_differs


def test_no_shared_patients_is_refused(tmp_path):
    """Intersecting to nothing is not a comparison."""
    _arm(tmp_path, "a", "c", [("er", 0, "x0", 0.1, 0), ("er", 0, "x1", 0.9, 1)])
    _arm(tmp_path, "b", "c", [("er", 0, "y0", 0.1, 0), ("er", 0, "y1", 0.9, 1)])
    with pytest.raises(BuildDbError, match="share no patient"):
        compare_arms(tmp_path, "c", ["a", "b"], reference="a")


def test_coverage_is_equal_when_arms_match(tmp_path):
    _arm(tmp_path, "a", "c", _rows([0.1, 0.2, 0.8, 0.9], LABELS))
    _arm(tmp_path, "b", "c", _rows([0.9, 0.8, 0.2, 0.1], LABELS))
    (comparison,) = compare_arms(tmp_path, "c", ["a", "b"], reference="a")
    assert comparison.coverage == {"a": 4, "b": 4} and not comparison.coverage_differs


def test_disagreeing_labels_are_refused(tmp_path):
    _arm(tmp_path, "a", "c", _rows([0.1, 0.2, 0.8, 0.9], LABELS))
    _arm(tmp_path, "b", "c", _rows([0.1, 0.2, 0.8, 0.9], [0, 1, 1, 1]))
    with pytest.raises(BuildDbError, match="disagree on the label"):
        compare_arms(tmp_path, "c", ["a", "b"], reference="a")


def test_unaggregated_arm_is_refused(tmp_path):
    _arm(tmp_path, "a", "c", _rows([0.1, 0.2, 0.8, 0.9], LABELS))
    with pytest.raises(BuildDbError, match="has to be aggregated first"):
        compare_arms(tmp_path, "c", ["a", "missing"], reference="a")


def test_reference_must_be_one_of_the_arms(tmp_path):
    _arm(tmp_path, "a", "c", _rows([0.1, 0.2, 0.8, 0.9], LABELS))
    _arm(tmp_path, "b", "c", _rows([0.1, 0.2, 0.8, 0.9], LABELS))
    with pytest.raises(BuildDbError, match="is not among the arms"):
        compare_arms(tmp_path, "c", ["a", "b"], reference="c")


def test_an_arm_against_itself_is_exactly_zero(tmp_path):
    """The negative control: a comparison of one arm under two names must show no difference at all."""
    rows = _rows([0.1, 0.7, 0.3, 0.9], LABELS)
    _arm(tmp_path, "a", "c", rows)
    _arm(tmp_path, "copy", "c", rows)
    (comparison,) = compare_arms(tmp_path, "c", ["a", "copy"], reference="a")
    assert comparison.pooled_deltas["copy"] == 0.0
    assert set(comparison.deltas["copy"].values()) == {0.0}


def test_arms_are_compared_on_every_shared_endpoint(tmp_path):
    both = _rows([0.1, 0.2, 0.8, 0.9], LABELS, field="er") + _rows([0.1, 0.2, 0.8, 0.9], LABELS, field="her2")
    _arm(tmp_path, "a", "c", both)
    _arm(tmp_path, "b", "c", both)
    comparisons = compare_arms(tmp_path, "c", ["a", "b"], reference="a")
    assert [c.field for c in comparisons] == ["er", "her2"]


def test_metric_is_undefined_for_a_single_class():
    assert np.isnan(binary_auroc(np.array([1.0, 1.0]), np.array([0.3, 0.8])))


def _predictions(results_dir, arm, cohort, field, header, rows, ensemble_rows=None):
    """Write one arm's predictions_<field>.csv, plus the ensemble table compare reads to learn which
    endpoints an arm scored."""
    out = results_dir / arm / cohort
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"predictions_{field}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    with (out / "per_patient_ensemble.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "outer_fold", "patient_code", "ensemble_prediction", "label", "n_replicates"])
        for patient, fold in ensemble_rows or [(r[0], r[1]) for r in rows]:
            writer.writerow([field, fold, patient, 0.0, 0.0, 5])


def test_regression_scores_on_r2(tmp_path):
    """R² reads straight off the ensemble table, and 0 is the mean-predictor rather than a floor."""
    truth = [1.0, 2.0, 3.0, 4.0]
    _arm(tmp_path, "good", "c", _rows(truth, truth, field="lf"))
    _arm(tmp_path, "flat", "c", _rows([2.5] * 4, truth, field="lf"))
    (comparison,) = compare_arms(tmp_path, "c", ["good", "flat"], reference="good", objective="regression")
    scores = {s.arm: s.pooled for s in comparison.scores}
    assert scores["good"] == pytest.approx(1.0)
    assert scores["flat"] == pytest.approx(0.0)  # predicting the mean


def test_multiclass_reads_probabilities_not_the_argmax_class(tmp_path):
    """The ensemble table holds only the argmax class, so a macro AUROC has to come from
    predictions_<field>.csv. A perfect and an inverted arm bracket the metric."""
    header = ["patient_code", "outer_fold", "prob_0", "prob_1", "prob_2", "pred", "label"]
    good = [["p0", 0, 0.8, 0.1, 0.1, 0, 0], ["p1", 0, 0.1, 0.8, 0.1, 1, 1], ["p2", 0, 0.1, 0.1, 0.8, 2, 2]]
    bad = [["p0", 0, 0.1, 0.1, 0.8, 2, 0], ["p1", 0, 0.8, 0.1, 0.1, 0, 1], ["p2", 0, 0.1, 0.8, 0.1, 1, 2]]
    _predictions(tmp_path, "good", "c", "grade", header, good)
    _predictions(tmp_path, "bad", "c", "grade", header, bad)
    (comparison,) = compare_arms(tmp_path, "c", ["good", "bad"], reference="good", objective="multiclass")
    scores = {s.arm: s.pooled for s in comparison.scores}
    assert scores["good"] == pytest.approx(1.0)
    assert scores["bad"] < scores["good"]


def test_resolve_metric_default_selection_and_errors():
    assert default_metric_key("binary") == "auroc"
    assert default_metric_key("multiclass") == "auroc"
    assert resolve_metric("multiclass").key == "auroc"  # None -> endpoint default
    assert resolve_metric("multiclass", "qwk").key == "qwk"
    assert resolve_metric("binary", "auroc").chance == 0.5
    with pytest.raises(BuildDbError, match="not available"):
        resolve_metric("multiclass", "r2")  # a real metric, but not for this objective
    with pytest.raises(BuildDbError, match="objectives"):
        resolve_metric("nonsense")


def test_compare_arms_scores_on_the_selected_metric(tmp_path):
    """A multiclass comparison can be made on QWK instead of the default macro-AUROC."""
    header = ["patient_code", "outer_fold", "prob_0", "prob_1", "prob_2", "pred", "label"]
    good = [["p0", 0, 0.8, 0.1, 0.1, 0, 0], ["p1", 0, 0.1, 0.8, 0.1, 1, 1], ["p2", 0, 0.1, 0.1, 0.8, 2, 2]]
    bad = [["p0", 0, 0.1, 0.1, 0.8, 2, 0], ["p1", 0, 0.8, 0.1, 0.1, 0, 1], ["p2", 0, 0.1, 0.8, 0.1, 1, 2]]
    _predictions(tmp_path, "good", "c", "grade", header, good)
    _predictions(tmp_path, "bad", "c", "grade", header, bad)
    (comparison,) = compare_arms(
        tmp_path, "c", ["good", "bad"], reference="good", objective="multiclass", metric_key="qwk"
    )
    scores = {s.arm: s.pooled for s in comparison.scores}
    assert scores["good"] == pytest.approx(1.0)  # perfect predictions -> QWK 1
    assert scores["bad"] < scores["good"]


def test_survival_needs_the_event_indicator(tmp_path):
    """The ensemble table records time as the label and drops `event`, so a C-index is only
    computable from predictions_<field>.csv. Higher risk on shorter survival is concordant."""
    header = ["patient_code", "outer_fold", "risk", "time", "event"]
    good = [["p0", 0, 2.0, 1.0, 1], ["p1", 0, 1.0, 5.0, 1], ["p2", 0, 0.0, 9.0, 1]]
    bad = [["p0", 0, 0.0, 1.0, 1], ["p1", 0, 1.0, 5.0, 1], ["p2", 0, 2.0, 9.0, 1]]
    _predictions(tmp_path, "good", "c", "os", header, good)
    _predictions(tmp_path, "bad", "c", "os", header, bad)
    (comparison,) = compare_arms(tmp_path, "c", ["good", "bad"], reference="good", objective="survival")
    scores = {s.arm: s.pooled for s in comparison.scores}
    assert scores["good"] == pytest.approx(1.0)
    assert scores["bad"] == pytest.approx(0.0)
    assert comparison.pooled_deltas["bad"] == pytest.approx(-1.0)


def test_missing_predictions_table_is_refused(tmp_path):
    """An arm aggregated before the endpoint's per-prediction table existed must be refused, not
    scored off the ensemble table it cannot support."""
    header = ["patient_code", "outer_fold", "risk", "time", "event"]
    rows = [["p0", 0, 2.0, 1.0, 1], ["p1", 0, 1.0, 5.0, 1]]
    _predictions(tmp_path, "a", "c", "os", header, rows)
    _arm(tmp_path, "b", "c", _rows([0.1, 0.2], [1.0, 5.0], field="os"))
    with pytest.raises(BuildDbError, match="predictions_os.csv"):
        compare_arms(tmp_path, "c", ["a", "b"], reference="a", objective="survival")


def test_report_writes_both_figures_and_the_table(tmp_path):
    _arm(tmp_path, "a", "c", _rows([0.1, 0.2, 0.8, 0.9], LABELS))
    _arm(tmp_path, "b", "c", _rows([0.9, 0.8, 0.2, 0.1], LABELS))
    comparisons = compare_arms(tmp_path, "c", ["a", "b"], reference="a")
    out = tmp_path / "out"
    summary = write_comparison(comparisons, "demo", out)
    assert (out / "metric.png").is_file() and (out / "deltas.png").is_file()
    text = summary.read_text()
    assert "demo" in text and "a" in text and "b" in text
    # The report states what it does NOT compute, so the numbers are not over-read.
    assert "No test, interval or p-value" in text
    with (out / "per_fold.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and {"field", "arm", "outer_fold", "auroc"} <= set(rows[0])
