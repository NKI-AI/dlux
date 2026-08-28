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
"""Tests for dlux.eval.external: grand K×J ensemble, per-outer stability, guards, report."""

from __future__ import annotations

import csv

import pytest
from dlux.config.cohort import Objective
from dlux.eval.external import (
    external_ensemble,
    supported_objectives,
    uses_vector_labels,
    uses_vector_probs,
    write_external_report,
)

_LABELS = {"p0": 1, "p1": 0, "p2": 1, "p3": 0}


def _per_model():
    """2 outer × 2 inner over the same 4 external patients. outer0 ranks correctly (AUROC 1.0),
    outer1 ranks inversely (AUROC 0.0); the grand mean of all four cancels to 0.5 everywhere."""
    return {
        (0, 0): {"p0": 0.9, "p1": 0.1, "p2": 0.8, "p3": 0.2},
        (0, 1): {"p0": 0.7, "p1": 0.3, "p2": 0.6, "p3": 0.4},
        (1, 0): {"p0": 0.1, "p1": 0.9, "p2": 0.2, "p3": 0.8},
        (1, 1): {"p0": 0.3, "p1": 0.7, "p2": 0.4, "p3": 0.6},
    }


def test_counts_and_grand_ensemble():
    r = external_ensemble("er", "val_cohort", _per_model(), _LABELS)
    assert (r.n_models, r.n_patients, r.n_positive) == (4, 4, 2)
    # grand prob = mean over all 4 models per patient; here every patient averages to 0.5
    assert r.grand_patient_probs == {"p0": 0.5, "p1": 0.5, "p2": 0.5, "p3": 0.5}
    assert r.grand["auroc"] == pytest.approx(0.5)  # tied scores -> chance


def test_per_outer_stability():
    r = external_ensemble("er", "val_cohort", _per_model(), _LABELS)
    # per-outer ensemble = mean over that outer's inner models
    assert r.per_outer_patient_probs[0]["p0"] == pytest.approx(0.8)  # mean(0.9, 0.7)
    assert r.per_outer_patient_probs[1]["p0"] == pytest.approx(0.2)  # mean(0.1, 0.3)
    assert {round(m["auroc"], 3) for m in r.per_outer} == {1.0, 0.0}  # o0 perfect, o1 inverted
    assert r.per_outer_mean["auroc"] == pytest.approx(0.5)
    assert r.per_outer_std["auroc"] == pytest.approx(0.5)


def test_sentinel_label_raises_not_dropped():
    # A missing label must NOT be silently excluded here — the all_test_<target> split is supposed to
    # have filtered it before inference, so its presence is a loud error. For classification "missing"
    # is a negative sentinel; each endpoint defines its own rule (see the survival case below).
    per_model = {(0, 0): {"p0": 0.9, "p1": 0.1}}
    with pytest.raises(ValueError, match="missing label"):
        external_ensemble("er", "val_cohort", per_model, {"p0": 1, "p1": -1})


def test_model_missing_a_patient_raises():
    per_model = {(0, 0): {"p0": 0.9}}  # p1 unscored
    with pytest.raises(ValueError, match="missing scored patients"):
        external_ensemble("er", "val_cohort", per_model, {"p0": 1, "p1": 0})


def test_empty_per_model_raises():
    with pytest.raises(ValueError, match="no development models"):
        external_ensemble("er", "val_cohort", {}, {"p0": 1})


def test_no_patients_to_score_raises():
    with pytest.raises(ValueError, match="no patients"):
        external_ensemble("er", "val_cohort", {(0, 0): {}}, {})


def test_write_report_produces_csv_md_and_figures(tmp_path):
    r = external_ensemble("er", "val_cohort", _per_model(), _LABELS)
    out = write_external_report([r], "dev", tmp_path / "external")

    assert (out / "external_summary.md").exists()
    assert (out / "figures" / "roc_er.png").exists()
    assert (out / "figures" / "auroc_by_outer.png").exists()  # one combined strip, not per-field

    # long-format per-patient CSV: field column + one row per (patient × model), models = o0/o1/grand
    with (out / "per_patient_external.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == {"field", "patient_code", "model", "prob", "label"}
    assert {row["model"] for row in rows} == {"o0", "o1", "grand"}
    assert {row["field"] for row in rows} == {"er"}
    assert len(rows) == 4 * 3  # 4 patients × (2 outer + grand)
    grand_p0 = next(row for row in rows if row["patient_code"] == "p0" and row["model"] == "grand")
    assert float(grand_p0["prob"]) == pytest.approx(0.5) and grand_p0["label"] == "1"


def test_write_report_multi_field(tmp_path):
    # Auto-discovered multi-endpoint report: a section + ROC per field, one combined AUROC strip.
    er = external_ensemble("er", "val_cohort", _per_model(), _LABELS)
    her2 = external_ensemble("her2", "val_cohort", _per_model(), _LABELS)
    out = write_external_report([er, her2], "dev", tmp_path / "external")

    assert (out / "figures" / "roc_er.png").exists() and (out / "figures" / "roc_her2.png").exists()
    assert (out / "figures" / "auroc_by_outer.png").exists()  # single combined strip across fields
    md = (out / "external_summary.md").read_text()
    assert "## `er`" in md and "## `her2`" in md
    with (out / "per_patient_external.csv").open() as f:
        assert {row["field"] for row in csv.DictReader(f)} == {"er", "her2"}


# -- multiclass external scoring ---------------------------------------------
_MC_LABELS = {"p0": 0, "p1": 1, "p2": 2, "p3": 1}


def _mc_per_model():
    """2 outer × 1 inner over 4 external patients, K=3. outer0 puts the mass on the true class for every
    patient (perfect argmax); outer1 is confidently wrong on p0 and p2. Probs are (K,) vectors."""
    o0 = {
        "p0": [0.8, 0.1, 0.1],
        "p1": [0.1, 0.8, 0.1],
        "p2": [0.1, 0.1, 0.8],
        "p3": [0.2, 0.7, 0.1],
    }
    o1 = {
        "p0": [0.1, 0.1, 0.8],  # wrong (true 0)
        "p1": [0.2, 0.6, 0.2],
        "p2": [0.7, 0.2, 0.1],  # wrong (true 2)
        "p3": [0.1, 0.8, 0.1],
    }
    return {(0, 0): o0, (1, 0): o1}


def _mc_result():
    return external_ensemble(
        "grade", "external", _mc_per_model(), _MC_LABELS, endpoint_type=Objective.multiclass, num_classes=3
    )


def test_multiclass_is_externally_scorable():
    # The capability must not depend on us shipping an example multiclass external study.
    assert "multiclass" in supported_objectives()
    assert uses_vector_probs(Objective.multiclass) and not uses_vector_probs(Objective.binary)


def test_multiclass_grand_ensemble_averages_vectors():
    r = _mc_result()
    assert (r.n_models, r.n_patients) == (2, 4)
    assert r.n_positive == 0  # "positive" is binary-only; a K-class endpoint reports no positive count
    # grand prob = elementwise mean of the two outer vectors
    assert list(r.grand_patient_probs["p0"]) == pytest.approx([0.45, 0.1, 0.45])
    assert list(r.grand_patient_probs["p1"]) == pytest.approx([0.15, 0.7, 0.15])


def test_multiclass_per_outer_accuracy_splits():
    r = _mc_result()
    accs = [m["accuracy"] for m in r.per_outer]
    assert accs == pytest.approx([1.0, 0.5])  # o0 argmax-perfect; o1 wrong on p0 and p2


def test_multiclass_report_writes_confusion_per_class_table_and_csv(tmp_path):
    out = write_external_report([_mc_result()], "dev", tmp_path / "external")

    # figure is a confusion matrix (not a ROC), and the combined strip still spans endpoint kinds
    assert (out / "figures" / "confusion_grade.png").exists()
    assert (out / "figures" / "auroc_by_outer.png").exists()

    md = (out / "external_summary.md").read_text()
    assert "## `grade`" in md
    assert "Per-class (GRAND ensemble, one-vs-rest" in md  # section_extra rendered
    assert "macro AUROC" in md  # multiclass column labels, not binary's

    # per-field CSV (class count varies by endpoint, so columns cannot be shared across fields)
    with (out / "per_patient_external_grade.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == {"patient_code", "model", "prob_0", "prob_1", "prob_2", "pred", "label"}
    assert {row["model"] for row in rows} == {"o0", "o1", "grand"}
    assert len(rows) == 4 * 3  # 4 patients × (2 outer + grand)
    grand_p1 = next(r for r in rows if r["patient_code"] == "p1" and r["model"] == "grand")
    assert grand_p1["pred"] == "1" and grand_p1["label"] == "1"


# -- survival external scoring -----------------------------------------------
# Coupled (time, event) label: the axis multiclass does NOT exercise, since multiclass has vector
# predictions with a scalar label and survival is the other way round.
_SURV_LABELS = {"p0": [5.0, 1.0], "p1": [20.0, 1.0], "p2": [12.0, 0.0], "p3": [30.0, 1.0]}


def _surv_per_model():
    """2 outer x 1 inner, scalar risk per patient. o0 ranks risk inversely to survival time (perfect
    concordance, C=1.0); o1 is exactly reversed (C=0.0)."""
    return {
        (0, 0): {"p0": 0.9, "p1": 0.4, "p2": 0.6, "p3": 0.1},
        (1, 0): {"p0": 0.1, "p1": 0.6, "p2": 0.4, "p3": 0.9},
    }


def _surv_result():
    return external_ensemble(
        "os", "external", _surv_per_model(), _SURV_LABELS, endpoint_type=Objective.survival, num_classes=1
    )


def test_survival_is_externally_scorable_with_a_vector_label():
    assert "survival" in supported_objectives()
    # the two shape axes are independent -- this is the case that proves they had to be split
    assert uses_vector_labels(Objective.survival) and not uses_vector_probs(Objective.survival)
    assert uses_vector_probs(Objective.multiclass) and not uses_vector_labels(Objective.multiclass)


def test_survival_counts_events_not_label_sum():
    r = _surv_result()
    # "positive" is the observed-event count (column 1), NOT the sum of the labels, which would be
    # nonsense here (it would add up follow-up times).
    assert (r.n_patients, r.n_positive) == (4, 3)


def test_survival_c_index_per_outer_and_grand():
    r = _surv_result()
    assert [m["c_index"] for m in r.per_outer] == pytest.approx([1.0, 0.0])  # o0 perfect, o1 inverted
    assert r.grand["c_index"] == pytest.approx(0.5)  # risks cancel to a constant -> ties -> chance


def test_survival_missing_label_is_nan_time_not_a_negative():
    # A negative VALUE is not the survival sentinel -- an absent follow-up time is.
    per_model = {(0, 0): {"p0": 0.9, "p1": 0.1}}
    with pytest.raises(ValueError, match="missing label"):
        external_ensemble(
            "os",
            "external",
            per_model,
            {"p0": [5.0, 1.0], "p1": [float("nan"), 0.0]},
            endpoint_type=Objective.survival,
        )


def test_survival_report_writes_km_and_long_csv(tmp_path):
    out = write_external_report([_surv_result()], "dev", tmp_path / "external")

    assert (out / "figures" / "km_os.png").exists()  # KM, not ROC or confusion
    md = (out / "external_summary.md").read_text()
    assert "## `os`" in md and "C-index" in md
    assert "4 patients (3 positive)" in md  # events surfaced in the header line

    with (out / "per_patient_external_survival.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == {"field", "patient_code", "model", "risk", "time", "event"}
    assert len(rows) == 4 * 3  # 4 patients x (2 outer + grand)
    p2 = next(r for r in rows if r["patient_code"] == "p2" and r["model"] == "grand")
    assert float(p2["time"]) == pytest.approx(12.0) and p2["event"] == "0"  # censored patient round-trips


# -- regression external scoring ---------------------------------------------
# Raw continuous targets, INCLUDING a negative one: the axis binary/multiclass cannot exercise, since
# their missing-label rule is "negative".
_REG_LABELS = {"p0": -0.5, "p1": 0.0, "p2": 1.5, "p3": 3.0}


def _reg_per_model():
    """2 outer x 1 inner; o0 predicts the target exactly, o1 is offset by a constant +1.0."""
    return {
        (0, 0): {"p0": -0.5, "p1": 0.0, "p2": 1.5, "p3": 3.0},
        (1, 0): {"p0": 0.5, "p1": 1.0, "p2": 2.5, "p3": 4.0},
    }


def _reg_result():
    return external_ensemble(
        "leukocyte_fraction", "external", _reg_per_model(), _REG_LABELS, endpoint_type=Objective.regression
    )


def test_regression_negative_target_is_data_not_a_missing_sentinel():
    # THE regression-specific guard: p0 = -0.5 is a legitimate target. Under the categorical rule
    # ("negative means missing") this whole cohort would be rejected as corrupt.
    r = _reg_result()
    assert r.n_patients == 4
    assert r.labels["p0"] == pytest.approx(-0.5)
    assert r.n_positive == 0  # a continuous target has no "positive" count


def test_regression_missing_is_nan():
    per_model = {(0, 0): {"p0": 0.9, "p1": 0.1}}
    with pytest.raises(ValueError, match="missing label"):
        external_ensemble(
            "leukocyte_fraction",
            "external",
            per_model,
            {"p0": 0.5, "p1": float("nan")},
            endpoint_type=Objective.regression,
        )


def test_regression_metrics_and_ensembling():
    r = _reg_result()
    # grand = mean of the two models -> a constant +0.5 offset from truth
    assert r.grand_patient_probs["p0"] == pytest.approx(0.0)
    assert r.per_outer[0]["r2"] == pytest.approx(1.0)  # o0 is exact
    assert r.per_outer[0]["mae"] == pytest.approx(0.0)
    assert r.per_outer[1]["mae"] == pytest.approx(1.0)  # o1 is off by exactly 1.0
    assert r.grand["spearman"] == pytest.approx(1.0)  # a constant offset preserves ranking


def test_regression_report_writes_scatter_and_no_shared_strip(tmp_path):
    out = write_external_report([_reg_result()], "dev", tmp_path / "external")

    assert (out / "figures" / "scatter_leukocyte_fraction.png").exists()
    # Regression opts OUT of the cross-field strip: that figure frames [0,1] scores against a 0.5
    # chance line, which is meaningless for R²/MAE.
    assert not (out / "figures" / "auroc_by_outer.png").exists()
    assert not (out / "figures" / "r2_by_outer.png").exists()

    with (out / "per_patient_external_regression.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == {"field", "patient_code", "model", "predicted", "actual"}
    assert len(rows) == 4 * 3
    p0 = next(r for r in rows if r["patient_code"] == "p0" and r["model"] == "grand")
    assert float(p0["actual"]) == pytest.approx(-0.5)  # negative target survives the round-trip


def test_binary_and_multiclass_in_one_report(tmp_path):
    # Mixed-endpoint report: each kind's handler owns its CSV strategy, so the two must not collide.
    out = write_external_report(
        [external_ensemble("er", "external", _per_model(), _LABELS), _mc_result()], "dev", tmp_path / "external"
    )
    assert (out / "per_patient_external.csv").exists()  # binary's combined long CSV
    assert (out / "per_patient_external_grade.csv").exists()  # multiclass's per-field CSV
    assert (out / "figures" / "roc_er.png").exists() and (out / "figures" / "confusion_grade.png").exists()


# -- regression_vector (expression) external scoring --------------------------
# The only endpoint whose predictions AND labels are both vectors -- the combination no other
# endpoint exercises.
_EXPR_LABELS = {
    "p0": [1.0, 5.0, 0.0],
    "p1": [2.0, 4.0, 0.0],
    "p2": [3.0, 3.0, 0.0],
    "p3": [4.0, 2.0, 0.0],
}


def _expr_per_model():
    """2 outer x 1 inner, 3 genes. Gene 0 is predicted in the right direction, gene 1 inverted, gene 2 is
    constant in the labels (so its correlation is undefined and it must be excluded, not counted as 0)."""
    o0 = {"p0": [1.0, 2.0, 7.0], "p1": [2.0, 3.0, 7.0], "p2": [3.0, 4.0, 7.0], "p3": [4.0, 5.0, 7.0]}
    o1 = {"p0": [1.5, 2.5, 7.0], "p1": [2.5, 3.5, 7.0], "p2": [3.5, 4.5, 7.0], "p3": [4.5, 5.5, 7.0]}
    return {(0, 0): o0, (1, 0): o1}


def _expr_result():
    return external_ensemble(
        "expression", "external", _expr_per_model(), _EXPR_LABELS, endpoint_type=Objective.regression_vector
    )


def test_expression_is_the_both_vectors_case():
    assert "regression_vector" in supported_objectives()
    assert uses_vector_probs(Objective.regression_vector) and uses_vector_labels(Objective.regression_vector)
    assert supported_objectives() == ["binary", "multiclass", "regression", "regression_vector", "survival"]


def test_expression_grand_ensemble_averages_gene_vectors():
    r = _expr_result()
    assert (r.n_models, r.n_patients, r.n_positive) == (2, 4, 0)
    assert list(r.grand_patient_probs["p0"]) == pytest.approx([1.25, 2.25, 7.0])


def test_expression_per_gene_correlations_exclude_undefined_genes():
    r = _expr_result()
    # gene 0 tracks its label (r=+1), gene 1 is exactly inverted (r=-1), gene 2 is constant -> excluded.
    # So the mean over SCORED genes is 0, not 1/3 of something that counted the constant gene as zero.
    assert r.grand["gene_pearson_mean"] == pytest.approx(0.0)
    assert r.grand["gene_spearman_median"] == pytest.approx(0.0)


def test_expression_report_writes_per_gene_csv_and_histogram(tmp_path):
    out = write_external_report([_expr_result()], "dev", tmp_path / "external")

    assert (out / "figures" / "gene_pearson_expression.png").exists()
    assert not (out / "figures" / "auroc_by_outer.png").exists()  # opts out of the shared strip

    # per-GENE, not per-patient: a per-patient row would be a 20k-gene vector.
    with (out / "per_gene_external_expression.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert set(rows[0].keys()) == {"gene_index", "grand_pearson"}
    assert len(rows) == 3
    assert float(rows[0]["grand_pearson"]) == pytest.approx(1.0)  # best-predicted gene sorts first
    assert rows[-1]["grand_pearson"] == ""  # the undefined (constant) gene is blank, not 0.0


def test_expression_report_shows_significant_gene_count_only_when_a_random_arm_was_scored(tmp_path):
    # No random arm (conservative is None) -> the null line is absent.
    plain = write_external_report([_expr_result()], "dev", tmp_path / "plain")
    assert "Well-predicted genes" not in (plain / "external_summary.md").read_text()
    # With a trained-vs-random count attached -> the line renders, with the df=n-3 underpowered caveat.
    res = _expr_result()
    res.conservative = {"n_significant": 1588, "n_scored": 17485, "alpha": 0.05, "fdr": 0.2}
    withnull = write_external_report([res], "dev", tmp_path / "withnull")
    text = (withnull / "external_summary.md").read_text()
    assert "1588 / 17485" in text and "df = n" in text
