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
"""Tests for dlux.data.splits: bin_continuous, facet-aware strata, nested_cv, simple, all_*."""

from __future__ import annotations

import pytest
from dlux.config.cohort import Cohort, Splits, SplitStrategy, Storage, StratifyMethod
from dlux.data.splits import (
    FIT,
    PREDICT,
    TEST,
    SplitVersion,
    _valid_patients_and_strata,
    bin_continuous,
    export_splits_csv,
    format_cv_split,
    generate_splits,
    import_splits_csv,
    parse_cv_split,
    valid_patients_by_field,
    validate_imported_splits,
)


def _cohort(contract: dict) -> Cohort:
    return Cohort(name="c", storage=Storage(image_dir="/i"), contract=contract)


def _splits(strategy: SplitStrategy, **split_kw) -> Splits:
    return Splits(strategy=strategy, **split_kw)


_BINARY = {"cancer_type": {"type": "binary", "map": {"BRCA": 0, "COAD": 1}}}


def _binary_labels(n_per_class: int) -> dict[str, dict[str, str]]:
    labels = {}
    for i in range(n_per_class):
        labels[f"brca{i}"] = {"cancer_type": "BRCA"}
        labels[f"coad{i}"] = {"cancer_type": "COAD"}
    return labels


# -- bin_continuous (the shared math) ----------------------------------------
def test_bin_continuous_methods_agree_at_median():
    vals = [1.0, 2.0, 3.0, 4.0]  # median 2.5
    med = bin_continuous(vals, method=StratifyMethod.median)
    q2 = bin_continuous(vals, method=StratifyMethod.quantile, k=2)
    thr = bin_continuous(vals, method=StratifyMethod.threshold, edges=[2.5])
    assert med == q2 == thr == [0, 0, 1, 1]  # binarize (k=2) == stratify median == threshold


def test_bin_continuous_quantile_k4():
    strata = bin_continuous([1, 2, 3, 4, 5, 6, 7, 8], method=StratifyMethod.quantile, k=4)
    assert sorted(set(strata)) == [0, 1, 2, 3]  # four bins


# -- fold-string helpers -----------------------------------------------------
def test_cv_split_string_round_trips():
    assert format_cv_split("cancer_type", 0, 3) == "cancer_type_cv_o0_i3"
    assert parse_cv_split("er_status_cv_o2_i1") == ("er_status", 2, 1)
    assert parse_cv_split("all_test_cancer_type") is None


# -- facet-aware strata ------------------------------------------------------
def test_strata_categorical_is_class_index():
    cohort = _cohort(_BINARY)
    codes, strata = _valid_patients_and_strata(cohort.contract["cancer_type"], _binary_labels(3))
    assert len(codes) == 6
    assert set(strata) == {0, 1}  # BRCA->0, COAD->1


def test_strata_binarize_target_is_median_split():
    contract = {
        "cnh_high": {
            "source": {"column": "CNH", "transform": {"binarize": {"method": "median"}}},
            "objective": "binary",
        }
    }
    cohort = _cohort(contract)
    labels = {f"p{i}": {"CNH": str(float(i))} for i in range(6)}  # 0..5, median 2.5
    codes, strata = _valid_patients_and_strata(cohort.contract["cnh_high"], labels)
    assert len(codes) == 6
    assert strata == [0, 0, 0, 1, 1, 1]  # binarized on the cohort median


def test_strata_override_column():
    # target = continuous score; stratify on a separate categorical column "site"
    contract = {
        "score": {
            "source": {"column": "score", "transform": {"numeric": "log1p"}},
            "objective": "regression",
            "stratify": {"column": "site"},
        }
    }
    cohort = _cohort(contract)
    labels = {f"p{i}": {"score": str(i + 1.0), "site": ("A" if i % 2 == 0 else "B")} for i in range(6)}
    codes, strata = _valid_patients_and_strata(cohort.contract["score"], labels)
    assert len(codes) == 6
    assert set(strata) == {0, 1}  # stratified on site (A/B), not the score


# -- nested_cv / simple / all_* (unchanged behaviors, new contract) ----------
def test_nested_cv_naming_and_count():
    versions = generate_splits(
        _cohort(_BINARY), _binary_labels(10), _splits(SplitStrategy.nested_cv, n_outer=2, n_inner=3)
    )
    assert len(versions) == 6
    assert {v.version for v in versions} == {f"cancer_type_cv_o{k}_i{j}" for k in range(2) for j in range(3)}


def test_nested_cv_shared_test_invariant():
    versions = {
        v.version: v
        for v in generate_splits(
            _cohort(_BINARY), _binary_labels(12), _splits(SplitStrategy.nested_cv, n_outer=3, n_inner=2)
        )
    }
    for k in range(3):
        test_sets = [
            frozenset(c for c, cat in versions[f"cancer_type_cv_o{k}_i{j}"].assignments if cat == TEST)
            for j in range(2)
        ]
        assert test_sets[0] == test_sets[1] and len(test_sets[0]) > 0


def test_nested_cv_partition_disjoint_and_complete():
    all_codes = set(_binary_labels(8))
    for v in generate_splits(
        _cohort(_BINARY), _binary_labels(8), _splits(SplitStrategy.nested_cv, n_outer=2, n_inner=2)
    ):
        by_cat: dict[str, set[str]] = {}
        for code, cat in v.assignments:
            by_cat.setdefault(cat, set()).add(code)
        fit, val, test = by_cat["fit"], by_cat["validate"], by_cat["test"]
        assert fit.isdisjoint(val) and fit.isdisjoint(test) and val.isdisjoint(test)
        assert fit | val | test == all_codes


def test_simple_three_way():
    versions = generate_splits(
        _cohort(_BINARY), _binary_labels(10), _splits(SplitStrategy.simple, ratios=[0.6, 0.2, 0.2])
    )
    assert len(versions) == 1
    assert {cat for _, cat in versions[0].assignments} == {"fit", "validate", "test"}


def test_all_test_is_per_target():
    # all_test filters to each field's valid patients -> one all_test_<field> split, all TEST.
    (vt,) = generate_splits(_cohort(_BINARY), _binary_labels(5), _splits(SplitStrategy.all_test))
    assert vt.version == "all_test_cancer_type" and all(cat == TEST for _, cat in vt.assignments)


def test_all_predict_is_whole_cohort():
    # all_predict is unlabeled inference -> a single whole-cohort split, all PREDICT.
    (vp,) = generate_splits(_cohort(_BINARY), _binary_labels(5), _splits(SplitStrategy.all_predict))
    assert vp.version == "all_predict" and all(cat == PREDICT for _, cat in vp.assignments)


def test_all_test_excludes_out_of_map_patients():
    labels = _binary_labels(4)
    labels["luad0"] = {"cancer_type": "LUAD"}  # not in {BRCA, COAD} -> no valid label
    (vt,) = generate_splits(_cohort(_BINARY), labels, _splits(SplitStrategy.all_test))
    placed = {c for c, _ in vt.assignments}
    assert "luad0" not in placed and placed == set(_binary_labels(4))


def test_out_of_map_excluded_in_internal_cv():
    labels = _binary_labels(6)
    labels["luad0"] = {"cancer_type": "LUAD"}  # not in {BRCA, COAD}
    placed = {
        c
        for v in generate_splits(_cohort(_BINARY), labels, _splits(SplitStrategy.nested_cv, n_outer=2, n_inner=2))
        for c, _ in v.assignments
    }
    assert "luad0" not in placed and placed == set(_binary_labels(6))


# -- expression (regression_vector): unstratified nested CV over RNA-covered patients --
_EXPRESSION = {"expression": {"type": "expression"}}


def test_expression_splits_over_rna_covered_only():
    # 8 manifest patients, but only 6 have an RNA-matrix row -> folds partition exactly those 6.
    labels = {f"p{i}": {} for i in range(8)}
    covered = {f"p{i}" for i in range(6)}
    versions = generate_splits(
        _cohort(_EXPRESSION),
        labels,
        _splits(SplitStrategy.nested_cv, n_outer=2, n_inner=2),
        rnaseq_covered={"expression": covered},
    )
    assert {v.version for v in versions} == {f"expression_cv_o{k}_i{j}" for k in range(2) for j in range(2)}
    for v in versions:
        placed = {c for c, _ in v.assignments}
        assert placed == covered  # p6, p7 (no RNA) never appear


def test_expression_no_rna_coverage_skips_field():
    # regression_vector field with an empty coverage set -> no splits emitted (nothing to fold).
    labels = {f"p{i}": {} for i in range(4)}
    versions = generate_splits(
        _cohort(_EXPRESSION),
        labels,
        _splits(SplitStrategy.nested_cv, n_outer=2, n_inner=2),
        rnaseq_covered={"expression": set()},
    )
    assert versions == []


# -- require_modalities gate (fusion fair-comparison) -------------------------
def test_require_coverage_restricts_all_fields():
    # 20 patients; the modality gate keeps a 12-patient subset — only those may enter any split.
    labels = _binary_labels(10)
    covered = {f"brca{i}" for i in range(6)} | {f"coad{i}" for i in range(6)}
    versions = generate_splits(
        _cohort(_BINARY), labels, _splits(SplitStrategy.nested_cv, n_outer=2, n_inner=2), require_coverage=covered
    )
    seen = {code for v in versions for code, _cat in v.assignments}
    assert seen == covered


def test_require_coverage_none_is_all():
    labels = _binary_labels(5)
    versions = generate_splits(_cohort(_BINARY), labels, _splits(SplitStrategy.nested_cv, n_outer=2, n_inner=2))
    assert {code for v in versions for code, _cat in v.assignments} == set(labels)


def test_require_coverage_gates_all_predict():
    labels = _binary_labels(5)  # 10 patients
    covered = {f"brca{i}" for i in range(3)}
    (vp,) = generate_splits(_cohort(_BINARY), labels, _splits(SplitStrategy.all_predict), require_coverage=covered)
    assert {code for code, _cat in vp.assignments} == covered


# -- predefined splits: export / import round-trip + validation ---------------
def _nested_versions():
    """Generated nested-CV splits for a small binary cohort (2x2 folds over 6 patients)."""
    cohort = _cohort(_BINARY)
    labels = _binary_labels(3)  # 6 patients, 3 per class
    splits = _splits(SplitStrategy.nested_cv, n_outer=2, n_inner=2)
    versions = generate_splits(cohort, labels, splits)
    return cohort, labels, splits, versions


def test_export_import_round_trips(tmp_path):
    cohort, _labels, _splits_plan, versions = _nested_versions()
    path = tmp_path / "cancer_type_splits.csv"
    export_splits_csv(versions, path)
    reloaded = import_splits_csv(path)
    assert {v.version for v in reloaded} == {v.version for v in versions}
    got = {v.version: set(v.assignments) for v in reloaded}
    want = {v.version: set(v.assignments) for v in versions}
    assert got == want  # (patient, category) membership per version is byte-for-byte preserved


def test_validate_accepts_generated_split():
    cohort, labels, splits, versions = _nested_versions()
    valid = valid_patients_by_field(cohort, labels)
    validate_imported_splits(versions, splits, cohort_patients=set(labels), valid_by_field=valid)  # no raise


def test_validate_rejects_unknown_file_patient():
    cohort, labels, splits, versions = _nested_versions()
    valid = valid_patients_by_field(cohort, labels)
    tampered = list(versions)
    tampered[0] = SplitVersion(versions[0].version, versions[0].description, versions[0].assignments + [("ghost", FIT)])
    with pytest.raises(ValueError, match="not in the cohort"):
        validate_imported_splits(tampered, splits, cohort_patients=set(labels), valid_by_field=valid)


def test_validate_drops_unknown_file_patient_when_allowed():
    # Matched cross-cohort import: the canonical split names a patient this cohort lacks (no slide).
    # allow_uncovered reconciles to the intersection, dropping the file-only patient from the versions.
    cohort, labels, splits, versions = _nested_versions()
    valid = valid_patients_by_field(cohort, labels)
    tampered = list(versions)
    tampered[0] = SplitVersion(versions[0].version, versions[0].description, versions[0].assignments + [("ghost", FIT)])
    reconciled = validate_imported_splits(
        tampered, splits, cohort_patients=set(labels), valid_by_field=valid, allow_uncovered=True
    )
    assert all("ghost" not in {code for code, _ in v.assignments} for v in reconciled)


def test_validate_rejects_missing_fold():
    cohort, labels, splits, versions = _nested_versions()
    valid = valid_patients_by_field(cohort, labels)
    with pytest.raises(ValueError, match="expected fold set"):
        validate_imported_splits(versions[:-1], splits, cohort_patients=set(labels), valid_by_field=valid)


def test_validate_rejects_bad_category_set():
    cohort, labels, splits, versions = _nested_versions()
    valid = valid_patients_by_field(cohort, labels)
    # Strip TEST from one fold: a nested-CV fold must carry fit + validate + test.
    bad = [(c, cat) for c, cat in versions[0].assignments if cat != TEST]
    tampered = [SplitVersion(versions[0].version, versions[0].description, bad)] + list(versions[1:])
    with pytest.raises(ValueError, match="missing categories"):
        validate_imported_splits(tampered, splits, cohort_patients=set(labels), valid_by_field=valid)


def test_uncovered_labelled_patient_errors_then_allowed():
    cohort, labels, splits, versions = _nested_versions()
    # Pretend one more labelled patient exists in the cohort but is absent from the imported file.
    valid = {"cancer_type": set(labels) | {"orphan"}}
    patients = set(labels) | {"orphan"}
    with pytest.raises(ValueError, match="absent from the imported splits"):
        validate_imported_splits(versions, splits, cohort_patients=patients, valid_by_field=valid)
    validate_imported_splits(
        versions, splits, cohort_patients=patients, valid_by_field=valid, allow_uncovered=True
    )  # no raise when opted in


def test_import_csv_rejects_bad_category(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("patient_id,split_version,category\np0,cancer_type_cv_o0_i0,frobnicate\n")
    with pytest.raises(ValueError, match="category"):
        import_splits_csv(path)


# -- discretize (continuous -> k-class target) strata ------------------------
def test_continuous_cuts_quantile_tertiles():
    from dlux.data.splits import continuous_cuts

    cuts = continuous_cuts([float(i) for i in range(9)], method=StratifyMethod.quantile, k=3)
    assert len(cuts) == 2  # k-1 interior edges for k=3


def test_discretize_strata_are_k_class():
    from dlux.data.splits import _valid_patients_and_strata

    contract = {
        "cnh": {
            "source": {"column": "cnh", "transform": {"discretize": {"method": "quantile", "k": 3}}},
            "objective": "multiclass",
        }
    }
    cohort = _cohort(contract)
    labels = {f"p{i}": {"cnh": str(float(i))} for i in range(9)}  # 0..8 -> tertiles
    codes, strata = _valid_patients_and_strata(cohort.contract["cnh"], labels)
    assert len(codes) == 9
    assert strata == [0, 0, 0, 1, 1, 1, 2, 2, 2]  # equal-frequency thirds, ties go up
