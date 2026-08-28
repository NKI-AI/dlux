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
"""Tests for the dlux facet contract schema (dlux.config.cohort)."""

from __future__ import annotations

import math

import pytest
from dlux.config.cohort import (
    Cohort,
    ContractField,
    Objective,
    Role,
    SplitParams,
    SplitStrategy,
    Storage,
    StratifyMethod,
    Study,
)
from pydantic import ValidationError


def _field(data: dict) -> ContractField:
    return ContractField.model_validate(data)


def _cohort(contract: dict) -> Cohort:
    return Cohort(name="c", storage=Storage(image_dir="/i"), contract=contract)


# -- sugar expansion ---------------------------------------------------------
def test_sugar_binary():
    f = _field({"type": "binary", "map": {"BRCA": 0, "COAD": 1}})
    assert f.objective is Objective.binary
    assert f.source.transform.categorical == {"BRCA": 0, "COAD": 1}
    assert f.source.column is None  # injected at Dataset level, not here
    assert f.num_outputs == 1 and f.target_dtype() == "float"


def test_sugar_multiclass():
    f = _field({"type": "multiclass", "map": {"BRCA": 0, "COAD": 1, "LUAD": 2}})
    assert f.objective is Objective.multiclass
    assert f.num_outputs == 3 and f.target_dtype() == "long"


def test_sugar_continuous_bare():
    f = _field({"type": "continuous"})
    assert f.objective is Objective.regression
    assert f.source.transform is None
    assert f.stratify.method is StratifyMethod.median  # default injected
    assert f.num_outputs == 1 and f.target_dtype() == "float"


def test_sugar_continuous_with_transform_and_stratify():
    f = _field({"type": "continuous", "target_transform": "log1p", "stratify": {"method": "quantile", "k": 4}})
    assert f.objective is Objective.regression
    assert f.source.transform.numeric == "log1p"
    assert f.stratify.method is StratifyMethod.quantile and f.stratify.k == 4


def test_sugar_and_facet_both_is_error():
    with pytest.raises(ValidationError):
        _field({"type": "binary", "objective": "binary", "map": {"A": 0, "B": 1}})


def test_survival_shorthand_builds_event_time_source():
    f = _field({"type": "survival", "event": "pfi_event", "time": "pfi_time"})
    assert f.objective is Objective.survival
    assert f.source.column == "pfi_event" and f.source.time_column == "pfi_time"
    assert f.source.transform is None


def test_survival_shorthand_requires_event_and_time():
    with pytest.raises(ValidationError):
        _field({"type": "survival", "event": "pfi_event"})  # missing time
    with pytest.raises(ValidationError):
        _field({"type": "survival", "time": "pfi_time"})  # missing event


def test_survival_num_outputs_is_deferred():
    # head width = task.n_bins, resolved at train time — not a contract fact.
    with pytest.raises(ValueError):
        _field({"type": "survival", "event": "pfi_event", "time": "pfi_time"}).num_outputs


def test_time_column_rejected_off_survival():
    with pytest.raises(ValidationError):
        _field({"source": {"column": "x", "time_column": "t"}, "objective": "regression"})


# -- facet form + the previously-impossible case -----------------------------
def test_facet_binary_canonical():
    f = _field({"source": {"column": "MPR", "transform": {"categorical": {"0": 0, "1": 1}}}, "objective": "binary"})
    assert f.objective is Objective.binary and f.source.column == "MPR"


def test_loss_is_not_a_contract_facet():
    """Class weighting moved to `task.loss.weight`: every facet left here is consumed while BUILDING
    the DB, and weighting is a per-run decision two arms must be able to differ on. `extra=forbid`
    makes a cohort still carrying it loud rather than silently ignored."""
    with pytest.raises(ValidationError):
        _field({"type": "binary", "map": {"0": 0, "1": 1}, "loss": {"weight": "balanced"}})


def test_binarize_continuous_to_binary():
    # the case that was inexpressible before
    f = _field({"source": {"column": "CNH", "transform": {"binarize": {"method": "median"}}}, "objective": "binary"})
    assert f.objective is Objective.binary and f.num_outputs == 1
    assert f.source.transform.binarize.method is StratifyMethod.median
    assert f.stratify.column is None and f.stratify.method is None  # defaults to the binarized target


def test_regression_with_numeric_and_explicit_stratify_column():
    f = _field(
        {
            "source": {"column": "ki67", "transform": {"numeric": "log1p"}},
            "objective": "regression",
            "stratify": {"column": "ki67", "method": "quantile", "k": 4},
        }
    )
    assert f.objective is Objective.regression
    assert f.stratify.column == "ki67" and f.stratify.k == 4


# -- cross-facet validation rejections ---------------------------------------
def test_two_transforms_rejected():
    with pytest.raises(ValidationError):
        _field({"source": {"transform": {"categorical": {"A": 0, "B": 1}, "numeric": "log1p"}}, "objective": "binary"})


def test_numeric_on_binary_rejected():
    with pytest.raises(ValidationError):
        _field({"source": {"transform": {"numeric": "log1p"}}, "objective": "binary"})


def test_binarize_on_regression_rejected():
    with pytest.raises(ValidationError):
        _field({"source": {"transform": {"binarize": {"method": "median"}}}, "objective": "regression"})


def test_categorical_on_regression_rejected():
    with pytest.raises(ValidationError):
        _field({"source": {"transform": {"categorical": {"A": 0}}}, "objective": "regression"})


def test_binary_needs_two_classes():
    with pytest.raises(ValidationError):
        _field({"type": "binary", "map": {"A": 0, "B": 1, "C": 2}})


def test_multiclass_needs_three_classes():
    with pytest.raises(ValidationError):
        _field({"type": "multiclass", "map": {"A": 0, "B": 1}})


def test_binarize_threshold_requires_value():
    with pytest.raises(ValidationError):
        _field({"source": {"transform": {"binarize": {"method": "threshold"}}}, "objective": "binary"})
    # median forbids value
    with pytest.raises(ValidationError):
        _field({"source": {"transform": {"binarize": {"method": "median", "value": 0.5}}}, "objective": "binary"})


def test_stratify_method_forbidden_on_categorical_target():
    with pytest.raises(ValidationError):
        _field({"type": "binary", "map": {"A": 0, "B": 1}, "stratify": {"method": "median"}})


def test_stratify_quantile_requires_k():
    with pytest.raises(ValidationError):
        _field({"type": "continuous", "stratify": {"column": "x", "method": "quantile"}})


# -- matrix source / expression (regression_vector) endpoint -----------------
def test_matrix_source_requires_regression_vector():
    # matrix source pairs ONLY with the expression objective, never plain regression.
    with pytest.raises(ValidationError):
        _field({"source": {"modality": "bulk_rna"}, "objective": "regression"})


def test_matrix_source_takes_no_column():
    with pytest.raises(ValidationError, match="takes no column"):
        _field({"source": {"modality": "bulk_rna", "column": "rnaseq"}, "objective": "regression_vector"})


def test_expression_sugar_builds_matrix_regression_vector():
    f = _field({"type": "expression"})
    assert f.objective is Objective.regression_vector
    assert f.source.modality == "bulk_rna" and f.source.column is None and f.source.transform is None


def test_regression_vector_is_unstratified():
    # a high-dim vector target has no stratification axis
    with pytest.raises(ValidationError):
        _field({"type": "expression", "stratify": {"method": "median"}})


def test_expression_num_outputs_is_deferred():
    # head width = gene-panel size, resolved at train time, not the contract
    with pytest.raises(ValueError):
        _ = _field({"type": "expression"}).num_outputs


# -- accessors ---------------------------------------------------------------
def test_num_outputs_and_dtype_and_sentinel():
    b = _field({"type": "binary", "map": {"A": 0, "B": 1}})
    m = _field({"type": "multiclass", "map": {"A": 0, "B": 1, "C": 2}})
    r = _field({"type": "continuous"})
    assert (b.num_outputs, b.target_dtype(), b.missing_sentinel()) == (1, "float", -1)
    assert (m.num_outputs, m.target_dtype(), m.missing_sentinel()) == (3, "long", -1)
    assert r.num_outputs == 1 and r.target_dtype() == "float" and math.isnan(r.missing_sentinel())


def test_metric_names_from_table():
    b = _field({"type": "binary", "map": {"A": 0, "B": 1}})
    assert "auroc" in b.metric_names("stream")
    assert {"auroc", "mcc", "f1"} <= set(b.metric_names("report"))


# -- Cohort-level wiring -----------------------------------------------------
def test_source_column_defaults_to_key():
    cohort = _cohort({"cancer_type": {"type": "binary", "map": {"BRCA": 0, "COAD": 1}}})
    assert cohort.contract["cancer_type"].source.column == "cancer_type"


def test_source_column_override_via_facet_form():
    cohort = _cohort(
        {
            "cnh_high": {
                "source": {"column": "CNH", "transform": {"binarize": {"method": "median"}}},
                "objective": "binary",
            }
        }
    )
    assert cohort.contract["cnh_high"].source.column == "CNH"  # endpoint name != source column


def test_reserved_name_clash():
    with pytest.raises(ValidationError):
        _cohort({"image_path": {"type": "binary", "map": {"A": 0, "B": 1}}})


# -- Study composition (role -> split strategy + study params) ----------------
def test_study_requires_a_development_cohort():
    with pytest.raises(ValidationError):
        Study(
            name="s", cohorts={"val_cohort": Role.validation}, targets=["t"], require_modalities=[]
        )  # validation-only -> no models to train


def test_study_requires_at_least_one_cohort():
    with pytest.raises(ValidationError):
        Study(name="s", cohorts={}, targets=["t"], require_modalities=[])


def test_role_maps_to_split_strategy():
    study = Study(
        name="s",
        cohorts={"dev_cohort": Role.development, "val_cohort": Role.validation},
        targets=["t"],
        require_modalities=[],
    )
    assert study.strategy_for("dev_cohort") is SplitStrategy.nested_cv
    assert study.strategy_for("val_cohort") is SplitStrategy.all_test


def test_strategy_for_unknown_cohort_raises():
    study = Study(name="s", cohorts={"dev_cohort": Role.development}, targets=["t"], require_modalities=[])
    with pytest.raises(KeyError):
        study.strategy_for("val_cohort")


def _two_endpoint_cohort() -> Cohort:
    return Cohort(
        name="c",
        storage=Storage(image_dir="/i"),
        contract={
            "leukocyte_fraction": {"type": "continuous"},
            "expression": {"type": "expression"},
        },
    )


def test_targets_and_require_modalities_are_required():
    """Omission used to mean "every endpoint the cohort offers" and "gate on nothing" — implicit defaults
    that made a sweep's expected shape unknowable downstream. Both must now be written."""
    with pytest.raises(ValidationError, match="targets"):
        Study(name="s", cohorts={"c": Role.development}, require_modalities=[])  # no targets
    with pytest.raises(ValidationError, match="require_modalities"):
        Study(name="s", cohorts={"c": Role.development}, targets=["expression"])  # no require_modalities
    Study(name="s", cohorts={"c": Role.development}, targets=["expression"], require_modalities=[])  # explicit


def test_filter_contract_scopes_to_targets():
    study = Study(name="s", cohorts={"c": Role.development}, targets=["expression"], require_modalities=[])
    filtered = study.filter_contract(_two_endpoint_cohort())
    assert set(filtered.contract) == {"expression"}  # leukocyte_fraction dropped


def test_filter_contract_rejects_unknown_target():
    study = Study(name="s", cohorts={"c": Role.development}, targets=["nonexistent"], require_modalities=[])
    with pytest.raises(ValueError):
        study.filter_contract(_two_endpoint_cohort())


def test_study_targets_must_be_nonempty_when_given():
    with pytest.raises(ValidationError):
        Study(name="s", cohorts={"c": Role.development}, targets=[], require_modalities=[])


def test_splits_for_composes_role_strategy_with_study_params():
    study = Study(
        name="s",
        cohorts={"dev_cohort": Role.development, "val_cohort": Role.validation},
        splits=SplitParams(n_outer=3, n_inner=2, random_state=7),
        targets=["t"],
        require_modalities=[],
    )
    dev = study.splits_for("dev_cohort")
    assert dev.strategy is SplitStrategy.nested_cv  # role-derived, not authored
    assert (dev.n_outer, dev.n_inner, dev.random_state) == (3, 2, 7)  # study params ride along
    assert study.splits_for("val_cohort").strategy is SplitStrategy.all_test


def test_require_modalities_validated_where_it_is_resolved():
    """The schema no longer keeps its own modality name list — the registry is the single source, so a
    name that cannot gate splits is rejected by build_db, naming the ones that can."""
    Study(name="s", cohorts={"c": "development"}, require_modalities=["proteomics"], targets=["t"])  # schema-legal
    from dlux.data.build_db import MODALITY_COVERAGE, _resolve_require_coverage
    from dlux.data.errors import BuildDbError

    assert "bulk_rna" in MODALITY_COVERAGE  # declares a coverage capability -> gateable
    assert "tile_features" not in MODALITY_COVERAGE  # no coverage capability -> cannot gate
    with pytest.raises(BuildDbError, match="proteomics"):
        _resolve_require_coverage(["proteomics"], None)


def test_require_modalities_threads_into_splits():
    s = Study(name="s", cohorts={"c": "development"}, require_modalities=["bulk_rna"], targets=["t"])
    assert s.splits_for("c").require_modalities == ["bulk_rna"]


def test_admin_censor_defaults_empty_and_is_optional():
    s = Study(name="s", cohorts={"c": Role.development}, targets=["efs"], require_modalities=[])
    assert s.admin_censor == {}


def test_admin_censor_key_must_be_a_study_target():
    with pytest.raises(ValidationError, match="admin_censor"):
        Study(
            name="s",
            cohorts={"c": Role.development},
            targets=["efs"],
            require_modalities=[],
            admin_censor={"os": 24},  # not a target
        )


def test_admin_censor_horizon_must_be_positive():
    with pytest.raises(ValidationError, match="positive"):
        Study(
            name="s",
            cohorts={"c": Role.development},
            targets=["efs"],
            require_modalities=[],
            admin_censor={"efs": 0},
        )


def test_apply_admin_censoring_caps_events_beyond_horizon():
    """Beyond the horizon: event -> 0 at time -> horizon. At/below it and missing times are untouched;
    the boundary (time == horizon) is kept as a real event."""
    import pandas as pd
    from dlux.data.build_db import _apply_admin_censoring

    cohort = _cohort({"efs": {"type": "survival", "event": "efs_event", "time": "efs"}})
    df = pd.DataFrame(
        {
            "patient_id": ["a", "b", "c", "d"],
            "efs": [10.0, 30.0, 24.0, float("nan")],
            "efs_event": [1, 1, 1, 0],
        }
    )
    out = _apply_admin_censoring(df, cohort, {"efs": 24})
    assert list(out["efs_event"]) == [1, 0, 1, 0]  # only b (30mo event) censored
    assert out["efs"].tolist()[:3] == [10.0, 24.0, 24.0]  # b capped to 24; a and boundary c unchanged
    assert math.isnan(out["efs"].tolist()[3])  # missing time left missing


def test_apply_admin_censoring_rejects_a_non_survival_target():
    import pandas as pd
    from dlux.data.build_db import _apply_admin_censoring
    from dlux.data.errors import BuildDbError

    cohort = _cohort({"lf": {"type": "continuous"}})
    df = pd.DataFrame({"patient_id": ["a"], "lf": [0.5]})
    with pytest.raises(BuildDbError, match="not a survival endpoint"):
        _apply_admin_censoring(df, cohort, {"lf": 24})


# -- discretize (continuous -> k-class target) -------------------------------
def _discretize_field(spec):
    return ContractField.model_validate(spec)


def test_discretize_num_outputs_quantile_and_threshold():
    q = _discretize_field(
        {
            "source": {"column": "cnh", "transform": {"discretize": {"method": "quantile", "k": 3}}},
            "objective": "multiclass",
        }
    )
    t = _discretize_field(
        {
            "source": {"column": "x", "transform": {"discretize": {"method": "threshold", "edges": [1.0, 2.0, 3.0]}}},
            "objective": "multiclass",
        }
    )
    assert q.num_outputs == 3 and t.num_outputs == 4  # k, and len(edges)+1


def test_discretize_rejects_k_below_three_and_wrong_objective():
    with pytest.raises(ValueError, match="k >= 3"):
        _discretize_field(
            {
                "source": {"column": "c", "transform": {"discretize": {"method": "quantile", "k": 2}}},
                "objective": "multiclass",
            }
        )
    for objective in ("binary", "regression"):
        with pytest.raises(ValueError):
            _discretize_field(
                {
                    "source": {"column": "c", "transform": {"discretize": {"method": "quantile", "k": 3}}},
                    "objective": objective,
                }
            )


def test_transform_still_exactly_one_kind():
    with pytest.raises(ValueError, match="exactly one"):
        ContractField.model_validate(
            {
                "source": {
                    "column": "c",
                    "transform": {"binarize": {"method": "median"}, "discretize": {"method": "quantile", "k": 3}},
                },
                "objective": "multiclass",
            }
        )
