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
"""Tests for the raw -> modeled-target seam: which transforms are wired, and what a binarize target
does with the build-db-resolved cut it depends on."""

from __future__ import annotations

import json
import math

import pytest
from dlux.config.cohort import ContractField
from dlux.tasks.classification import EXCLUDED
from dlux.tasks.target import build_field_transform, load_target_stats

_BINARIZE = {
    "source": {"column": "leukocyte_fraction", "transform": {"binarize": {"method": "median"}}},
    "objective": "binary",
}
_STATS = {"transform": "binarize", "column": "leukocyte_fraction", "threshold": 0.2, "n": 321}


def _field(spec):
    return ContractField.model_validate(spec)


def test_binarize_cuts_at_the_resolved_threshold():
    t = build_field_transform(_field(_BINARIZE), _STATS)
    assert (t(0.15), t(0.42)) == (0, 1)
    assert t(0.2) == 1, "the cut point itself belongs to the positive class"


def test_binarize_excludes_a_missing_value():
    t = build_field_transform(_field(_BINARIZE), _STATS)
    assert t("") == EXCLUDED and t(None) == EXCLUDED and t("n/a") == EXCLUDED


def test_binarize_refuses_to_guess_a_threshold():
    """Falling back to a per-fold median would move the ground truth between folds, which is exactly
    what resolving the cut once at build-db exists to prevent. Refuse instead."""
    with pytest.raises(ValueError, match="build-db-resolved threshold"):
        build_field_transform(_field(_BINARIZE))
    with pytest.raises(ValueError, match="build-db-resolved threshold"):
        build_field_transform(_field(_BINARIZE), {"transform": "binarize", "column": "x"})  # entry without the cut


def test_log1p_needs_no_statistic():
    t = build_field_transform(
        _field({"source": {"column": "tmb", "transform": {"numeric": "log1p"}}, "objective": "regression"})
    )
    assert t(3) == pytest.approx(math.log1p(3))
    assert math.isnan(t("")), "unparseable -> the regression NaN sentinel"


def test_zscore_is_not_a_contract_transform():
    """Standardising a target is `task.target_normalize`, which de-standardises before metrics. As a
    contract transform it would only change the reported units."""
    with pytest.raises(Exception):
        _field({"source": {"column": "x", "transform": {"numeric": "zscore"}}, "objective": "regression"})


def test_load_target_stats(tmp_path):
    p = tmp_path / "c_contract_stats.json"
    p.write_text(json.dumps({"leukocyte_high": _STATS}))
    assert load_target_stats(p, "leukocyte_high")["threshold"] == 0.2
    assert load_target_stats(p, "not_an_endpoint") is None  # endpoint needs no statistic
    assert load_target_stats(tmp_path / "absent.json", "x") is None  # cohort resolved none at all


# -- discretize (continuous -> k classes) ------------------------------------
_DISCRETIZE = {
    "source": {"column": "cnh", "transform": {"discretize": {"method": "quantile", "k": 3}}},
    "objective": "multiclass",
}
_DISCRETIZE_STATS = {"transform": "discretize", "column": "cnh", "edges": [2.6667, 5.3333], "n": 9}


def test_discretize_cuts_at_resolved_edges():
    t = build_field_transform(_field(_DISCRETIZE), _DISCRETIZE_STATS)
    # value AT a cut goes to the higher class (same >= tie rule as binarize / np.digitize)
    assert [t(v) for v in (0.0, 2.0, 2.6667, 4.0, 5.3333, 9.0)] == [0, 0, 1, 1, 2, 2]


def test_discretize_missing_is_excluded():
    t = build_field_transform(_field(_DISCRETIZE), _DISCRETIZE_STATS)
    assert t("nan") == EXCLUDED and t("") == EXCLUDED


def test_discretize_needs_resolved_edges():
    with pytest.raises(ValueError, match="discretize target needs its build-db-resolved edges"):
        build_field_transform(_field(_DISCRETIZE), None)


def test_discretize_threshold_variant():
    field = _field(
        {
            "source": {"column": "x", "transform": {"discretize": {"method": "threshold", "edges": [2.0, 5.0]}}},
            "objective": "multiclass",
        }
    )
    t = build_field_transform(field, {"transform": "discretize", "column": "x", "edges": [2.0, 5.0]})
    assert [t(v) for v in (0, 2, 3, 5, 9)] == [0, 1, 1, 2, 2]
