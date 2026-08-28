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
"""Tests for the dataset report's class-weighting callout: when it fires and what it emits."""

from __future__ import annotations

import pytest
from dlux.data.analyze import _imbalance_lines, class_weight_snippet
from pydantic import ValidationError


def test_no_callout_when_the_endpoint_is_near_balanced():
    assert _imbalance_lines([42, 48]) == []
    assert _imbalance_lines([201, 225, 384]) == []  # 1.9:1 -> weights land near 1


def test_callout_fires_on_an_imbalanced_endpoint():
    body = "\n".join(_imbalance_lines([155, 640]))  # 4.1:1
    assert "4.1:1" in body
    assert "weight: balanced" in body


def test_multiclass_ratio_is_majority_over_minority():
    """Not first-vs-last: the ratio that matters spans the whole distribution."""
    assert "5.0:1" in "\n".join(_imbalance_lines([500, 100, 300]))


def test_an_absent_class_is_excluded_rather_than_dividing_by_zero():
    assert "4.0:1" in "\n".join(_imbalance_lines([100, 0, 400]))  # ratio over present classes


def test_a_single_class_has_no_imbalance():
    assert _imbalance_lines([40]) == []


def test_snippet_is_valid_yaml_for_an_experiment_override():
    import yaml

    assert yaml.safe_load(class_weight_snippet()) == {"task": {"loss": {"weight": "balanced"}}}


def test_loss_spec_rejects_a_non_positive_weight():
    """The schema catches it at config-load time, before a run starts."""
    from dlux.config.task import LossSpec

    assert LossSpec().weight == "none"
    assert LossSpec(weight="balanced").weight == "balanced"
    with pytest.raises(ValidationError, match="must be > 0"):
        LossSpec(weight={"0": 1.0, "1": 0.0})
    with pytest.raises(ValidationError):
        LossSpec(weight="blanaced")  # typo, not a literal
