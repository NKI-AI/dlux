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
"""Tests for dlux.tasks.regression + the identity-regression target transform (NaN-masked)."""

from __future__ import annotations

import math

import pytest
import torch
from dlux.config.cohort import ContractField
from dlux.tasks.regression import build_regression_metrics, regression_loss, regression_metric_tensors
from dlux.tasks.target import build_field_transform
from dlux.tasks.wsi_regression import WSIRegressionTask


def _reg_task(target_mean=0.0, target_std=1.0):
    """A WSIRegressionTask with no adapters exercised — only the loss/metric standardisation paths."""
    field = ContractField.model_validate({"type": "continuous", "stratify": {"method": "median"}})
    return WSIRegressionTask(
        target="y",
        contract_field=field,
        inputs={},
        data_description=None,
        target_mean=target_mean,
        target_std=target_std,
    )


def test_mse_matches_manual():
    preds = torch.tensor([[1.0], [2.0], [3.0]])
    target = torch.tensor([1.0, 0.0, 3.0])  # squared errors 0, 4, 0 -> mean 4/3
    assert float(regression_loss(preds, target)) == pytest.approx(4 / 3)


def test_loss_masks_nan_targets():
    preds = torch.tensor([[1.0], [5.0]])
    target = torch.tensor([1.0, float("nan")])  # second is missing
    masked = regression_loss(preds, target)
    only_valid = regression_loss(preds[:1], target[:1])
    assert torch.allclose(masked, only_valid)


def test_all_missing_gives_zero_loss_with_graph():
    preds = torch.tensor([[1.0], [2.0]], requires_grad=True)
    target = torch.tensor([float("nan"), float("nan")])
    loss = regression_loss(preds, target)
    assert float(loss) == 0.0
    loss.backward()  # graph intact -> no error


def test_metric_tensors_filter_nan():
    preds = torch.tensor([[1.0], [2.0], [3.0]])
    target = torch.tensor([1.0, float("nan"), 3.0])
    p, t = regression_metric_tensors(preds, target)
    assert p.tolist() == [1.0, 3.0] and t.tolist() == [1.0, 3.0]


def test_build_metrics_keys():
    m = build_regression_metrics("validate")
    assert set(m.keys()) == {"validate/mae", "validate/rmse", "validate/r2", "validate/spearman"}


def test_identity_regression_transform():
    field = ContractField.model_validate({"type": "continuous", "stratify": {"method": "median"}})
    tf = build_field_transform(field)
    assert tf("40") == 40.0
    assert math.isnan(tf(""))  # blank -> NaN missing
    assert math.isnan(tf("n/a"))  # unparseable -> NaN


def test_zscore_loss_standardizes_target():
    # A head that perfectly predicts the STANDARDISED target -> ~0 loss (loss compares in z-space).
    task = _reg_task(target_mean=10.0, target_std=5.0)
    target = torch.tensor([10.0, 15.0, 5.0])
    preds_std = torch.tensor([[0.0], [1.0], [-1.0]])  # == (target - 10) / 5
    assert float(task.compute_loss({"logits": preds_std}, {"label": target})) == pytest.approx(0.0)


def test_zscore_metrics_destandardize_to_raw():
    # Standardised head output -> metrics see RAW units (predictions de-standardised, target untouched).
    task = _reg_task(target_mean=10.0, target_std=5.0)
    target = torch.tensor([10.0, 15.0, 5.0])
    preds_std = torch.tensor([[0.0], [1.0], [-1.0]])
    p, t = task.select_metric_tensors({"logits": preds_std}, {"label": target})
    assert torch.allclose(p, target) and torch.allclose(t, target)


def test_identity_normalization_is_noop():
    task = _reg_task()  # μ=0, σ=1
    target = torch.tensor([1.0, 2.0, 3.0])
    preds = torch.tensor([[1.0], [2.0], [3.0]])
    assert float(task.compute_loss({"logits": preds}, {"label": target})) == pytest.approx(0.0)
    p, _ = task.select_metric_tensors({"logits": preds}, {"label": target})
    assert torch.allclose(p, target)
