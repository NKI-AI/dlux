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
"""Regression loss + metrics, torch-only, so testable.

Targets equal to the missing sentinel (NaN), a label that was missing or out of the contract, are
masked out of both the loss and the metrics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection, R2Score, SpearmanCorrCoef


def regression_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE over the non-missing (non-NaN) targets. Nothing valid in the batch -> zero loss, graph intact."""
    preds = preds.squeeze(-1)
    target = target.float()
    mask = ~torch.isnan(target)
    if not bool(mask.any()):
        return preds.sum() * 0.0
    return F.mse_loss(preds[mask], target[mask])


def regression_metric_tensors(preds: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(preds, target)`` for torchmetrics, with NaN (missing) targets filtered out."""
    preds = preds.squeeze(-1)
    target = target.float()
    mask = ~torch.isnan(target)
    return preds[mask], target[mask]


def build_regression_metrics(split: str) -> MetricCollection:
    """MAE, RMSE, R2, and Spearman rank correlation for the given split (`train`/`validate`/`test`)."""
    return MetricCollection(
        {
            f"{split}/mae": MeanAbsoluteError(),
            f"{split}/rmse": MeanSquaredError(squared=False),
            f"{split}/r2": R2Score(),
            f"{split}/spearman": SpearmanCorrCoef(),
        }
    )
