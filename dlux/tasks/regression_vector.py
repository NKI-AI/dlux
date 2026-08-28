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
"""Expression (regression_vector) loss + streaming metrics, torch-only, so testable.

The target is a per-gene vector (log1p + per-gene z-scored by the task). Elements equal to the
missing sentinel (NaN), a patient/gene absent from the RNA matrix, are masked out of both the loss
and the streaming metrics. The headline per-gene Pearson / top-k / Spearman-median is computed at
aggregate time over pooled out-of-fold predictions (a vector metric needs all test patients), not
streamed per batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchmetrics import Metric, MetricCollection


def regression_vector_loss(preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Masked MSE over the non-missing (non-NaN) elements of the (B, G) target. Nothing valid in the
    batch -> zero loss with the graph intact."""
    target = target.float()
    mask = ~torch.isnan(target)
    if not bool(mask.any()):
        return preds.sum() * 0.0
    return F.mse_loss(preds[mask], target[mask])


class GenePearsonMean(Metric):
    """Mean per-gene Pearson over the full split, accumulated across batches (streamable model-selection
    signal, the aggregate-time headline made available per epoch).

    Running per-gene sums are accumulated from each (B, G) batch (NaN target elements masked per
    element, so a patient absent from the RNA matrix contributes nothing), so ``compute()`` is the
    NaN-safe mean over genes of each gene's Pearson across all patients seen this epoch. Genes seen on
    < 2 patients or constant (in preds or labels) are dropped. Higher is better, monitor
    ``validate/gene_pearson_mean`` to select checkpoints directly on the metric we care about. Pearson
    is invariant to the per-gene affine z-score, so this matches the log1p-space per-gene Pearson that
    aggregate reports."""

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, num_genes: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_genes = int(num_genes)
        for name in ("sum_x", "sum_y", "sum_xy", "sum_x2", "sum_y2", "count"):
            self.add_state(name, default=torch.zeros(self.num_genes, dtype=torch.float64), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = preds.double()
        target = target.double()
        mask = ~torch.isnan(target)  # (B, G): score only patient×gene pairs with a real label
        p = torch.where(mask, preds, torch.zeros_like(preds))
        t = torch.where(mask, target, torch.zeros_like(target))
        self.sum_x += p.sum(dim=0)
        self.sum_y += t.sum(dim=0)
        self.sum_xy += (p * t).sum(dim=0)
        self.sum_x2 += (p * p).sum(dim=0)
        self.sum_y2 += (t * t).sum(dim=0)
        self.count += mask.double().sum(dim=0)

    def compute(self) -> torch.Tensor:
        n = self.count
        n_safe = torch.where(n > 0, n, torch.ones_like(n))  # avoid div-by-zero; gated out below
        cov = self.sum_xy - self.sum_x * self.sum_y / n_safe
        var_x = self.sum_x2 - self.sum_x**2 / n_safe
        var_y = self.sum_y2 - self.sum_y**2 / n_safe
        denom = torch.sqrt(var_x * var_y)
        r = cov / torch.where(denom > 0, denom, torch.ones_like(denom))
        valid = (n >= 2) & (denom > 0) & torch.isfinite(r)
        if not bool(valid.any()):
            return torch.tensor(float("nan"), device=n.device)
        return r[valid].mean()


def build_gene_metrics(split: str, num_genes: int) -> MetricCollection:
    """Streaming mean per-gene Pearson for ``split`` (train/validate/test). MSE is already logged via
    the loss; the per-gene Pearson is the metric to select on, so it is the one streamed here."""
    return MetricCollection({f"{split}/gene_pearson_mean": GenePearsonMean(num_genes)})
