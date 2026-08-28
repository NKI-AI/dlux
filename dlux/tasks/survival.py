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
"""Discrete-time survival loss + Harrell's C-index metric, torch/numpy only (no ahcore, no lifelines).

Follow-up is discretised into ``n_bins`` intervals (quantile edges from the fit split's event times).
The head emits one hazard logit per bin (``h_j = P(event in bin j | survived to j)``), and the loss is
the discrete-time negative log-likelihood (Zadeh & Schmid 2020; the loss PORPOISE/MCAT use): an
uncensored patient contributes surviving to its bin then having the event. A censored patient only
contributes surviving through its bin. Risk for ranking is ``-sum_j S_j`` (less survival mass -> higher
risk). Concordance is Harrell's C over comparable pairs (the shorter-time patient must have an event).
"""

from __future__ import annotations

import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.utilities import dim_zero_cat

# The C-index math is torch-free and shared with the aggregate reader, one definition of Harrell's C.
from dlux.eval.survival_metrics import concordance_index


def _hazards_and_survival(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-bin hazards ``h_j = sigmoid(logit_j)`` and the survival function ``S_j = prod_{k<=j}(1-h_k)``."""
    hazards = torch.sigmoid(logits)
    return hazards, torch.cumprod(1.0 - hazards, dim=1)


def survival_nll(
    logits: torch.Tensor, bin_idx: torch.Tensor, event: torch.Tensor, *, eps: float = 1e-7
) -> torch.Tensor:
    """Discrete-time NLL from per-bin hazard logits, the patient's time-bin index, and the event
    indicator (1 = event observed, 0 = censored). Plain likelihood, no extra up-weighting term."""
    hazards, surv = _hazards_and_survival(logits)  # (B, n_bins)
    # S_padded[:, j] = P(survive through bin j-1); S_padded[:, 0] = 1 (everyone starts alive).
    surv_padded = torch.cat([torch.ones_like(surv[:, :1]), surv], dim=1)  # (B, n_bins+1)
    y = bin_idx.view(-1, 1).long()
    e = event.view(-1, 1).to(logits.dtype)
    uncensored = -torch.log(torch.gather(surv_padded, 1, y).clamp(min=eps)) - torch.log(
        torch.gather(hazards, 1, y).clamp(min=eps)
    )
    censored = -torch.log(torch.gather(surv_padded, 1, y + 1).clamp(min=eps))
    return (e * uncensored + (1.0 - e) * censored).mean()


def survival_risk(logits: torch.Tensor) -> torch.Tensor:
    """Scalar risk per patient for ranking: ``-sum_j S_j`` (higher = shorter predicted survival)."""
    _hazards, surv = _hazards_and_survival(logits)
    return -surv.sum(dim=1)


class ConcordanceIndex(Metric):
    """Harrell's C-index accumulated over an epoch. ``update(risk, target)`` where ``target`` is the
    ``(N, 2)`` ``[time, event]`` tensor the survival task hands over via ``select_metric_tensors``,
    C-index is rank-based, so it is computed once over the whole epoch, not averaged per batch."""

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("risk", default=[], dist_reduce_fx="cat")
        self.add_state("time", default=[], dist_reduce_fx="cat")
        self.add_state("event", default=[], dist_reduce_fx="cat")

    def update(self, risk: torch.Tensor, target: torch.Tensor) -> None:
        self.risk.append(risk.detach().reshape(-1))
        self.time.append(target[:, 0].detach().reshape(-1))
        self.event.append(target[:, 1].detach().reshape(-1))

    def compute(self) -> torch.Tensor:
        risk = dim_zero_cat(self.risk).cpu().numpy()
        time = dim_zero_cat(self.time).cpu().numpy()
        event = dim_zero_cat(self.event).cpu().numpy()
        return torch.tensor(concordance_index(time, risk, event), dtype=torch.float32)


def build_survival_metrics(split: str) -> MetricCollection:
    """The ``{split}/c_index`` collection (a single accumulating Harrell-C metric)."""
    return MetricCollection({f"{split}/c_index": ConcordanceIndex()})
