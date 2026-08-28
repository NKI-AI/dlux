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
"""Survival metric math, pure numpy (no torch/ahcore), so both the train-time torchmetric wrapper
(``dlux.tasks.survival``) and the torch-free aggregate reader (``dlux.eval.aggregate``) share one
definition of Harrell's C. Parallel to ``dlux.eval.gene_metrics`` for the expression endpoint."""

from __future__ import annotations

import numpy as np

SURVIVAL_METRIC_LABELS = {"c_index": "C-index"}


def concordance_index(time: np.ndarray, risk: np.ndarray, event: np.ndarray) -> float:
    """Harrell's C over comparable pairs. A pair is comparable when the shorter-survival patient had an
    observed event; it is concordant when that patient also carries the higher risk (ties in risk = ½).
    NaN when no comparable pair exists (e.g. a fold with no observed events)."""
    time = np.asarray(time, dtype=np.float64)
    risk = np.asarray(risk, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    concordant = permissible = 0.0
    for i in range(time.size):
        if event[i] != 1.0:  # only the earlier-time patient having an event makes a pair comparable
            continue
        longer = time > time[i]
        n_longer = int(longer.sum())
        if n_longer == 0:
            continue
        permissible += n_longer
        concordant += float((risk[i] > risk[longer]).sum()) + 0.5 * float((risk[i] == risk[longer]).sum())
    return concordant / permissible if permissible > 0 else float("nan")


def bin_hazards(logits: np.ndarray) -> np.ndarray:
    """Per-bin hazards ``h_j = sigmoid(logit_j)`` from an ``(N, n_bins)`` block of head outputs.
    ``h_j`` is P(event in bin j | alive entering bin j)."""
    return 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))


def bin_survival(hazards: np.ndarray) -> np.ndarray:
    """``S_j = prod_{k<=j} (1 - h_k)``, P(surviving through the end of bin j)."""
    return np.cumprod(1.0 - np.asarray(hazards, dtype=np.float64), axis=-1)


def bin_ranges(edges: np.ndarray) -> list[str]:
    """Label each of the ``len(edges) + 1`` bins by the follow-up interval it spans, e.g.
    ``["0-183", "183-402", "402-1094", "1094+"]`` for the three interior cut points
    ``[183, 402, 1094]``. Units are the time column's, since the edges are quantiles of it."""
    values = [f"{float(e):g}" for e in np.asarray(edges, dtype=np.float64).ravel()]
    lower = ["0", *values]
    return [f"{lo}-{hi}" for lo, hi in zip(lower, values)] + [f"{lower[-1]}+"]


def kaplan_meier(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier survival curve as ``(t_steps, survival)`` step arrays (both start at ``(0, 1)``).
    Censored patients leave the risk set at their time but do not drop the estimate."""
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    steps = [0.0]
    surv = [1.0]
    s = 1.0
    for ut in np.unique(time):
        deaths = int(((time == ut) & (event == 1.0)).sum())
        at_risk = int((time >= ut).sum())
        if at_risk > 0 and deaths > 0:
            s *= 1.0 - deaths / at_risk
        steps.append(float(ut))
        surv.append(s)
    return np.asarray(steps), np.asarray(surv)
