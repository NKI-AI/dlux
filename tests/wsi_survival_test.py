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
"""Tests for the survival lane: the discrete-time NLL + risk + Harrell-C helpers, and the
WSISurvivalTask time-bin discretisation / target selection / loss (on fake batches — no cache)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from dlux.config.cohort import ContractField
from dlux.tasks.survival import (
    ConcordanceIndex,
    concordance_index,
    survival_nll,
    survival_risk,
)
from dlux.tasks.wsi_survival import WSISurvivalTask

_SURVIVAL = {"type": "survival", "event": "pfi_event", "time": "pfi_time"}


def _task(n_bins: int = 4, edges=(10.0, 20.0, 30.0)) -> WSISurvivalTask:
    return WSISurvivalTask(
        target="pfi",
        contract_field=ContractField.model_validate(_SURVIVAL),
        inputs={},  # methods under test do not read the streams
        data_description=None,  # not touched by the methods under test
        time_edges=np.asarray(edges, dtype=np.float32),
        n_bins=n_bins,
    )


# -- Harrell's C ------------------------------------------------------------
def test_concordance_perfect_and_reversed():
    time = np.array([1.0, 2.0, 3.0])
    event = np.array([1.0, 1.0, 1.0])
    # shortest survival carries the highest risk -> every comparable pair concordant.
    assert concordance_index(time, np.array([3.0, 2.0, 1.0]), event) == pytest.approx(1.0)
    assert concordance_index(time, np.array([1.0, 2.0, 3.0]), event) == pytest.approx(0.0)


def test_concordance_ties_are_half():
    # equal risk on the one comparable pair (t=1 event vs t=2) -> 0.5.
    c = concordance_index(np.array([1.0, 2.0]), np.array([5.0, 5.0]), np.array([1.0, 0.0]))
    assert c == pytest.approx(0.5)


def test_concordance_no_comparable_pair_is_nan():
    # the only event is on the longest-surviving patient -> no comparable pair.
    assert math.isnan(concordance_index(np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([0.0, 1.0])))


def test_metric_matches_function():
    metric = ConcordanceIndex()
    risk = torch.tensor([3.0, 2.0, 1.0])
    target = torch.tensor([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])  # [time, event]
    metric.update(risk, target)
    assert float(metric.compute()) == pytest.approx(1.0)


# -- discrete-time loss / risk ---------------------------------------------
def test_nll_finite_and_prefers_correct_hazard():
    bin_idx = torch.tensor([1, 2])
    event = torch.tensor([1.0, 1.0])  # both uncensored
    base = survival_nll(torch.zeros(2, 4), bin_idx, event)
    assert base.ndim == 0 and torch.isfinite(base)
    # raising the hazard logit in each patient's true event bin must lower the NLL.
    sharp = torch.zeros(2, 4)
    sharp[0, 1] = 4.0
    sharp[1, 2] = 4.0
    assert float(survival_nll(sharp, bin_idx, event)) < float(base)


def test_risk_increases_with_hazard():
    low = survival_risk(torch.full((1, 4), -3.0))  # low hazards -> high survival mass -> low risk
    high = survival_risk(torch.full((1, 4), 3.0))  # high hazards -> low survival mass -> high risk
    assert float(high) > float(low)


# -- task -------------------------------------------------------------------
def test_select_targets_bins_time():
    t = _task(n_bins=4, edges=(10.0, 20.0, 30.0))
    assert t.num_classes == 4
    batch = {
        "patient_label": {"patient.pfi_event": torch.tensor([1.0, 0.0]), "patient.pfi_time": torch.tensor([5.0, 25.0])}
    }
    tgt = t.select_targets(batch)
    assert torch.equal(tgt["bin"], torch.tensor([0, 2]))  # 5 -> bin 0, 25 -> bin 2
    assert tgt["event"].dtype == torch.float32 and tgt["time"].dtype == torch.float32


def test_time_edges_exposed_for_persistence():
    assert _task(edges=(10.0, 20.0, 30.0)).time_edges.tolist() == [10.0, 20.0, 30.0]
    # the inference-only construction has no fit split, so it has no edges to hand over.
    inference_only = WSISurvivalTask(
        target="pfi",
        contract_field=ContractField.model_validate(_SURVIVAL),
        inputs={},
        data_description=None,
        time_edges=None,
        n_bins=4,
    )
    assert inference_only.time_edges is None


def test_compute_loss_and_metric_tensors():
    t = _task()
    batch = {
        "patient_label": {"patient.pfi_event": torch.tensor([1.0, 0.0]), "patient.pfi_time": torch.tensor([5.0, 25.0])}
    }
    tgt = t.select_targets(batch)
    loss = t.compute_loss({"logits": torch.zeros(2, 4)}, tgt)
    assert loss.ndim == 0 and torch.isfinite(loss)
    risk, target = t.select_metric_tensors({"logits": torch.zeros(2, 4)}, tgt)
    assert risk.shape == (2,) and target.shape == (2, 2)


def test_rejects_non_survival_objective():
    with pytest.raises(ValueError, match="survival objective only"):
        WSISurvivalTask(
            target="msi_status",
            contract_field=ContractField.model_validate({"type": "binary", "map": {"MSS": 0, "MSI-H": 1}}),
            inputs={},
            data_description=None,
            time_edges=np.asarray([1.0], dtype=np.float32),
            n_bins=2,
        )
