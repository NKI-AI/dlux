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
"""Tests for dlux.tasks.classification: label mapping, EXCLUDED masking, metrics."""

from __future__ import annotations

import pytest
import torch
from dlux.tasks.classification import (
    EXCLUDED,
    LabelMapping,
    balanced_class_weights,
    balanced_pos_weight,
    build_classification_metrics,
    classification_loss,
    classification_metric_tensors,
    resolve_class_weights,
    resolve_pos_weight,
)


def test_label_mapping_excludes_unknown():
    m = LabelMapping({"BRCA": 0, "COAD": 1})
    assert m["BRCA"] == 0
    assert m["COAD"] == 1
    assert m["LUAD"] == EXCLUDED  # unmapped -> excluded, not a KeyError


def test_binary_loss_masks_excluded():
    logits = torch.tensor([[2.0], [-2.0], [0.0]])
    # third label EXCLUDED -> must not affect the loss
    label = torch.tensor([1, 0, EXCLUDED])
    loss_masked = classification_loss(logits, label, is_binary=True)
    loss_two = classification_loss(logits[:2], label[:2], is_binary=True)
    assert torch.allclose(loss_masked, loss_two)


def test_all_excluded_gives_zero_loss_with_graph():
    logits = torch.tensor([[1.5], [-0.5]], requires_grad=True)
    label = torch.tensor([EXCLUDED, EXCLUDED])
    loss = classification_loss(logits, label, is_binary=True)
    assert float(loss) == 0.0
    loss.backward()  # graph intact -> no error


def test_multiclass_loss_runs():
    logits = torch.randn(4, 3)
    label = torch.tensor([0, 2, 1, EXCLUDED])
    loss = classification_loss(logits, label, is_binary=False)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_metric_tensors_filter_excluded():
    logits = torch.tensor([[1.0], [2.0], [3.0]])
    label = torch.tensor([1, EXCLUDED, 0])
    preds, target = classification_metric_tensors(logits, label, is_binary=True)
    assert preds.shape == (2,)
    assert target.tolist() == [1, 0]


def test_metric_tensors_are_probabilities():
    """torchmetrics infers logits-vs-probabilities from the value range, so it must be handed
    probabilities."""
    logits = torch.tensor([[-0.58], [0.84], [0.31]])
    preds, _ = classification_metric_tensors(logits, torch.tensor([0, 1, 1]), is_binary=True)
    assert torch.allclose(preds, torch.sigmoid(logits.squeeze(-1)))
    assert bool(((preds >= 0) & (preds <= 1)).all())

    multi = torch.tensor([[2.0, -1.0, 0.5], [0.1, 0.2, 0.3]])
    preds, _ = classification_metric_tensors(multi, torch.tensor([0, 2]), is_binary=False)
    assert torch.allclose(preds.sum(dim=-1), torch.ones(2))


def test_streamed_metric_matches_whole_batch_for_near_zero_logits():
    """A metric streamed one sample at a time must equal the same metric computed in one shot,
    including for near-zero logits that straddle [0, 1]."""
    from torchmetrics import AUROC

    logits = torch.tensor([[-0.58], [0.84], [0.31], [-0.12], [0.66], [-0.41]])
    label = torch.tensor([0, 1, 1, 0, 1, 0])
    preds, target = classification_metric_tensors(logits, label, is_binary=True)

    streamed = AUROC(task="binary")
    for p, t in zip(preds, target):
        streamed.update(p.unsqueeze(0), t.unsqueeze(0))
    one_shot = AUROC(task="binary")
    one_shot.update(preds, target)
    assert float(streamed.compute()) == pytest.approx(float(one_shot.compute()))


def test_build_metrics_keys():
    bm = build_classification_metrics(is_binary=True, num_classes=1, split="validate")
    assert set(bm.keys()) == {"validate/accuracy", "validate/auroc"}
    mm = build_classification_metrics(is_binary=False, num_classes=3, split="test")
    assert set(mm.keys()) == {"test/accuracy", "test/auroc"}


# -- class weighting (balanced pos_weight) -----------------------------------
def test_balanced_pos_weight():
    assert balanced_pos_weight([1, 1, 1, 0]) == pytest.approx(1 / 3)  # 1 neg / 3 pos
    assert balanced_pos_weight([0, 0, 0, 0, 1]) == pytest.approx(4.0)  # 4 neg / 1 pos
    assert balanced_pos_weight([1, 0, EXCLUDED, EXCLUDED]) == pytest.approx(1.0)  # excluded ignored -> 1/1
    assert balanced_pos_weight([1, 1, 1]) == 1.0  # single-class train split -> nothing to balance


def test_resolve_pos_weight():
    assert resolve_pos_weight("none", [0, 1, 1]) is None
    assert resolve_pos_weight("balanced", [0, 0, 0, 1]) == pytest.approx(3.0)  # 3 neg / 1 pos
    assert resolve_pos_weight({"0": 1.0, "1": 4.0}, []) == pytest.approx(4.0)  # manual ratio, no data needed


def test_pos_weight_upweights_positive_class():
    logits = torch.zeros(2, 1)  # sigmoid(0) = 0.5 for both examples
    label = torch.tensor([1, 0])
    base = classification_loss(logits, label, is_binary=True)
    weighted = classification_loss(logits, label, is_binary=True, pos_weight=3.0)
    assert float(weighted) > float(base)  # positive example's BCE term tripled -> larger total loss


# -- class weighting (balanced multiclass CE weights) ------------------------
def test_balanced_class_weights():
    assert balanced_class_weights([0, 1, 2], 3) == pytest.approx([1.0, 1.0, 1.0])  # uniform -> no reweighting
    # 6 samples, K=3, supports 3/2/1 -> 6/(3*3), 6/(3*2), 6/(3*1)
    assert balanced_class_weights([0, 0, 0, 1, 1, 2], 3) == pytest.approx([2 / 3, 1.0, 2.0])
    assert balanced_class_weights([0, 0, EXCLUDED, 1, 1], 2) == pytest.approx([1.0, 1.0])  # excluded ignored
    assert balanced_class_weights([0, 0, 0], 3) == pytest.approx([1 / 3, 1.0, 1.0])  # absent classes -> 1.0
    assert balanced_class_weights([], 3) == [1.0, 1.0, 1.0]  # nothing to balance


def test_balanced_class_weights_average_one_per_sample():
    """Each class contributes n/K of the total weight, so the average weight PER SAMPLE is 1.0 —
    turning weighting on redistributes the loss between classes without rescaling it."""
    indices = [0] * 7 + [1] * 2 + [2] * 1
    weights = balanced_class_weights(indices, 3)
    assert sum(weights[c] for c in indices) / len(indices) == pytest.approx(1.0)


def test_a_zero_weight_is_refused_rather_than_silently_unweighted():
    """`{"0": 0, "1": 1}` used to reach `w1 / w0 if w0 else None` and come back None, i.e. training
    unweighted while the config said otherwise."""
    with pytest.raises(ValueError, match="must be > 0"):
        resolve_pos_weight({"0": 0.0, "1": 1.0}, [])
    with pytest.raises(ValueError, match="must be > 0"):
        resolve_class_weights({"0": 1.0, "1": -2.0}, [], 2)


def test_the_dict_form_is_scale_invariant_in_both_objectives():
    """Only the ratios matter: BCE takes the single relative weight, CE normalises by the sum."""
    assert resolve_pos_weight({"0": 1.0, "1": 4.0}, []) == resolve_pos_weight({"0": 3.0, "1": 12.0}, [])
    logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    label = torch.tensor([0, 1])
    a = classification_loss(logits, label, is_binary=False, class_weights=[1.0, 4.0, 1.0])
    b = classification_loss(logits, label, is_binary=False, class_weights=[3.0, 12.0, 3.0])
    assert float(a) == pytest.approx(float(b))


def test_resolve_class_weights():
    assert resolve_class_weights("none", [0, 1, 2], 3) is None
    assert resolve_class_weights("balanced", [0, 0, 1, 1, 1, 2], 3) == pytest.approx([1.0, 2 / 3, 2.0])
    manual = resolve_class_weights({"0": 2.0, "2": 4.0}, [], 3)  # manual weights need no data
    assert manual == [2.0, 1.0, 4.0]  # omitted class 1 defaults to 1.0


def test_class_weights_upweight_the_named_class():
    """CE normalises by the SUMMED weights, so weighting shifts the balance between examples rather
    than scaling the loss — it only moves when the per-example losses differ."""
    logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 3.0]])  # class 0 predicted well, class 1 badly
    label = torch.tensor([0, 1])
    base = classification_loss(logits, label, is_binary=False)
    weighted = classification_loss(logits, label, is_binary=False, class_weights=[1.0, 5.0, 1.0])
    assert float(weighted) > float(base)  # the badly-predicted class-1 example now dominates


def test_uniform_class_weights_leave_the_loss_unchanged():
    logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    label = torch.tensor([0, 1])
    base = classification_loss(logits, label, is_binary=False)
    flat = classification_loss(logits, label, is_binary=False, class_weights=[2.0, 2.0, 2.0])
    assert float(flat) == pytest.approx(float(base))  # a constant weight cancels in the normalisation


def test_class_weights_are_ignored_on_the_binary_path():
    """The two kwargs are not interchangeable: binary takes a scalar, multiclass a vector."""
    logits, label = torch.zeros(2, 1), torch.tensor([1, 0])
    base = classification_loss(logits, label, is_binary=True)
    assert float(classification_loss(logits, label, is_binary=True, class_weights=[9.0, 9.0])) == pytest.approx(
        float(base)
    )
