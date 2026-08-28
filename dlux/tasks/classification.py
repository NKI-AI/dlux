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
"""Classification loss + metrics helpers, torch-only (no ahcore), so testable.

Shared by the classification tasks. Targets equal to ``EXCLUDED`` (-1), i.e.
labels that were missing or fell outside the dataset contract's value map, are
masked out of both the loss and the metrics.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torchmetrics import AUROC, Accuracy, MetricCollection

# Sentinel target for a missing / out-of-contract-map label (never a valid class index).
EXCLUDED = -1


class LabelMapping(dict):
    """Dict that returns ``EXCLUDED`` for unknown keys, so unmapped labels are excluded."""

    def __missing__(self, key: str) -> int:
        return EXCLUDED


def balanced_pos_weight(class_indices: Sequence[int]) -> float:
    """BCE ``pos_weight`` for inverse-frequency (``"balanced"``) binary training: ``n_neg / n_pos``
    over the {0, 1} labels (``EXCLUDED``/-1 ignored). Falls back to 1.0 if a class is absent, a
    single-class train split has nothing to balance."""
    pos = sum(1 for c in class_indices if c == 1)
    neg = sum(1 for c in class_indices if c == 0)
    if pos == 0 or neg == 0:
        return 1.0
    return neg / pos


def resolve_pos_weight(weight_spec: Union[str, dict], class_indices: Sequence[int]) -> Optional[float]:
    """Resolve a binary ``task.loss.weight`` to a BCE ``pos_weight`` (positive-class weight relative
    to the negative), or ``None`` for unweighted.

    ``"none"`` -> None; ``"balanced"`` -> inverse-frequency from ``class_indices`` (the FIT split,
    leakage-clean); a ``{"0": w0, "1": w1}`` dict -> the ratio ``w1 / w0``. BCE takes a single
    relative weight, so only the ratio survives, the same scale-invariance multiclass CE has, where
    normalising by the summed weights cancels it."""
    if weight_spec == "none":
        return None
    if weight_spec == "balanced":
        return balanced_pos_weight(class_indices)
    if isinstance(weight_spec, dict):
        w0, w1 = float(weight_spec.get("0", 1.0)), float(weight_spec.get("1", 1.0))
        if w0 <= 0 or w1 <= 0:
            raise ValueError(f"binary loss weights must be > 0, got {{'0': {w0}, '1': {w1}}}")
        return w1 / w0
    raise ValueError(f"unsupported binary loss weight spec: {weight_spec!r}")


def balanced_class_weights(class_indices: Sequence[int], num_classes: int) -> list[float]:
    """CE ``weight`` vector for inverse-frequency (``"balanced"``) multiclass training:
    ``n / (K * n_c)`` per class over the class indices (``EXCLUDED``/-1 ignored).

    A class absent from the split gets 1.0, there is nothing to balance it against. Every present
    class contributes ``n / K`` of the total weight, so the average weight per sample is 1.0: the loss
    is redistributed between classes, not rescaled."""
    counts = [sum(1 for c in class_indices if c == k) for k in range(num_classes)]
    total = sum(counts)
    if total == 0:
        return [1.0] * num_classes
    return [total / (num_classes * n) if n else 1.0 for n in counts]


def resolve_class_weights(
    weight_spec: Union[str, dict], class_indices: Sequence[int], num_classes: int
) -> Optional[list[float]]:
    """Resolve a multiclass ``task.loss.weight`` to a CE ``weight`` vector of length ``num_classes``,
    or ``None`` for unweighted.

    ``"none"`` -> None; ``"balanced"`` -> inverse-frequency from ``class_indices`` (the FIT split,
    leakage-clean); a ``{"0": w0, "1": w1, ...}`` dict -> those weights, 1.0 for any class it omits."""
    if weight_spec == "none":
        return None
    if weight_spec == "balanced":
        return balanced_class_weights(class_indices, num_classes)
    if isinstance(weight_spec, dict):
        weights = [float(weight_spec.get(str(k), 1.0)) for k in range(num_classes)]
        if any(w <= 0 for w in weights):
            raise ValueError(f"multiclass loss weights must be > 0, got {weights}")
        return weights
    raise ValueError(f"unsupported multiclass loss weight spec: {weight_spec!r}")


def classification_loss(
    logits: torch.Tensor,
    label: torch.Tensor,
    is_binary: bool,
    *,
    pos_weight: Optional[float] = None,
    class_weights: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """BCE (binary, 1 logit) or CE (multiclass, K logits), ignoring ``EXCLUDED`` labels.

    ``pos_weight`` (binary) up-weights the positive class in the BCE; ``class_weights`` (multiclass)
    is the per-class CE weight vector. Both are e.g. inverse-frequency for imbalanced endpoints, and
    ``None`` is the unweighted loss."""
    mask = label != EXCLUDED
    if not bool(mask.any()):
        return logits.sum() * 0.0  # nothing labelled in this batch, zero loss, graph intact
    if is_binary:
        pw = None if pos_weight is None else torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1)[mask], label[mask].float(), pos_weight=pw)
    cw = (
        None
        if class_weights is None
        else torch.as_tensor(list(class_weights), dtype=logits.dtype, device=logits.device)
    )
    return F.cross_entropy(logits[mask], label[mask], weight=cw)


def classification_metric_tensors(
    logits: torch.Tensor, label: torch.Tensor, is_binary: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(probabilities, target)`` for torchmetrics, with ``EXCLUDED`` labels filtered out.

    Probabilities, not logits: torchmetrics infers which it was given from the value range of each
    update, so only an explicit activation makes the metric independent of batching.
    """
    mask = label != EXCLUDED
    if is_binary:
        return torch.sigmoid(logits.squeeze(-1)[mask]), label[mask]
    return torch.softmax(logits[mask], dim=-1), label[mask]


def build_classification_metrics(is_binary: bool, num_classes: int, split: str) -> MetricCollection:
    """AUROC + Accuracy for the given split (`train`/`validate`/`test`)."""
    if is_binary:
        return MetricCollection({f"{split}/accuracy": Accuracy(task="binary"), f"{split}/auroc": AUROC(task="binary")})
    return MetricCollection(
        {
            f"{split}/accuracy": Accuracy(task="multiclass", num_classes=num_classes),
            f"{split}/auroc": AUROC(task="multiclass", num_classes=num_classes),
        }
    )
