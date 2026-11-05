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
"""WSIClassificationTask, the common case.

Patient-level binary or multiclass classification from the declared input streams, fed to the
configured model. The target is a patient label mapped via the dataset contract (raw value -> class
index, `-1` for missing/out-of-map, excluded from loss + metrics).

Regression, survival and bulk RNA are separate task classes. For other endpoints, write a custom
``Task`` against the ahcore Task protocol. It plugs into the same LightningModule and datamodule.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
from torchmetrics import MetricCollection

from ahcore.data.interfaces import ShapePolicy
from ahcore.manifest import DataDescription
from ahcore.tasks import ConcreteTaskWithAdapters
from ahcore.tasks.interfaces import TaskWithMetrics

from dlux.config.cohort import ContractField, Objective, SplitCategory
from dlux.modalities import PatientField
from dlux.tasks.classification import (
    build_classification_metrics,
    classification_loss,
    classification_metric_tensors,
)
from dlux.tasks.fit_state import FitState
from dlux.tasks.target import build_field_transform


class WSIClassificationTask(ConcreteTaskWithAdapters, TaskWithMetrics):
    """Binary/multiclass patient endpoint from the declared input streams."""

    name = "wsi_classification"

    def __init__(
        self,
        target: str,
        contract_field: ContractField,
        data_description: DataDescription,
        inputs: Dict[str, Any],
        target_stats: Optional[dict] = None,
        pos_weight: Optional[float] = None,
        class_weights: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if contract_field.objective not in (Objective.binary, Objective.multiclass):
            raise ValueError(
                f"WSIClassificationTask handles binary/multiclass objectives only, "
                f"got '{contract_field.objective.value}' — use the appropriate task for that objective."
            )
        self._is_binary = contract_field.objective == Objective.binary
        self._num_classes = contract_field.num_outputs  # 1 (binary) or K (multiclass)
        # Labels are stored under the sheet column, which is the endpoint name only when the contract
        # does not redirect it. `leukocyte_high` sourcing `leukocyte_fraction` is the case that separates
        # them. wsi_survival keys on source.column for the same reason.
        # Cohort injects the contract key as the column of a field that does not name one, and
        # `target` is that key, so the same rule applies to a field built on its own.
        self._label_column = contract_field.source.column or target
        self._label_key = f"patient.{self._label_column}"
        # raw -> class idx via the single target-transform seam (categorical -> LabelMapping, -1 for out-of-map)
        self._label_mapping = build_field_transform(contract_field, target_stats)
        self._data_description = data_description
        self._endpoint = PatientField(
            key="patient_label", fields=[self._label_column], transforms={self._label_column: self._label_mapping}
        )
        self.inputs = dict(inputs)
        self._streams = (*self.inputs.values(), self._endpoint)
        self._pos_weight = pos_weight  # BCE positive-class weight (binary imbalance); None = unweighted
        self._class_weights = class_weights  # CE per-class weights (multiclass imbalance); None = unweighted

    @property
    def streams(self) -> tuple:
        """Every data stream this task reads, the declared inputs plus the target. The public handle for
        anything that must visit all of them regardless of role (e.g. persisting fit-derived state)."""
        return self._streams

    # The resolved class weighting, for a subclass that brings its own loss (see ``compute_loss``).
    @property
    def class_weights(self) -> Optional[Sequence[float]]:
        return self._class_weights

    def fit_state(self) -> FitState:
        return FitState(pos_weight=self._pos_weight, class_weights=self._class_weights)

    # -- batching -------------------------------------------------------------
    def collate_policies(self) -> Dict[str, ShapePolicy]:
        return {stream.key: stream.collate for stream in self._streams}

    @property
    def data_description(self):
        return self._data_description

    def build_adapters(self) -> Dict[str, Any]:
        return {stream.key: stream.adapter() for stream in self._streams}

    # -- input / target selection --------------------------------------------
    def select_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Every declared input, keyed by its name, which is also the model's branch key."""
        return {name: stream.select(batch[name]) for name, stream in self.inputs.items()}

    def select_targets(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {"label": torch.as_tensor(batch["patient_label"][self._label_key], dtype=torch.long)}

    # -- loss / metrics (delegated to the torch-only helpers) -----------------
    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        return classification_loss(
            predictions["logits"],
            targets["label"],
            self._is_binary,
            pos_weight=self._pos_weight,
            class_weights=self._class_weights,
        )

    def select_metric_tensors(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return classification_metric_tensors(predictions["logits"], targets["label"], self._is_binary)

    def train_metrics(self) -> MetricCollection:
        return build_classification_metrics(self._is_binary, self._num_classes, SplitCategory.FIT)

    def val_metrics(self) -> MetricCollection:
        return build_classification_metrics(self._is_binary, self._num_classes, SplitCategory.VALIDATE)

    def test_metrics(self) -> MetricCollection:
        return build_classification_metrics(self._is_binary, self._num_classes, SplitCategory.TEST)

    @property
    def num_classes(self) -> int:
        """Model head output dim (1 binary / K multiclass), the model config must match this."""
        return self._num_classes
