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
"""TileClassificationTask: a binary patient endpoint from raw tiles, supervised per tile.

The tile counterpart of ``WSIClassificationTask``. Each sample is one cached tile image, and its target is
the patient label, given to every tile of the patient's slides (weak labelling). The model is a trainable
tile classifier, and the loss is a per-tile BCE. The patient score, the mean of a slide's tile
probabilities, is formed at eval by ``dlux.eval.tile_pooling``, so this task is a plain tile classifier.

It reuses the classification machinery: the same contract-field transform and the same loss and metric
helpers. Its metrics are per tile and only for monitoring. The reported per-patient number comes from the
pooled predictions.
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
from dlux.modalities.patient_field_tile import PatientFieldTile
from dlux.tasks.classification import (
    build_classification_metrics,
    classification_loss,
    classification_metric_tensors,
)
from dlux.tasks.fit_state import FitState
from dlux.tasks.target import build_field_transform


class TileClassificationTask(ConcreteTaskWithAdapters, TaskWithMetrics):
    """Binary patient endpoint from raw tiles, supervised per tile (per-tile BCE, patient label on every tile)."""

    name = "tile_classification"
    sample_unit = "tile"

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
        if contract_field.objective != Objective.binary:
            raise ValueError(
                f"TileClassificationTask handles binary objectives only, got "
                f"'{contract_field.objective.value}'. The tile classifier and its mean-probability pooling are "
                f"defined for a single-logit endpoint."
            )
        self._is_binary = True
        self._num_classes = contract_field.num_outputs  # 1 for binary
        self._label_column = contract_field.source.column or target
        self._label_key = f"patient.{self._label_column}"
        self._label_mapping = build_field_transform(contract_field, target_stats)
        self._data_description = data_description
        # The target: the patient label on every tile, so a TileAdapter.
        self._endpoint = PatientFieldTile(
            key="patient_label", fields=[self._label_column], transforms={self._label_column: self._label_mapping}
        )
        self.inputs = dict(inputs)
        self._streams = (*self.inputs.values(), self._endpoint)
        self._pos_weight = pos_weight
        self._class_weights = class_weights

    @property
    def streams(self) -> tuple:
        return self._streams

    def fit_state(self) -> FitState:
        return FitState(pos_weight=self._pos_weight, class_weights=self._class_weights)

    def collate_policies(self) -> Dict[str, ShapePolicy]:
        return {stream.key: stream.collate for stream in self._streams}

    @property
    def data_description(self):
        return self._data_description

    def build_adapters(self) -> Dict[str, Any]:
        return {stream.key: stream.adapter() for stream in self._streams}

    def select_inputs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Every declared input, keyed by its name, which is also the model's input_key."""
        return {name: stream.select(batch[name]) for name, stream in self.inputs.items()}

    def select_targets(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {"label": torch.as_tensor(batch["patient_label"][self._label_key], dtype=torch.long)}

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
        # Per-tile probabilities against the patient label: a monitoring metric, not the reported number.
        return classification_metric_tensors(predictions["logits"], targets["label"], self._is_binary)

    def train_metrics(self) -> MetricCollection:
        return build_classification_metrics(self._is_binary, self._num_classes, SplitCategory.FIT)

    def val_metrics(self) -> MetricCollection:
        return build_classification_metrics(self._is_binary, self._num_classes, SplitCategory.VALIDATE)

    def test_metrics(self) -> MetricCollection:
        return build_classification_metrics(self._is_binary, self._num_classes, SplitCategory.TEST)

    @property
    def num_classes(self) -> int:
        return self._num_classes
