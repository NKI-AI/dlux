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
"""WSIRegressionTask, a patient-level continuous endpoint from one precomputed slide-feature set.

Attention-MIL (ABMIL) aggregates each slide's tile-feature bag into one scalar prediction (the head
outputs 1 value, used raw, no activation), trained with MSE. The target is a patient value mapped
through the contract's numeric transform (raw -> float, NaN for missing/out-of-map, masked in the
loss + metrics). A separate task from WSIClassificationTask; both plug into the same LightningModule.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torchmetrics import MetricCollection

from ahcore.data.interfaces import ShapePolicy
from ahcore.manifest import DataDescription
from ahcore.tasks import ConcreteTaskWithAdapters
from ahcore.tasks.interfaces import TaskWithMetrics

from dlux.config.cohort import ContractField, Objective, SplitCategory
from dlux.modalities import PatientField
from dlux.tasks.regression import build_regression_metrics, regression_loss, regression_metric_tensors
from dlux.tasks.target import build_field_transform


class WSIRegressionTask(ConcreteTaskWithAdapters, TaskWithMetrics):
    """Continuous patient endpoint from one precomputed slide-feature set (ABMIL, MSE)."""

    name = "wsi_regression"

    def __init__(
        self,
        target: str,
        contract_field: ContractField,
        data_description: DataDescription,
        inputs: Dict[str, Any],
        target_stats: Optional[dict] = None,
        target_mean: float = 0.0,
        target_std: float = 1.0,
    ) -> None:
        super().__init__()
        if contract_field.objective != Objective.regression:
            raise ValueError(
                f"WSIRegressionTask handles regression objectives only, got '{contract_field.objective.value}' — "
                f"use WSIClassificationTask for binary/multiclass."
            )
        self._target = target
        # Keyed on the sheet column, not the endpoint name, the contract may redirect one to the other.
        self._label_column = contract_field.source.column
        self._label_key = f"patient.{self._label_column}"
        # raw value -> float target (NaN for missing/out-of-map) via the single target-transform seam
        self._target_transform = build_field_transform(contract_field, target_stats)
        self._data_description = data_description
        self._endpoint = PatientField(
            key="patient_label", fields=[self._label_column], transforms={self._label_column: self._target_transform}
        )
        self.inputs = dict(inputs)
        self._streams = (*self.inputs.values(), self._endpoint)
        # Optional target standardisation: the head learns in z-space (μ,σ from this fold's train split,
        # passed in leakage-clean), so the loss target is standardised and predictions are de-standardised
        # back to raw units for metrics + persistence. Defaults (0, 1) = identity (raw-scale MSE).
        self._target_mean = float(target_mean)
        self._target_std = float(target_std) if float(target_std) > 0.0 else 1.0

    @property
    def streams(self) -> tuple:
        """Every data stream this task reads, the declared inputs plus the target. The public handle for
        anything that must visit all of them regardless of role (e.g. persisting fit-derived state)."""
        return self._streams

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
        return {"label": torch.as_tensor(batch["patient_label"][self._label_key], dtype=torch.float)}

    # -- loss / metrics -------------------------------------------------------
    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Standardise the target so the head trains in z-space; NaN (missing) stays NaN and is masked.
        target = (targets["label"] - self._target_mean) / self._target_std
        return regression_loss(predictions["logits"], target)

    def select_metric_tensors(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # De-standardise predictions to raw units so MAE/RMSE are comparable across experiments.
        preds_raw = predictions["logits"] * self._target_std + self._target_mean
        return regression_metric_tensors(preds_raw, targets["label"])

    @property
    def target_mean(self) -> float:
        return self._target_mean

    @property
    def target_std(self) -> float:
        return self._target_std

    def train_metrics(self) -> MetricCollection:
        return build_regression_metrics(SplitCategory.FIT)

    def val_metrics(self) -> MetricCollection:
        return build_regression_metrics(SplitCategory.VALIDATE)

    def test_metrics(self) -> MetricCollection:
        return build_regression_metrics(SplitCategory.TEST)

    @property
    def num_classes(self) -> int:
        """The ABMIL head outputs a single scalar for regression."""
        return 1
