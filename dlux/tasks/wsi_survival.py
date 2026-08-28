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
"""WSISurvivalTask, a time-to-event patient endpoint from one precomputed slide-feature set (ABMIL).

Reads two patient columns (an event indicator and a follow-up time) and models them with the
discrete-time hazard formulation (see ``dlux.tasks.survival``): the ABMIL head emits one hazard logit
per time bin, the loss is the discrete-time NLL, and the streamed metric is Harrell's C-index. The
bin edges are the fold's fit-split quantiles (resolved in ``build_task``, leakage-clean) so time is
discretised identically for every patient. H&E-only.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torchmetrics import MetricCollection

from ahcore.data.interfaces import ShapePolicy
from ahcore.manifest import DataDescription
from ahcore.tasks import ConcreteTaskWithAdapters
from ahcore.tasks.interfaces import TaskWithMetrics

from dlux.config.cohort import ContractField, Objective, SplitCategory
from dlux.modalities import PatientField
from dlux.tasks.survival import build_survival_metrics, survival_nll, survival_risk


def _as_float(value: Any) -> float:
    return float(value)


class WSISurvivalTask(ConcreteTaskWithAdapters, TaskWithMetrics):
    """Time-to-event patient endpoint from one precomputed slide-feature set (ABMIL, discrete-time NLL)."""

    name = "wsi_survival"

    def __init__(
        self,
        target: str,
        contract_field: ContractField,
        data_description: DataDescription,
        inputs: Dict[str, Any],
        n_bins: int,
        time_edges: np.ndarray | None = None,
        target_stats: Optional[dict] = None,  # unused: this objective has no build-db statistic
    ) -> None:
        super().__init__()
        if contract_field.objective != Objective.survival:
            raise ValueError(
                f"WSISurvivalTask handles the survival objective only, got '{contract_field.objective.value}'."
            )
        self._target = target
        self._event_col = contract_field.source.column
        self._time_col = contract_field.source.time_column
        self._event_key = f"patient.{self._event_col}"
        self._time_key = f"patient.{self._time_col}"
        self._n_bins = int(n_bins)
        # Interior bin edges (n_bins-1 cut points); bucketize(time) -> bin index in [0, n_bins-1].
        # Training-only state, derived from this fold's fit split. Scoring needs only the risk score
        # and the raw (time, event), so a task built to score an external cohort, which has no fit
        # split, legitimately has none, and says so if asked to compute a loss.
        self._edges = None if time_edges is None else torch.as_tensor(np.asarray(time_edges), dtype=torch.float32)
        self._data_description = data_description
        self._endpoint = PatientField(
            key="patient_label",
            fields=[self._event_col, self._time_col],
            transforms={self._event_col: _as_float, self._time_col: _as_float},
        )
        self.inputs = dict(inputs)
        self._streams = (*self.inputs.values(), self._endpoint)

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
        event = torch.as_tensor(batch["patient_label"][self._event_key], dtype=torch.float32)
        time = torch.as_tensor(batch["patient_label"][self._time_key], dtype=torch.float32)
        targets = {"event": event, "time": time}
        if self._edges is not None:
            targets["bin"] = torch.bucketize(time, self._edges.to(time.device))  # -> [0, n_bins-1]
        return targets

    # -- loss / metrics -------------------------------------------------------
    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "bin" not in targets:
            raise ValueError(
                "WSISurvivalTask was built without `time_edges`, so follow-up time cannot be discretised "
                "and the discrete-time NLL is undefined. That is the inference-only construction (an "
                "external cohort has no fit split to derive edges from); training must pass time_edges."
            )
        return survival_nll(predictions["logits"], targets["bin"], targets["event"])

    def select_metric_tensors(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # (risk, [time, event]), the C-index metric needs all three, packed into the 2-tensor seam.
        risk = survival_risk(predictions["logits"])
        target = torch.stack([targets["time"], targets["event"]], dim=-1)
        return risk, target

    def train_metrics(self) -> MetricCollection:
        return build_survival_metrics(SplitCategory.FIT)

    def val_metrics(self) -> MetricCollection:
        return build_survival_metrics(SplitCategory.VALIDATE)

    def test_metrics(self) -> MetricCollection:
        return build_survival_metrics(SplitCategory.TEST)

    @property
    def num_classes(self) -> int:
        """Model head output dim = the number of discrete-time hazard bins."""
        return self._n_bins

    @property
    def time_edges(self) -> np.ndarray | None:
        """This fold's interior bin edges (``n_bins - 1`` cut points, in the time column's units), or
        None for the inference-only construction. Public because the per-bin hazard logits are
        meaningless without them: every fold quantiles its own fit split, so bin j spans a different
        interval in each. Persisted alongside the predictions."""
        return None if self._edges is None else self._edges.numpy()
