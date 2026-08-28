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
"""WSIBulkRNATask, predict a patient's bulk RNA-seq gene-expression vector from one precomputed
slide-feature set (ABMIL, multi-output MSE).

Attention-MIL aggregates each slide's tile-feature bag into a G-dimensional prediction (head width =
the gene-panel size), trained with NaN-masked MSE. The target is the patient's raw RSEM vector (from
the ``RNAExpressionAdapter``), transformed in the task: ``log1p`` then **per-gene z-score** using this
fold's train-split per-gene μ/σ (leakage-clean, passed in) so every gene contributes comparably to the
loss. Predictions are de-standardised back to log1p space for persistence/metrics. A separate task
from WSIRegressionTask (scalar); both plug into the same LightningModule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torchmetrics import MetricCollection

from ahcore.data.interfaces import ShapePolicy
from ahcore.manifest import DataDescription
from ahcore.tasks import ConcreteTaskWithAdapters
from ahcore.tasks.interfaces import TaskWithMetrics

from dlux.config.cohort import ContractField, Objective, SplitCategory
from dlux.modalities import BulkRNA
from dlux.tasks.regression_vector import build_gene_metrics, regression_vector_loss


class WSIBulkRNATask(ConcreteTaskWithAdapters, TaskWithMetrics):
    """Patient gene-expression vector from one precomputed slide-feature set (ABMIL, masked MSE)."""

    name = "wsi_bulk_rna"

    def __init__(
        self,
        target: str,
        contract_field: ContractField,
        data_description: DataDescription,
        inputs: Dict[str, Any],
        matrix_path: Path,
        gene_means: np.ndarray,
        gene_stds: np.ndarray,
        gene_ids: Optional[list] = None,
        gene_panel: Optional[str] = None,
        target_stats: Optional[dict] = None,  # unused: this objective has no build-db statistic
    ) -> None:
        super().__init__()
        if contract_field.objective != Objective.regression_vector:
            raise ValueError(
                f"WSIBulkRNATask handles regression_vector (expression) only, got '{contract_field.objective.value}'."
            )
        self._target = target
        self._data_description = data_description
        self.inputs = dict(inputs)
        self._rna = BulkRNA(  # RNA as the target here
            key="expression",
            matrix_path=matrix_path,
            gene_means=gene_means,
            gene_stds=gene_stds,
            gene_ids=gene_ids,
            panel_name=gene_panel,
        )
        self._streams = (*self.inputs.values(), self._rna)

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
        return {"label": batch[self._rna.key]["expression"].float()}  # (B, G) raw RSEM, NaN for missing

    def _standardize(self, raw: torch.Tensor) -> torch.Tensor:
        """Raw RSEM (B, G) -> log1p -> per-gene z. As the target, a missing patient's NaN is kept so the
        loss masks it rather than learning from an imputed zero."""
        return self._rna.standardize(raw)

    # -- loss / metrics -------------------------------------------------------
    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        return regression_vector_loss(predictions["logits"], self._standardize(targets["label"]))

    def select_metric_tensors(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 2-D (B, G) z-space preds + target, GenePearsonMean accumulates per-gene and masks NaN itself
        # (Pearson is invariant to the affine z-score, so this equals the log1p-space per-gene Pearson).
        return predictions["logits"], self._standardize(targets["label"])

    def train_metrics(self) -> MetricCollection:
        return build_gene_metrics(SplitCategory.FIT, self._rna.n_genes)

    def val_metrics(self) -> MetricCollection:
        return build_gene_metrics(SplitCategory.VALIDATE, self._rna.n_genes)

    def test_metrics(self) -> MetricCollection:
        return build_gene_metrics(SplitCategory.TEST, self._rna.n_genes)

    @property
    def num_classes(self) -> int:
        """The ABMIL head outputs one value per gene (the panel size)."""
        return self._rna.n_genes

    @property
    def gene_means(self) -> torch.Tensor:
        return self._rna.gene_means

    @property
    def gene_stds(self) -> torch.Tensor:
        return self._rna.gene_stds
