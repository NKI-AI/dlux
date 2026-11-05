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
"""Tile predictions pooled to slides: a slide's score is the mean of its tiles' probabilities.

A tile with a negative (excluded) label is left out, and every tile of a slide must carry the same label.

``TilePooler`` pools the test tiles during ``trainer.test`` into the per-slide prediction dict that
``collect_predictions`` returns for a slide task. ``SlidePooledValAUROC`` pools the validation tiles the same
way and logs the AUROC over slides as ``validate/auroc_slide``, a curve to watch. Checkpoint selection stays on
``validate/loss``.

The pooling is defined for a binary endpoint (one logit per tile).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def _add_label(labels: dict[str, int], slide_id: str, label: int) -> None:
    if labels.setdefault(slide_id, label) != label:
        raise ValueError(f"Slide {slide_id} has tiles labelled {labels[slide_id]} and {label}.")


class TilePooler(pl.Callback):
    """Pools each test tile's probability and label into its slide during ``trainer.test``. Each tile's slide
    comes from its ``tile_ref``. The pooled logit is the logit of the pooled probability, so the two agree."""

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._prob_sum: dict[str, float] = defaultdict(float)
        self._complement_sum: dict[str, float] = defaultdict(float)
        self._count: dict[str, int] = defaultdict(int)
        self._label: dict[str, int] = {}

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        logits = outputs["predictions"]["logits"].detach().reshape(-1).double().cpu()
        # sigmoid(-x) is 1 - sigmoid(x) without the cancellation, so a confident tile keeps a finite logit.
        probs, complements = torch.sigmoid(logits).tolist(), torch.sigmoid(-logits).tolist()
        labels = outputs["targets"]["label"].detach().reshape(-1).cpu().tolist()
        for i, tile_ref in enumerate(batch["tile_ref"]):
            label = int(labels[i])
            if label < 0:
                continue
            sid = tile_ref.slide_id
            _add_label(self._label, sid, label)
            self._prob_sum[sid] += probs[i]
            self._complement_sum[sid] += complements[i]
            self._count[sid] += 1

    def predictions(self, patient_of: dict[str, str], target_key: str) -> dict[str, Any]:
        """The pooled slides as the dict ``collect_predictions`` returns, with each slide's patient from
        ``patient_of``. Raises when no labelled tile was pooled, or a slide has no patient."""
        slides = sorted(self._count)
        if not slides:
            raise ValueError("No labelled test tile was pooled.")
        unknown = [s for s in slides if s not in patient_of]
        if unknown:
            raise ValueError(f"{len(unknown)} pooled slide(s) have no patient in the test split, e.g. {unknown[:3]}.")
        return {
            "slide_ids": slides,
            "patient_codes": [patient_of[s] for s in slides],
            "logits": np.array(
                [np.log(self._prob_sum[s]) - np.log(self._complement_sum[s]) for s in slides], dtype=np.float64
            ),
            "probs": np.array([self._prob_sum[s] / self._count[s] for s in slides], dtype=np.float64),
            "labels": np.array([self._label[s] for s in slides], dtype=np.int64),
            "endpoint_type": "binary",
            "num_classes": 1,
            "target_key": target_key,
            "attention": [],
            "time_edges": None,
        }


class SlidePooledValAUROC(pl.Callback):
    """Logs the AUROC over validation slides, each scored as the mean of its tiles' probabilities, as
    ``validate/auroc_slide``. The value is NaN when the AUROC is undefined: fewer than two slides, one class only,
    or a non-finite score."""

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._prob_sum: dict[str, float] = defaultdict(float)
        self._count: dict[str, int] = defaultdict(int)
        self._label: dict[str, int] = {}

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        probs = torch.sigmoid(outputs["predictions"]["logits"].detach().reshape(-1)).cpu().numpy()
        labels = outputs["targets"]["label"].detach().reshape(-1).cpu().numpy()
        for i, tile_ref in enumerate(batch["tile_ref"]):
            label = int(labels[i])
            if label < 0:
                continue
            sid = tile_ref.slide_id
            _add_label(self._label, sid, label)
            self._prob_sum[sid] += float(probs[i])
            self._count[sid] += 1

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        slides = list(self._count)
        y = np.array([self._label[s] for s in slides])
        x = np.array([self._prob_sum[s] / self._count[s] for s in slides])
        if len(slides) < 2 or len(set(y.tolist())) != 2 or not np.isfinite(x).all():
            auroc = float("nan")
        else:
            auroc = float(roc_auc_score(y, x))
        pl_module.log("validate/auroc_slide", auroc, prog_bar=True)
