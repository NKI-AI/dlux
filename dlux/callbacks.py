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
"""Lightning callbacks specific to dlux training."""

from __future__ import annotations

import logging

from lightning.pytorch import Callback, LightningModule, Trainer

logger = logging.getLogger("dlux.train")


class LogMetrics(Callback):
    """Write each epoch's metrics as plain log records so they reach the Hydra ``train.log``.

    Lightning renders metrics with rich to the console, and its progress bar never reaches a log file,
    so ``train.log`` otherwise holds none of a run's numbers. This emits the same values through the
    standard logger, whose records propagate to the root file handler. One line per validation epoch and
    one after testing, so a finished run's loss and metric curve can be read back from the log alone.
    """

    def _emit(self, trainer: Trainer, phase: str) -> None:
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        metrics = {str(k): float(v) for k, v in trainer.callback_metrics.items()}
        if metrics:
            body = ", ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
            logger.info("[%s] epoch %d | %s", phase, trainer.current_epoch, body)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._emit(trainer, "validate")

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._emit(trainer, "test")
