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
import math
import sys
import time
from typing import Any

import hydra
import torch
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.callbacks import ProgressBar, RichProgressBar
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig

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


def _fmt(seconds: float) -> str:
    """H:MM:SS (or M:SS under an hour) for a duration; ``?`` for a non-finite ETA."""
    if seconds != seconds or seconds == float("inf"):
        return "?"
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class PeriodicProgressLog(Callback):
    """Emit one INFO line at most every ``every_n_seconds`` of training (batch index, rate, ETA)
    and one at each validation's start and end (with ``validate/loss``). The log's own timestamp
    prefix carries the wall-clock time."""

    def __init__(self, every_n_seconds: float = 60.0) -> None:
        self._every = every_n_seconds
        self._epoch_start = 0.0
        self._last_log = 0.0
        self._val_start = 0.0

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._epoch_start = self._last_log = time.monotonic()
        logger.info(f"train epoch {trainer.current_epoch}: {trainer.num_training_batches} batches")

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        now = time.monotonic()
        if now - self._last_log < self._every:
            return
        self._last_log = now
        total = trainer.num_training_batches
        done = batch_idx + 1
        elapsed = now - self._epoch_start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else float("inf")
        logger.info(
            f"train epoch {trainer.current_epoch} | batch {done}/{total} ({100 * done / total:.1f}%) "
            f"| {rate:.2f} it/s | elapsed {_fmt(elapsed)} | ETA {_fmt(eta)}"
        )

    def on_validation_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        self._val_start = time.monotonic()
        logger.info(f"validation @ step {trainer.global_step} started")

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking:
            return
        msg = f"validation @ step {trainer.global_step} done in {_fmt(time.monotonic() - self._val_start)}"
        val_loss = trainer.callback_metrics.get("validate/loss")
        if val_loss is not None:
            msg += f" | validate/loss {float(val_loss):.4f}"
        logger.info(msg)


class FailOnNonFiniteLoss(Callback):
    """Stop the run at the first training step whose loss is not finite."""

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        value = float(loss.detach()) if torch.is_tensor(loss) else float(loss)  # type: ignore[arg-type]
        if not math.isfinite(value):
            raise RuntimeError(
                f"non-finite training loss {value} at epoch {trainer.current_epoch}, global step "
                f"{trainer.global_step}, batch {batch_idx}"
            )


def select_callbacks(cfg: DictConfig) -> list[Callback]:
    """Instantiate ``cfg.callbacks``. Keep ``RichProgressBar`` on a TTY; otherwise drop it and
    append ``PeriodicProgressLog`` so a redirected slurm log still shows progress."""
    callbacks = list(hydra.utils.instantiate(cfg.callbacks).values())
    if sys.stdout.isatty():
        return callbacks
    callbacks = [cb for cb in callbacks if not isinstance(cb, RichProgressBar)]
    callbacks.append(PeriodicProgressLog())
    return callbacks


def has_progress_bar(callbacks: list[Callback]) -> bool:
    """Whether ``callbacks`` hold a progress bar. Pass it as the Trainer's ``enable_progress_bar``:
    Lightning adds a bar of its own when that flag is on and the callbacks hold none, and rejects a bar
    in the callbacks when it is off."""
    return any(isinstance(cb, ProgressBar) for cb in callbacks)
