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
"""Tests for dlux.callbacks: the progress-bar selection on and off a terminal, the non-finite loss stop, and
the epoch metrics that reach the log file."""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, RichProgressBar
from lightning.pytorch.trainer import Trainer
from omegaconf import OmegaConf

from dlux.callbacks import (
    FailOnNonFiniteLoss,
    LogMetrics,
    PeriodicProgressLog,
    _fmt,
    has_progress_bar,
    select_callbacks,
)

_CFG = OmegaConf.create(
    {
        "callbacks": {
            "learning_rate_monitor": {"_target_": "lightning.pytorch.callbacks.LearningRateMonitor"},
            "rich_progress_bar": {"_target_": "lightning.pytorch.callbacks.RichProgressBar"},
        }
    }
)


@pytest.mark.parametrize("tty", [True, False])
def test_a_trainer_accepts_the_selected_callbacks(monkeypatch: pytest.MonkeyPatch, tty: bool) -> None:
    monkeypatch.setattr(sys.stdout, "isatty", lambda: tty)

    callbacks = select_callbacks(_CFG)

    kinds = [type(cb) for cb in callbacks]
    if tty:
        assert kinds == [LearningRateMonitor, RichProgressBar]
    else:
        assert kinds == [LearningRateMonitor, PeriodicProgressLog]
    assert has_progress_bar(callbacks) is tty
    trainer = Trainer(
        callbacks=callbacks, enable_progress_bar=has_progress_bar(callbacks), logger=False, accelerator="cpu"
    )
    assert (trainer.progress_bar_callback is not None) is tty


def _trainer(**kwargs: Any) -> Any:
    return cast(
        Any, SimpleNamespace(current_epoch=2, global_step=40, is_global_zero=True, sanity_checking=False, **kwargs)
    )


@pytest.mark.parametrize("outputs", [torch.tensor(float("nan")), {"loss": torch.tensor(float("inf"))}])
def test_a_non_finite_training_loss_stops_the_run(outputs: Any) -> None:
    with pytest.raises(RuntimeError, match="non-finite training loss .* epoch 2, global step 40, batch 5"):
        FailOnNonFiniteLoss().on_train_batch_end(_trainer(), cast(Any, None), outputs, None, 5)


def test_a_finite_training_loss_passes() -> None:
    FailOnNonFiniteLoss().on_train_batch_end(_trainer(), cast(Any, None), {"loss": torch.tensor(0.3)}, None, 0)


def test_epoch_metrics_reach_the_log_except_during_the_sanity_check(caplog: pytest.LogCaptureFixture) -> None:
    metrics = {"validate/loss": torch.tensor(0.25), "validate/auroc": torch.tensor(0.5)}
    with caplog.at_level(logging.INFO, logger="dlux.train"):
        LogMetrics().on_validation_epoch_end(_trainer(callback_metrics=metrics), cast(Any, None))
        LogMetrics().on_validation_epoch_end(
            cast(Any, SimpleNamespace(is_global_zero=True, sanity_checking=True, callback_metrics=metrics)),
            cast(Any, None),
        )
    assert [r.getMessage() for r in caplog.records] == [
        "[validate] epoch 2 | validate/auroc=0.5000, validate/loss=0.2500"
    ]


def test_durations_print_as_clock_time_and_an_unknown_eta_as_a_question_mark() -> None:
    assert (_fmt(75), _fmt(3725), _fmt(float("inf")), _fmt(float("nan"))) == ("1:15", "1:02:05", "?", "?")
