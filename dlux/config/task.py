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
"""Typed pieces of the ``task`` config group.

The task config is otherwise read field by field in ``tasks/build.py``, each with its own guard. The
blocks that have real structure get a schema here instead, so a typo is a load-time error rather than
a value that silently does nothing.

The task, not the cohort contract, is where a training-time choice about the target belongs: the
contract states what an endpoint is and is consumed by ``build_db``, while these knobs are consumed
only by a run. ``target_normalize`` already lived here; ``loss`` joins it.
"""

from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, field_validator


class LossSpec(BaseModel):
    """Class weighting for a classification endpoint. Ignored by the other objectives.

    Missing targets are always excluded from the loss, independently of this (objective-aware
    sentinel), so this is only about the balance between the classes that are present.
    """

    model_config = ConfigDict(extra="forbid")

    # "balanced" = inverse-frequency from the fold's FIT split (computed at train time,
    # leakage-clean). A dict sets the per-class weights directly, e.g. {"0": 1.0, "1": 4.0}.
    # Weights are relative in both objectives: binary collapses them to the single ratio BCE
    # accepts, and multiclass CE normalises by their sum, so an overall scale cancels either way.
    weight: Union[Literal["none", "balanced"], dict[str, float]] = "none"

    @field_validator("weight")
    @classmethod
    def _weights_must_be_positive(cls, value):
        """A zero or negative weight is a config error, not a way to drop a class.

        A zero or negative weight would divide through to ``None`` (silently unweighted training), so
        reject it."""
        if isinstance(value, dict):
            bad = sorted(k for k, w in value.items() if w <= 0)
            if bad:
                raise ValueError(f"loss weight(s) for class(es) {bad} must be > 0")
        return value
