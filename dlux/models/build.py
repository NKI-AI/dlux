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
"""Model construction: size a configured model from the task, then instantiate it.

Some widths are only knowable at run time: the endpoint's head width, and the width of each input
stream (a cached tile-feature vector's length, the active gene panel's size). They are written into
``cfg`` rather than passed as arguments, so the resolved config snapshotted as the run's ``config.yaml``
artifact records exactly what was built.

The link between a stream and the model slot it feeds is the **input name**. A fusion model has one
named branch per input, so ``inputs.bulk_rna`` sizes ``model.branches.bulk_rna``. A single-input model
names the stream it reads in ``input_key``. This file knows only what an input is called and how wide it
is, so a new modality needs no change here.
"""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig

# Encoders disagree about what to call their input width (ABMILEncoder takes `feature_dim`, MLPEncoder
# takes `input_dim`), so the field is discovered from what the config declares rather than mapped here.
_WIDTH_KEYS = ("feature_dim", "input_dim")


def _write_width(node: Any, width: int, where: str) -> None:
    for key in _WIDTH_KEYS:
        if key in node:
            node[key] = width
            return
    raise ValueError(
        f"{where} declares none of {list(_WIDTH_KEYS)}, so its input width cannot be set. "
        f"Give the encoder one of those fields (it may be null — it is filled at run time)."
    )


def build_lit_module(cfg: DictConfig, task: Any) -> Any:
    """Size the configured model from ``task`` and its input streams, then instantiate it."""
    size_model(cfg, task)
    return hydra.utils.instantiate(cfg.lit_module, task=task)


def size_model(cfg: DictConfig, task: Any) -> None:
    """Write the run-time widths into ``cfg.lit_module.model`` in place.

    Separate from instantiation so it can be checked without constructing a model. Raises when the model
    names an input the task does not provide, whether as a branch or as a single model's ``input_key``.
    Such a stream never receives data, so the model would otherwise be built from whatever its config
    defaulted to and fail at forward time instead of here."""
    model = cfg.lit_module.model
    model.num_classes = task.num_classes  # head width, from the endpoint
    branches = model.get("branches")

    if branches is not None:
        unfed = sorted(set(branches) - set(task.inputs))
        if unfed:
            raise ValueError(
                f"model declares branch(es) {unfed} that the task does not provide as inputs "
                f"(inputs: {sorted(task.inputs)}). Branch names must match input names."
            )
    else:
        input_key = str(model.get("input_key", ""))
        if input_key and input_key not in task.inputs:
            raise ValueError(
                f"model declares input_key '{input_key}' that the task does not provide as an input "
                f"(inputs: {sorted(task.inputs)}). input_key must match an input name."
            )

    for name, stream in task.inputs.items():
        width = getattr(stream, "width", None)
        if width is None:
            continue
        if branches is not None:
            if name in branches:  # an input without a branch is legitimate: the model simply ignores it
                _write_width(branches[name], int(width), f"model.branches.{name}")
        elif str(model.get("input_key", "")) == name:
            _write_width(model, int(width), "model")
