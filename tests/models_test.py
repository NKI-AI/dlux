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
"""Tests for run-time model sizing.

The link between a data stream and the model slot it feeds is the INPUT NAME, so these check that a
width lands where its name says and nowhere else — and that a mismatch is an error rather than a model
quietly built from its config defaults."""

from __future__ import annotations

import pytest
from dlux.models import size_model
from omegaconf import OmegaConf


class _Stream:
    def __init__(self, width: int) -> None:
        self.width = width


class _Task:
    def __init__(self, num_classes: int, **inputs: int) -> None:
        self.num_classes = num_classes
        self.inputs = {name: _Stream(w) for name, w in inputs.items()}


def _fusion_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "lit_module": {
                "model": {
                    "num_classes": 1,
                    "branches": {
                        "tile_features": {"feature_dim": None},  # ABMILEncoder spells it feature_dim
                        "bulk_rna": {"input_dim": 1},  # MLPEncoder spells it input_dim
                    },
                }
            }
        }
    )


def _flat_cfg() -> OmegaConf:
    return OmegaConf.create(
        {"lit_module": {"model": {"num_classes": 1, "input_key": "tile_features", "feature_dim": None}}}
    )


def test_branch_widths_are_written_by_input_name():
    cfg = _fusion_cfg()
    size_model(cfg, _Task(4, tile_features=1536, bulk_rna=2000))
    assert cfg.lit_module.model.num_classes == 4
    assert cfg.lit_module.model.branches.tile_features.feature_dim == 1536
    assert cfg.lit_module.model.branches.bulk_rna.input_dim == 2000  # each encoder's own spelling


def test_single_input_model_is_sized_via_input_key():
    cfg = _flat_cfg()
    size_model(cfg, _Task(2, tile_features=1536))
    assert cfg.lit_module.model.feature_dim == 1536


def test_an_input_the_model_ignores_is_allowed():
    """rna_only declares a tile_features input its model has no branch for — it is loaded so the
    min_tiles filter drops the same patients as the other arms."""
    cfg = OmegaConf.create({"lit_module": {"model": {"num_classes": 1, "branches": {"bulk_rna": {"input_dim": 1}}}}})
    size_model(cfg, _Task(4, tile_features=1536, bulk_rna=2000))
    assert cfg.lit_module.model.branches.bulk_rna.input_dim == 2000


def test_a_branch_with_no_matching_input_is_an_error():
    """The dangerous direction: a branch nothing feeds would be built from its config default."""
    cfg = _fusion_cfg()
    with pytest.raises(ValueError, match=r"branch\(es\) \['bulk_rna'\]"):
        size_model(cfg, _Task(4, tile_features=1536))


def test_a_branch_declaring_no_width_field_is_an_error():
    cfg = OmegaConf.create({"lit_module": {"model": {"num_classes": 1, "branches": {"tile_features": {"x": 1}}}}})
    with pytest.raises(ValueError, match="feature_dim"):
        size_model(cfg, _Task(2, tile_features=1536))


def test_an_input_key_with_no_matching_input_is_an_error():
    """The single-model mirror of the branch case. Without this the width is simply never written and
    the run dies at forward time with a KeyError, far from the config that caused it."""
    cfg = _flat_cfg()
    with pytest.raises(ValueError, match="input_key 'tile_features'"):
        size_model(cfg, _Task(2, bulk_rna=2000))


def test_a_model_with_no_input_key_is_allowed():
    """Not every single-input model names its stream — only a MISMATCH is the error."""
    cfg = OmegaConf.create({"lit_module": {"model": {"num_classes": 1}}})
    size_model(cfg, _Task(3, tile_features=1536))
    assert cfg.lit_module.model.num_classes == 3
