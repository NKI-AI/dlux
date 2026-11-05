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
"""Tests for dlux.models.tile_cnn: pretrained weights load onto the backbone, and a frozen stage stays frozen."""

from pathlib import Path

import pytest
import torch
from torch import nn

from dlux.models.tile_cnn import TileCNN


def _torchvision_shaped_state(model: TileCNN) -> dict[str, torch.Tensor]:
    """The backbone's weights under torchvision's bare ResNet18 keys, with a 1000-class ``fc``."""
    state = {
        key: torch.randn_like(value) if value.is_floating_point() else value
        for key, value in model.backbone.state_dict().items()
    }
    state["fc.weight"], state["fc.bias"] = torch.randn(1000, 512), torch.randn(1000)
    return state


def test_a_local_state_dict_loads_onto_the_backbone_and_drops_the_imagenet_head(tmp_path: Path) -> None:
    state = _torchvision_shaped_state(TileCNN(input_key="tiles"))
    torch.save(state, tmp_path / "resnet18.pth")

    model = TileCNN(input_key="tiles", weights=str(tmp_path / "resnet18.pth"))

    loaded = model.backbone.state_dict()
    assert all(torch.equal(loaded[key], state[key]) for key in loaded)
    assert model.fc.out_features == 1
    assert model({"tiles": torch.rand(2, 3, 64, 64)})["logits"].shape == (2, 1)


def test_a_state_dict_with_other_names_is_refused(tmp_path: Path) -> None:
    torch.save({"encoder.conv1.weight": torch.zeros(64, 3, 7, 7)}, tmp_path / "other.pth")
    with pytest.raises(ValueError, match="unexpected keys"):
        TileCNN(input_key="tiles", weights=str(tmp_path / "other.pth"))


def test_frozen_stages_keep_their_weights_and_batchnorm_statistics_in_training() -> None:
    model = TileCNN(input_key="tiles", freeze_through="layer3")
    model.train()

    assert not any(p.requires_grad for p in model.backbone.layer3.parameters())
    assert all(p.requires_grad for p in model.backbone.layer4.parameters())
    assert not any(m.training for m in model.backbone.layer3.modules() if isinstance(m, nn.BatchNorm2d))
    assert all(m.training for m in model.backbone.layer4.modules() if isinstance(m, nn.BatchNorm2d))
