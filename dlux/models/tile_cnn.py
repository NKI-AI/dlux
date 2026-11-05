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
"""A trainable ResNet18 tile classifier, the model of ``TileClassificationTask``.

One raw tile in, one logit out. The encoder is trainable, optionally frozen up to a named stage
(``freeze_through``). The module names match torchvision's ResNet18, so the official ImageNet state_dict
loads onto it directly (its 1000-class ``fc`` is dropped).
That is why the network is written out here: it needs no torchvision dependency.

``weights``:
  - ``None``            random initialisation.
  - ``"imagenet"``      the official torchvision ResNet18 weights, fetched+cached via torch.hub.
  - a filesystem path   a local ``.pth`` state_dict (an offline node with no torch.hub access).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

from ahcore.models.interfaces import Classifier

logger = logging.getLogger(__name__)

# Official torchvision ResNet18 ImageNet-1k weights (IMAGENET1K_V1). Fetched with torch.hub, so no
# torchvision dependency. Cached under the torch hub dir after the first download.
_IMAGENET_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """The ResNet18/34 residual block (torchvision-named submodules)."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class _ResNet18Backbone(nn.Module):
    """ResNet18 up to (and including) the global average pool. Names match torchvision so its
    state_dict loads directly. Emits a (B, 512) feature per image."""

    def __init__(self) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [BasicBlock(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class TileCNN(nn.Module, Classifier):
    """Trainable ResNet18 + single linear head, tile in -> logit out.

    ``forward`` reads the raw tile batch under ``input_key`` (shape (B, C, H, W)) and returns
    ``{"logits": (B, num_classes)}``, the same contract as ABMIL, so it plugs into the shared
    LightningModule unchanged. The patient score (the mean of a slide's tile probabilities) is formed at
    eval by ``dlux.eval.tile_pooling``."""

    def __init__(
        self,
        input_key: str,
        num_classes: int = 1,
        weights: Optional[str] = None,
        freeze_through: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._input_key = input_key
        self._frozen_stages: tuple[str, ...] = ()
        self.backbone = _ResNet18Backbone()
        self.fc = nn.Linear(512, num_classes)
        if weights is not None:
            self._load_pretrained(weights)
        if freeze_through is not None:
            self._freeze_through(freeze_through)

    def _load_pretrained(self, weights: str) -> None:
        """Load torchvision-compatible ResNet18 weights into the backbone, dropping the 1000-class fc."""
        if weights == "imagenet":
            state = torch.hub.load_state_dict_from_url(_IMAGENET_URL, progress=False, map_location="cpu")
        else:
            path = Path(weights)
            if not path.is_file():
                raise FileNotFoundError(f"TileCNN weights='{weights}' is neither 'imagenet' nor an existing .pth file.")
            state = torch.load(path, map_location="cpu")
            state = state.get("state_dict", state) if isinstance(state, dict) else state
        # torchvision keys are bare (conv1.weight, layer1.0.conv1.weight, fc.weight). Our backbone holds
        # everything but fc, so drop fc.* and load the rest into self.backbone as-is.
        backbone_state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        if unexpected:
            raise ValueError(f"TileCNN pretrained load: unexpected keys {unexpected[:5]} (naming mismatch).")
        if missing:
            raise ValueError(f"TileCNN pretrained load: missing backbone keys {missing[:5]}.")
        logger.info("TileCNN: loaded pretrained backbone from %s", weights)

    def _freeze_through(self, stage: str) -> None:
        """Freeze every backbone stage up to and including ``stage``. The rest, and fc, stay trainable."""
        stages = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")
        if stage not in stages:
            raise ValueError(f"freeze_through='{stage}' must be one of {stages}.")
        self._frozen_stages = stages[: stages.index(stage) + 1]
        for name in self._frozen_stages:
            for p in getattr(self.backbone, name).parameters():
                p.requires_grad_(False)
        logger.info("TileCNN: froze backbone stages %s; training the remainder + fc.", self._frozen_stages)

    def _apply_frozen_bn_eval(self) -> None:
        for name in self._frozen_stages:
            for module in getattr(self.backbone, name).modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

    def train(self, mode: bool = True) -> "TileCNN":
        super().train(mode)
        self._apply_frozen_bn_eval()  # frozen BatchNorm stays in eval so its running stats stay fixed
        return self

    def forward(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        x = sample[self._input_key]  # (B, C, H, W)
        return {"logits": self.fc(self.backbone(x))}  # (B, num_classes)
