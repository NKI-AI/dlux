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
"""Tests for the .pack config parser (dlux.data.masks.parse_seg_config)."""

from __future__ import annotations

import pytest
from dlux.data.masks import parse_seg_config

_XML = """<?xml version="1.0"?>
<AifoModelConfiguration>
  <ModelName>tissue-bg</ModelName>
  <Version>1.0</Version>
  <Mpp>8.0</Mpp>
  <Task><Type>Segmentation</Type><MergeMethod>crop</MergeMethod></Task>
  <TileSize><Width>512</Width><Height>512</Height></TileSize>
  <TileOverlap><Width>0</Width><Height>0</Height></TileOverlap>
  <Labels>
    <Label><Name>Background</Name><HexColor>#000000</HexColor><Index>0</Index></Label>
    <Label><Name>Tissue</Name><HexColor>#0000FF</HexColor><Index>1</Index></Label>
  </Labels>
  <Normalization>
    <Mean><Channel0>0.485</Channel0><Channel1>0.456</Channel1><Channel2>0.406</Channel2></Mean>
    <Std><Channel0>0.229</Channel0><Channel1>0.224</Channel1><Channel2>0.225</Channel2></Std>
  </Normalization>
</AifoModelConfiguration>
"""


def test_parse_seg_config():
    cfg = parse_seg_config(_XML)
    assert cfg.mpp == 8.0
    assert cfg.tile_size == (512, 512)
    assert cfg.tile_overlap == (0, 0)
    assert cfg.mean == [0.485, 0.456, 0.406]
    assert cfg.std == [0.229, 0.224, 0.225]


def test_parse_seg_config_rejects_bad_root():
    with pytest.raises(ValueError, match="AifoModelConfiguration"):
        parse_seg_config("<Wrong><Mpp>8.0</Mpp></Wrong>")


def test_parse_seg_config_missing_field():
    # has Normalization but no Mpp -> the missing required scalar is reported.
    xml = (
        "<AifoModelConfiguration>"
        "<TileSize><Width>512</Width><Height>512</Height></TileSize>"
        "<TileOverlap><Width>0</Width><Height>0</Height></TileOverlap>"
        "<Normalization><Mean><Channel0>0.5</Channel0></Mean><Std><Channel0>0.5</Channel0></Std></Normalization>"
        "</AifoModelConfiguration>"
    )
    with pytest.raises(ValueError, match="Mpp"):
        parse_seg_config(xml)
