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
"""Tile images: one precomputed tile image per tile, read from the tile cache.

The image counterpart of ``TileFeatures``: where that reads a precomputed feature vector per tile, this
reads the precomputed tile image (populated by ``bin/extract_tiles.py``), so a trainable-CNN tile
classifier reads cached pixels instead of re-opening the slide. The cache is keyed by the tiling grid
alone (no extractor identity — there is no model). Each sample is one tile, hence a ``fixed`` collate,
and the stream has no scalar ``width`` (a CNN reads pixels, not a feature vector)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from ahcore.data.adapters.cached_tile_image import CachedTileImageAdapter
from ahcore.data.interfaces import ShapePolicy
from ahcore.manifest import DataDescription
from ahcore.transforms.tile import TileTransformFactory

from dlux.modalities.context import ModalityContext


class TileImage:
    """A named cached-tile-image stream. One tile image per sample, addressed by (slide_id, index) in the
    cache; ``key`` distinguishes several such streams in one batch."""

    name = "tile_image"
    collate: ShapePolicy = "fixed"

    def __init__(self, *, key: str, cache_root: Path, data_description: DataDescription) -> None:
        self.key = key
        self.cache_root = Path(cache_root)
        self.data_description = data_description

    @classmethod
    def from_spec(cls, *, key: str, spec: dict, ctx: ModalityContext) -> "TileImage":
        unknown = set(spec) - {"modality", "cache_root"}
        if unknown:
            raise ValueError(f"input '{key}' (tile_image): unknown option(s) {sorted(unknown)}")
        # Defaults to the shared cohort cache root (feature-cache footing); a run overrides it to a
        # node-local drive (e.g. task.inputs.tiles.cache_root=/local/scratch/tile_cache) where the
        # tiles were staged, since the many-small-files tile cache reads far faster off local disk.
        cache_root = spec.get("cache_root") or ctx.cohorts_dir
        if cache_root is None:
            raise ValueError(f"input '{key}' (tile_image) needs cache_root or cohorts_dir to locate the tile cache")
        return cls(key=key, cache_root=Path(cache_root), data_description=ctx.data_description)

    @property
    def width(self) -> Optional[int]:
        """No scalar input width: the model is a CNN over pixels. ``size_model`` skips width injection."""
        return None

    def select(self, entry: dict) -> "torch.Tensor":
        """The model-facing tensor for this stream: the (B, C, H, W) tile-image batch."""
        return entry["image"]

    def adapter(self) -> CachedTileImageAdapter:
        # The cache stores raw pixels; the adapter re-applies for_tile_classification (ImageToTensor +
        # ImageNet normalize), the same transform the streaming path ran inside dlup.
        return CachedTileImageAdapter(
            cache_root=self.cache_root,
            data_description=self.data_description,
            tile_transform=TileTransformFactory.for_tile_classification(),
        )
