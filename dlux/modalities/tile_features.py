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
"""Tile features: one precomputed feature vector per tile, read from the extraction cache.

The cache is keyed by the feature extractor's identity (name + weight hash) together with the tiling
grid, so a task reading these must be handed the same identity extraction wrote with. A slide's tiles
arrive as a variable-length bag, hence the ``padded`` collate policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch

from ahcore.data.adapters.cached_tile_features import CachedSlideFeaturesAdapter
from ahcore.data.interfaces import ShapePolicy
from ahcore.manifest import DataDescription
from ahcore.utils.cache_keys import hash_model_state_dict_light

from dlux.modalities.context import ModalityContext


class TileFeatures:
    """A named tile-feature stream. Several instances (different grids or extractors) can coexist in one
    task; ``key`` is what distinguishes them in the batch."""

    name = "tile_features"
    collate: ShapePolicy = "padded"

    def __init__(
        self,
        *,
        key: str,
        cache_root: Path,
        model_name: str,
        model_hash: str,
        feature_dim: int,
        data_description: DataDescription,
    ) -> None:
        self.key = key
        self.cache_root = Path(cache_root)
        self.model_name = model_name
        self.model_hash = model_hash
        self.feature_dim = int(feature_dim)  # the width this stream contributes to the model
        self.data_description = data_description

    @staticmethod
    def identify(feature_extractor_cfg: Any) -> tuple[str, str, int]:
        """``(model_name, model_hash, feature_dim)`` for an extractor config.

        This triple is the stream's identity: the first two key the feature cache, the third is the width
        of one cached vector. Extraction and training must derive it the same way or the reader looks in a
        namespace the writer never filled, and finds nothing ("no features found", not a crash). Hence
        one function, used by both ``extract_features`` and the training path."""
        extractor = hydra.utils.instantiate(feature_extractor_cfg)
        return extractor.model_name, hash_model_state_dict_light(extractor.state_dict()), int(extractor.feature_dim)

    @classmethod
    def from_spec(cls, *, key: str, spec: dict, ctx: ModalityContext) -> "TileFeatures":
        """Construct from an ``inputs:`` entry, deriving the extractor identity that locates the cache.

        The extractor is never run here. The features are already cached. It is instantiated only to be
        identified, which is why it belongs to this stream rather than to the pipeline."""
        unknown = set(spec) - {"modality"}
        if unknown:
            raise ValueError(f"input '{key}' (tile_features): unknown option(s) {sorted(unknown)}")
        model_name, model_hash, feature_dim = cls.identify(ctx.feature_extractor)
        return cls(
            key=key,
            cache_root=ctx.cohorts_dir,
            model_name=model_name,
            model_hash=model_hash,
            feature_dim=feature_dim,
            data_description=ctx.data_description,
        )

    @property
    def width(self) -> int:
        """The width this stream contributes to the model: one cached tile-feature vector."""
        return self.feature_dim

    def select(self, entry: dict) -> "torch.Tensor":
        """The model-facing tensor for this stream: the padded (B, T, D) tile-feature bag."""
        return entry["features"]

    def adapter(self) -> CachedSlideFeaturesAdapter:
        return CachedSlideFeaturesAdapter(
            cache_root=self.cache_root,
            model_name=self.model_name,
            model_hash=self.model_hash,
            data_description=self.data_description,
        )
