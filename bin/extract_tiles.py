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
"""Cache a cohort's raw tissue tiles, so that a tile task trains on cached tiles instead of the slides.

    extract_tiles study=tcga_subtyping cohort=tcga_brca_coad_christiana
    extract_tiles study=tcga_subtyping cohort=tcga_brca_coad_christiana tiling=<grid> tile_cache_root=<dir> limit=3

The cache is split-agnostic: every slide in the cohort's db, at the grid of ``tiling``. ``study`` only
locates the db (``studies/<study>/db/<cohort>/manifest.db``). ``limit`` caches the first N slides, and the run then
ends INCOMPLETE. Training reads the cache through ``task.inputs.tiles.cache_root``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ahcore.data.adapters.cached_tile_image import precompute_tile_images
from ahcore.data.mm_dataset import MultiModalDataset
from ahcore.transforms.tile import TileTransformFactory
from ahcore.utils.io import print_config

from dlux.config.cohort import Cohort, SplitCategory, Study
from dlux.config.resolvers import register_resolvers
from dlux.data import layout
from dlux.data.compile import compile_data_description
from dlux.data.slidescore import export_slidescore_key

register_resolvers()


@hydra.main(version_base=None, config_path="../config", config_name="extract_tiles")
def main(cfg: DictConfig) -> None:
    print_config(cfg, fields=("study", "cohort", "tiling", "paths", "tile_cache_root", "limit", "precomputer"))

    study = Study(**OmegaConf.to_container(cfg.study, resolve=True))  # type: ignore[arg-type]
    cohort = Cohort(**OmegaConf.to_container(cfg.cohort, resolve=True))  # type: ignore[arg-type]
    tiling = OmegaConf.to_container(cfg.tiling, resolve=True)
    assert isinstance(tiling, dict)
    if cohort.name not in study.cohorts:
        sys.exit(
            f"[extract_tiles] cohort '{cohort.name}' is not part of study '{study.name}' "
            f"(has: {sorted(study.cohorts)})."
        )

    # Before any slide is opened. No-op for a local cohort.
    export_slidescore_key(cohort, cfg)

    database_uri = layout.db_uri(cfg.paths.studies_dir, study.name, cohort.name)
    data_description = compile_data_description(cohort, tiling, database_uri, split_version=None)
    # Tiles are cached RAW, in [0, 1] without normalization: the training-time reader applies
    # for_tile_classification, so the classifier sees the same input as when it reads the slide.
    dataset = MultiModalDataset(
        data_description=data_description,
        stage=SplitCategory.PREDICT,
        tile_transform=TileTransformFactory.for_mil_classification(),
        task=None,
    )
    slide_ids = None
    if cfg.limit is not None:
        slide_ids = list(dataset.slide_views())[: int(cfg.limit)]

    precompute_tile_images(
        dataset=dataset,
        cache_root=Path(cfg.tile_cache_root),
        grid_cache_root=Path(cfg.paths.cohorts_dir),
        slide_ids=slide_ids,
        batch_size=int(cfg.precomputer.batch_size),
        num_workers=int(cfg.precomputer.num_workers),
        only_missing=bool(cfg.precomputer.only_missing),
        jpeg_quality=int(cfg.precomputer.jpeg_quality),
    )


if __name__ == "__main__":
    main()
