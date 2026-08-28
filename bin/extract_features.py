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
"""Pre-compute per-slide tile features over the whole manifest (split-agnostic).

    extract_features study=tcga_subtyping cohort=tcga_brca_coad_christiana feature_extractor=phikon_tile   # real run (GPU)
    extract_features study=tcga_subtyping cohort=tcga_brca_coad_christiana dry_run=true                    # enumerate slides only (CPU)

Features are cached under ``cohorts/<cohort>/cache/``, keyed by model/grid/slide. The key is
independent of split or study, so a cohort's features are extracted once and reused by every study it
appears in. The ``study`` here only locates the cohort's DB (``studies/<study>/db/<cohort>.db``).
Extraction runs over every slide in that manifest (``split_version=None``). Multi-GPU precompute is a
single launch (``precomputer.devices``).
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import hydra
import torch
from dlux.config.cohort import Cohort, SplitCategory, Study
from dlux.data import layout
from dlux.data.analyze_features import analyze_features
from dlux.data.compile import compile_data_description
from dlux.modalities import TileFeatures
from omegaconf import DictConfig, OmegaConf

from ahcore.data.adapters.cached_tile_features import precompute_tile_features
from ahcore.data.mm_dataset import MultiModalDataset
from ahcore.transforms.tile import TileTransformFactory
from ahcore.utils.io import print_config


def _build_dataset(cohort_dict: dict, tiling_dict: dict, database_uri: str) -> MultiModalDataset:
    """Zero-configuration factory (picklable across the precompute spawn)."""
    cohort = Cohort(**cohort_dict)
    data_description = compile_data_description(cohort, tiling_dict, database_uri, split_version=None)
    return MultiModalDataset(
        data_description=data_description,
        stage=SplitCategory.PREDICT,
        tile_transform=TileTransformFactory.for_mil_classification(),
        task=None,
    )


def _build_feature_model(feature_extractor_dict: dict) -> torch.nn.Module:
    return hydra.utils.instantiate(feature_extractor_dict)


@hydra.main(version_base=None, config_path="../config", config_name="extract_features")
def main(cfg: DictConfig) -> None:
    print_config(
        cfg, fields=("study", "cohort", "tiling", "feature_extractor", "paths", "precomputer", "dry_run", "analyze")
    )

    study = Study(**OmegaConf.to_container(cfg.study, resolve=True))  # type: ignore[arg-type]
    cohort_dict = OmegaConf.to_container(cfg.cohort, resolve=True)
    tiling_dict = OmegaConf.to_container(cfg.tiling, resolve=True)
    cohort_name = str(cfg.cohort.name)
    if cohort_name not in study.cohorts:
        sys.exit(
            f"[extract_features] cohort '{cohort_name}' is not part of study '{study.name}' "
            f"(has: {sorted(study.cohorts)})."
        )
    database_uri = layout.db_uri(cfg.paths.studies_dir, study.name, cohort_name)

    if cfg.dry_run:
        dataset = _build_dataset(cohort_dict, tiling_dict, database_uri)  # type: ignore[arg-type]
        print(f"[extract_features] dry run: {len(dataset)} slides enumerated for '{cohort_name}' (split-agnostic)")
        return

    feature_extractor_dict = OmegaConf.to_container(cfg.feature_extractor, resolve=True)
    # One CPU instantiation reads the model identity; workers build their own on-GPU copies. Writer and
    # reader derive the cache namespace from the same function, so they cannot mismatch.
    model_name, model_hash, feature_dim = TileFeatures.identify(cfg.feature_extractor)

    devices = cfg.precomputer.devices
    world_size = torch.cuda.device_count() if devices == "auto" else int(devices)
    # A CUDA-less host (Mac, CI) reports 0 devices; clamp to a single worker so the CPU/MPS path in
    # precompute_tile_features runs, instead of spawning zero workers and silently doing nothing.
    world_size = max(world_size, 1)

    cache_path = precompute_tile_features(
        dataset_factory=partial(_build_dataset, cohort_dict, tiling_dict, database_uri),
        feature_model_factory=partial(_build_feature_model, feature_extractor_dict),
        cache_root=Path(cfg.paths.cohorts_dir),
        model_name=model_name,
        model_hash=model_hash,
        feature_dim=feature_dim,
        batch_size=cfg.precomputer.batch_size,
        num_workers=cfg.precomputer.num_workers,
        only_missing=cfg.precomputer.only_missing,
        verify_cached=cfg.precomputer.verify_cached,
        tile_output_dir=None,
        tile_output_size=None,
        world_size=world_size,
    )

    if cfg.analyze:
        summary = analyze_features(cache_path)
        if summary is None:
            print(
                "[extract_features] nothing cached — no slide produced tiles. Likely the cohort's masks "
                "(mask_dir) are empty/missing or the tiling mask_threshold rejected everything; check the "
                f"extraction report for per-slide status. ({cache_path})"
            )
        else:
            print(f"[extract_features] wrote {summary} + tiles.png")


if __name__ == "__main__":
    main()
