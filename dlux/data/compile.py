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
"""Compile an ahcore ``DataDescription`` from a dlux ``Dataset`` + ``tiling``.

Users author ``Cohort`` (locked cohort truth) and ``tiling`` (read geometry)
separately. This assembles the ahcore ``DataDescription`` that DataManager /
datamodule consume. It is never authored by hand.

``split_version=None`` => split-agnostic (iterate the whole manifest), used by
feature extraction, since the cache is keyed by manifest/model/grid, not split.
A concrete version selects a fold at train / eval time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

from ahcore.manifest import DataDescription, GridDescription

from dlux.config.cohort import Cohort


def compile_data_description(
    cohort: Cohort,
    tiling: Mapping,
    database_uri: str,
    split_version: Optional[str] = None,
) -> DataDescription:
    """Build the ahcore ``DataDescription`` for ``cohort`` at the given ``tiling``.

    ``database_uri`` is the study-derived DB location (``studies/<study>/db/<cohort>.db``),
    not carried by the cohort, since the same cohort is materialised differently per study.
    Geometry is deterministic here, fit-time tile sampling lives in the augmentations config,
    not the grid. Head size is resolved by the task, not carried on the data description.
    """
    grid = GridDescription(
        mpp=tiling["mpp"],
        tile_size=tuple(tiling["tile_size"]),
        tile_overlap=tuple(tiling["tile_overlap"]),
        random_sample_in_grid=False,
    )
    # annotations_dir is a required Path on DataDescription; fall back to image_dir when maskless.
    annotations_dir = cohort.storage.mask_dir or cohort.storage.image_dir
    return DataDescription(
        manifest_database_uri=database_uri,
        manifest_name=cohort.name,
        split_version=split_version,
        data_dir=Path(cohort.storage.image_dir),
        annotations_dir=Path(annotations_dir),
        mask_label=None,
        mask_threshold=tiling.get("mask_threshold"),
        tiling_mode=tiling.get("tiling_mode"),
        training_grid=grid,
        inference_grid=grid,
        use_roi=False,
    )
