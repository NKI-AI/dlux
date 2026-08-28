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
"""Tests for masks.csv, the sheet that keeps derived mask paths out of slides.csv."""

from __future__ import annotations

import pytest
from dlux.data.errors import BuildDbError
from dlux.data.sheets import MASK_PATH, RESERVED_SLIDE_COLUMNS, read_masks


def _write(path, text):
    path.write_text(text)
    return path


def test_absent_sheet_means_an_unmasked_cohort(tmp_path):
    # No file at all is a supported state, not an error: the cohort simply has no masks.
    assert read_masks(tmp_path / "masks.csv") == {}


def test_reads_the_mapping(tmp_path):
    sheet = _write(
        tmp_path / "masks.csv",
        "image_path,mask_path\na/one.svs,a/one.svs.mask.tiff\nb/two.svs,flat_two.tiff\n",
    )
    # Paths are recorded rather than derived, so both conventions coexist in one cohort.
    assert read_masks(sheet) == {"a/one.svs": "a/one.svs.mask.tiff", "b/two.svs": "flat_two.tiff"}


def test_blank_mask_is_not_a_mask(tmp_path):
    sheet = _write(tmp_path / "masks.csv", "image_path,mask_path\na.svs,\nb.svs,   \nc.svs,c.tiff\n")
    assert read_masks(sheet) == {"c.svs": "c.tiff"}


def test_missing_column_is_refused(tmp_path):
    sheet = _write(tmp_path / "masks.csv", "image_path\na.svs\n")
    with pytest.raises(BuildDbError, match="missing required column"):
        read_masks(sheet)


def test_duplicate_slide_is_refused(tmp_path):
    # Two masks for one slide has no defined answer, so it fails rather than picking one.
    sheet = _write(tmp_path / "masks.csv", "image_path,mask_path\na.svs,one.tiff\na.svs,two.tiff\n")
    with pytest.raises(BuildDbError, match="duplicate"):
        read_masks(sheet)


def test_mask_for_an_unknown_slide_is_refused(tmp_path):
    # A stale row naming a slide the cohort no longer has: silently ignoring it hides a broken sheet.
    sheet = _write(tmp_path / "masks.csv", "image_path,mask_path\ngone.svs,gone.tiff\n")
    with pytest.raises(BuildDbError, match="absent from slides.csv"):
        read_masks(sheet, known_slides={"kept.svs"})
    assert read_masks(sheet) == {"gone.svs": "gone.tiff"}  # unchecked without the slide set


def test_mask_path_is_no_longer_a_slides_column():
    # slides.csv describes the slide; the mask is derived data and lives in its own sheet.
    assert MASK_PATH not in RESERVED_SLIDE_COLUMNS
