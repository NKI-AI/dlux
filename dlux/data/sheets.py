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
"""Read + validate the rigid ``patients.csv`` / ``slides.csv`` input sheets.

See ``docs/specs/SHEET_SPEC.md``. Values are read as strings so raw label values
survive verbatim (e.g. grade ``"3"``, not ``3``); missing cells become NaN.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dlux.config.cohort import Cohort
from dlux.data.errors import BuildDbError

PATIENT_ID = "patient_id"
IMAGE_PATH = "image_path"
MASK_PATH = "mask_path"
READER = "reader"
STAINING = "staining"

MASKS_SHEET = "masks.csv"

# Columns in slides.csv that are structural (map to dedicated image fields), not per-image labels.
RESERVED_SLIDE_COLUMNS = {PATIENT_ID, IMAGE_PATH, READER, STAINING}


def read_patients(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def read_slides(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str)


def read_masks(path: Path, known_slides: set[str] | None = None) -> dict[str, str]:
    """``image_path -> mask_path`` from a cohort's ``masks.csv``, empty when the file is absent.

    Masks are derived per-slide data: either computed by ``generate_masks`` or shipped with the
    cohort. Keeping them out of ``slides.csv`` means acquiring masks never rewrites the sheet that
    describes the slides. ``mask_path`` is relative to the cohort's ``storage.mask_dir``, mirroring
    how ``image_path`` is relative to ``image_dir``.

    Paths are recorded rather than derived, so masks named by any convention are usable as-is.
    ``known_slides`` opts into rejecting rows that name a slide the cohort does not have.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    frame = pd.read_csv(path, dtype=str)
    for column in (IMAGE_PATH, MASK_PATH):
        if column not in frame.columns:
            raise BuildDbError(f"{path} missing required column '{column}' (needs {IMAGE_PATH},{MASK_PATH})")
    duplicates = frame[IMAGE_PATH][frame[IMAGE_PATH].duplicated()].tolist()
    if duplicates:
        raise BuildDbError(f"{path} has duplicate '{IMAGE_PATH}': {sorted(set(duplicates))[:10]}")
    masks = {
        str(row[IMAGE_PATH]): str(row[MASK_PATH])
        for _, row in frame.iterrows()
        if not pd.isna(row[MASK_PATH]) and str(row[MASK_PATH]).strip()
    }
    if known_slides is not None:
        orphans = set(masks) - known_slides
        if orphans:
            raise BuildDbError(f"{path} references slides absent from slides.csv: {sorted(orphans)[:10]}")
    return masks


def validate_sheets(patients: pd.DataFrame, slides: pd.DataFrame, cohort: Cohort) -> None:
    """Enforce the rigid contract. Hard failures raise; soft issues warn."""
    errors: list[str] = []

    if PATIENT_ID not in patients.columns:
        errors.append(f"patients.csv missing required column '{PATIENT_ID}'")
    # A field reads its source.column (defaults to the field key) + any stratify.column.
    needed_columns: set[str] = set()
    for field in cohort.contract.values():
        if field.source.column:
            needed_columns.add(field.source.column)
        if field.source.time_column:  # survival: the follow-up-time column
            needed_columns.add(field.source.time_column)
        if field.stratify.column:
            needed_columns.add(field.stratify.column)
    for column in sorted(needed_columns):
        if column not in patients.columns:
            errors.append(f"patients.csv missing contract source/stratify column '{column}'")
    for column in (PATIENT_ID, IMAGE_PATH):
        if column not in slides.columns:
            errors.append(f"slides.csv missing required column '{column}'")
    if errors:
        raise BuildDbError("Sheet validation failed:\n  - " + "\n  - ".join(errors))

    duplicate_patients = patients[PATIENT_ID][patients[PATIENT_ID].duplicated()].tolist()
    if duplicate_patients:
        raise BuildDbError(f"patients.csv has duplicate '{PATIENT_ID}': {sorted(set(duplicate_patients))[:10]}")

    duplicate_images = slides[IMAGE_PATH][slides[IMAGE_PATH].duplicated()].tolist()
    if duplicate_images:
        raise BuildDbError(f"slides.csv has duplicate '{IMAGE_PATH}': {sorted(set(duplicate_images))[:10]}")

    patient_ids = set(patients[PATIENT_ID].astype(str))
    orphans = set(slides[PATIENT_ID].astype(str)) - patient_ids
    if orphans:
        raise BuildDbError(f"slides.csv references unknown '{PATIENT_ID}': {sorted(orphans)[:10]}")

    # Soft: categorical raw values outside the contract map are excluded, not fatal.
    for field_name, field in cohort.contract.items():
        transform = field.source.transform
        if transform is not None and transform.categorical is not None:
            column = field.source.column
            present = set(patients[column].dropna().astype(str))
            unknown = present - set(transform.categorical)
            if unknown:
                print(
                    f"[validate] '{field_name}' (column '{column}'): {len(unknown)} value(s) not in map, "
                    f"excluded: {sorted(unknown)[:10]}"
                )
