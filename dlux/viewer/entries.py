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
"""Resolve a cohort's ``sheets/slides.csv`` into the slide list the viewer serves.

Reads the sheets, not the manifest DB, so the viewer works on any cohort without a study and before
``build_db`` has run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dlux.config.cohort import Cohort
from dlux.data import sheets
from dlux.data.sheets import IMAGE_PATH, read_slides


@dataclass(frozen=True)
class SlideEntry:
    """One servable slide. ``slide_id`` is the sheet's ``image_path``, relative, stable, readable,
    and is what the HTTP endpoints take as ``?slide=``."""

    slide_id: str
    image_path: Path
    mask_path: Path | None
    reader: str
    patient_id: str

    @property
    def name(self) -> str:
        return Path(self.slide_id).name


def resolve_entries(cohort: Cohort, cohorts_dir: Path, *, only: list[str] | None = None) -> list[SlideEntry]:
    """Build the slide list for ``cohort``.

    Args:
        cohort: the cohort whose sheets and storage roots are read.
        cohorts_dir: root holding ``<cohort>/sheets/slides.csv``.
        only: optional subset of slide ids (or bare filenames) to keep, for inspecting a handful of
            slides, e.g. those named in a corruption report.

    Returns:
        One entry per unique ``image_path``, in sheet order.

    Raises:
        FileNotFoundError: the cohort has no slides sheet.
        ValueError: ``only`` names a slide the cohort does not have.
    """
    slides_csv = cohorts_dir / cohort.name / "sheets" / "slides.csv"
    if not slides_csv.exists():
        raise FileNotFoundError(f"slides sheet not found: {slides_csv} (build the cohort sheets first)")

    frame = read_slides(slides_csv)
    image_dir = Path(cohort.storage.image_dir)
    mask_dir = Path(cohort.storage.mask_dir) if cohort.storage.mask_dir else None
    masks_by_image = sheets.read_masks(slides_csv.with_name(sheets.MASKS_SHEET))

    entries: dict[str, SlideEntry] = {}
    for row in frame.itertuples(index=False):
        slide_id = str(getattr(row, IMAGE_PATH))
        if slide_id in entries:  # a patient may appear on several rows; the slide list is per-slide
            continue
        mask_rel = masks_by_image.get(slide_id, "")
        entries[slide_id] = SlideEntry(
            slide_id=slide_id,
            image_path=image_dir / slide_id,
            mask_path=(mask_dir / mask_rel) if (mask_dir and mask_rel) else None,
            reader=cohort.storage.default_reader,
            patient_id=str(getattr(row, "patient_id", "")),
        )

    if only is None:
        return list(entries.values())

    # Accept either a full slide id or a bare filename, so a list lifted from a report is usable as-is.
    by_name: dict[str, str] = {}
    for slide_id in entries:
        by_name.setdefault(Path(slide_id).name, slide_id)
    selected: list[SlideEntry] = []
    missing: list[str] = []
    for wanted in only:
        key = wanted if wanted in entries else by_name.get(Path(wanted).name)
        if key is None:
            missing.append(wanted)
        else:
            selected.append(entries[key])
    if missing:
        raise ValueError(f"cohort '{cohort.name}' has no slide matching: {missing}")
    return selected
