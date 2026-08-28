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
"""Shared helpers for the per-cohort sheet builders (``scripts/cohort_builders/<cohort>/sheets.py``).

Each builder turns a cohort's metadata + slide layout into the dlux input sheets
(``<cohort>_patients.csv`` + ``<cohort>_slides.csv``), which ``dlux build_db`` then ingests
(probing geometry and generating splits). The builders never touch the DB, geometry, or splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dlux.data.sheets import IMAGE_PATH, MASK_PATH, MASKS_SHEET, PATIENT_ID


def clean(value: object) -> str:
    """Raw label value → string for the sheet; NaN/None → empty (dlux reads empty as missing)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none") else s


def index_masks(masks_dir: Optional[Path], mask_ext: str = ".tiff") -> Dict[str, str]:
    """Map mask *basename* → path (relative to ``masks_dir``), for NON-EMPTY files with extension
    ``mask_ext`` ONLY.

    Restricting to the mask extension is the point: a bare stem match against *all* files
    wrongly grabs unrelated artifacts (pickles, thumbnails, sidecar files) that merely share
    a slide's stem. Zero-byte masks (a failed mask-generation run) are skipped so they read as
    missing rather than crashing extraction with an unreadable-image error."""
    if masks_dir is None:
        return {}
    return {
        p.name: p.relative_to(masks_dir).as_posix()
        for p in masks_dir.rglob(f"*{mask_ext}")
        if p.is_file() and p.stat().st_size > 0
    }


def mask_for(slide_path: Path, mask_index: Dict[str, str], mask_ext: str = ".tiff") -> Optional[str]:
    """Mask matching ``slide_path`` by the exact convention ``<slide stem><mask_ext>`` —
    e.g. ``DCIS_X.svs`` → ``DCIS_X.tiff``, ``Y PRIM.ndpi`` → ``Y PRIM.tiff`` — else ``None``."""
    return mask_index.get(slide_path.stem + mask_ext)


def slide_row(patient_id: str, image_path: str, mask: Optional[str]) -> Dict[str, str]:
    """One gathered slide; carries ``mask_path`` when a mask was matched.

    ``write_sheets`` splits that key out into ``masks.csv`` — a mask is data about a slide we or the
    provider produced, not part of the slide's own description."""
    row = {PATIENT_ID: patient_id, IMAGE_PATH: image_path}
    if mask is not None:
        row[MASK_PATH] = mask
    return row


def warn_missing_masks(cohort: str, masks_dir: Optional[Path], slide_rows: List[Dict], mask_ext: str) -> None:
    """If masks were requested, warn loudly about slides that got no matching mask"""
    if masks_dir is None:
        return
    missing = [r for r in slide_rows if MASK_PATH not in r]
    if missing:
        print(
            f"[{cohort}] WARNING: {len(missing)}/{len(slide_rows)} slides have no matching "
            f"'{mask_ext}' mask under {masks_dir} (convention: <slide-stem>{mask_ext}). "
        )


def write_sheets(out_dir: Path, prefix: str, patient_rows: List[Dict], slide_rows: List[Dict]) -> None:
    """Write ``patients.csv`` + ``slides.csv`` (+ ``masks.csv`` when masks were matched) under
    ``out_dir`` (typically cohorts/<cohort>/sheets/).

    Masks go to their own sheet keyed by ``image_path``, so acquiring masks later never rewrites the
    slide description. A cohort whose masks ship with the data gets that sheet from here; one whose
    masks we compute gets it from ``generate_masks``.

    ``prefix`` names the cohort in log lines only — the filenames are unprefixed because the cohort is
    already the directory (layout A)."""
    if not patient_rows:
        raise SystemExit(f"[{prefix}] no patients gathered — check --metadata-file / --images paths and columns.")
    out_dir.mkdir(parents=True, exist_ok=True)
    patients_csv = out_dir / "patients.csv"
    slides_csv = out_dir / "slides.csv"
    mask_rows = [
        {IMAGE_PATH: r[IMAGE_PATH], MASK_PATH: r[MASK_PATH]} for r in slide_rows if r.get(MASK_PATH) is not None
    ]
    pd.DataFrame(patient_rows).to_csv(patients_csv, index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k != MASK_PATH} for r in slide_rows]).to_csv(slides_csv, index=False)
    written = f"{patients_csv.name}, {slides_csv.name}"
    if mask_rows:
        masks_csv = out_dir / MASKS_SHEET
        pd.DataFrame(mask_rows).to_csv(masks_csv, index=False)
        written += f", {masks_csv.name}"
    print(
        f"[{prefix}] {len(patient_rows)} patients, {len(slide_rows)} slides "
        f"({len(mask_rows)} with masks) → {written} in {out_dir}"
    )
