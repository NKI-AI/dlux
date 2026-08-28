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
"""Materialise a ``Cohort`` (under a study's split plan) into an ahcore sqlite manifest DB.

Ingest the rigid sheets -> (optionally) probe each slide for geometry via dlup ->
write patient/image/mask/label rows -> generate split versions from the split plan.
This is the ``build-db`` step. The user's per-cohort preprocessing (in ``scripts/``) produces the sheets.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from ahcore.manifest import (
    CategoryEnum,
    Image,
    ImageLabels,
    Manifest,
    Mask,
    Patient,
    PatientLabels,
    Split,
    SplitDefinitions,
    open_db,
    read_imageio_annotation,
)
from dlup import ImageConfig, MaskConfig, TilingConfig
from dlup.data.dataset import SlideDataset
from dlup.utils.backends import ImageBackend

from dlux.config.cohort import Cohort, Objective, SplitCategory, Splits, StratifyMethod
from dlux.data import sheets
from dlux.data.errors import BuildDbError
from dlux.data.splits import (
    continuous_cuts,
    export_splits_csv,
    generate_splits,
    import_splits_csv,
    valid_patients_by_field,
    validate_imported_splits,
)
from dlux.modalities import MODALITIES

_CATEGORY = {
    SplitCategory.FIT: CategoryEnum.FIT,
    SplitCategory.VALIDATE: CategoryEnum.VALIDATE,
    SplitCategory.TEST: CategoryEnum.TEST,
    SplitCategory.PREDICT: CategoryEnum.PREDICT,
}


class ProbeMode(str, Enum):
    """How thoroughly build-db validates each slide.

    - ``none``    , don't open slides; leave height/width/mpp NULL (test the sheet/
                     split logic without WSI files present).
    - ``metadata``, open the slide header only (fast; a few ms/slide); records real
                     geometry. Does NOT catch tiling/mask failures.
    - ``full``    , construct the tiling ``SlideDataset`` (parse the mask, build the
                     grid) and require a non-empty grid. Slower but reliable: catches
                     broken masks, foreground-computation failures, and empty grids.
    """

    none = "none"
    metadata = "metadata"
    full = "full"


# Coarse grid to open the slide header for `metadata` mode; not the training grid.
_METADATA_TILING = TilingConfig(mpp=16.0, tile_size=(512, 512), tile_overlap=(0, 0))

# Representative grid for `full` mode (overridable via build_db.yaml `probe_grid`).
# mask_threshold 0.0 validates that tiling *runs* rather than curating tissue.
DEFAULT_PROBE_GRID: dict = {"mpp": 2.0, "tile_size": [224, 224], "tile_overlap": [0, 0], "mask_threshold": 0.0}


def _geometry(slide_image) -> dict[str, float]:
    width, height = slide_image.size
    return {"height": int(height), "width": int(width), "mpp": float(slide_image.mpp)}


def _probe_slide(
    slide_path: Path, reader: str, mode: ProbeMode, mask_path: Optional[Path], grid: dict
) -> Optional[dict[str, float]]:
    """Open/validate a slide and return its geometry. None on failure (logged)."""
    try:
        image_config = ImageConfig(backend=ImageBackend[reader], apply_color_profile=False)
        if mode == ProbeMode.metadata:
            dataset = SlideDataset.from_standard_tiling(
                path=slide_path, tiling_config=_METADATA_TILING, image_config=image_config
            )
            return _geometry(dataset.slide_image)

        # full: parse the mask (if any), build the real grid, require it to be non-empty.
        mask_config = None
        if mask_path is not None:
            # Read through the same helper training uses, so the probe validates the array that will
            # actually be loaded.
            mask_config = MaskConfig(
                mask=read_imageio_annotation(mask_path), mask_threshold=grid["mask_threshold"], crop=False
            )
        dataset = SlideDataset.from_standard_tiling(
            path=slide_path,
            tiling_config=TilingConfig(
                mpp=grid["mpp"], tile_size=tuple(grid["tile_size"]), tile_overlap=tuple(grid["tile_overlap"])
            ),
            mask_config=mask_config,
            image_config=image_config,
        )
        if len(dataset) == 0:
            print(f"[build_db] full probe: empty grid for {slide_path}")
            return None
        return _geometry(dataset.slide_image)
    except Exception as exc:  # noqa: BLE001, per-slide failures are skipped, not fatal
        print(f"[build_db] probe failed for {slide_path}: {exc}")
        return None


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value)) or value == ""


@dataclass
class _BuildCtx:
    """Ingested inputs the endpoint builders read from.

    Carries only modality-agnostic context; a builder needing extra inputs (e.g. the expression
    rider's RNA matrix) resolves them itself from the cohort + ``cohorts_dir``."""

    patient_labels: dict[str, dict[str, str]]  # patient_code -> {column: raw value}
    manifest_codes: set[str]  # every patient_code in the manifest
    cohort_name: str
    cohorts_dir: Optional[Path]  # cohort data root; None -> derived-artifact side-inputs unavailable


# -- endpoint builders (dispatched on objective via _BUILDERS) ---------------
# A builder owns one endpoint kind's build-time contributions: fold-invariant provenance stats
# (written to the sidecar json) and patient coverage (fed to split generation). None from either
# means "this endpoint contributes nothing to that artifact".
class _SheetBuilder:
    """Central sheet-tabular endpoints: resolve fold-invariant target transform statistics.

    A ``binarize`` (or ``discretize``) cut must be identical in every fold, or the ground truth itself
    moves between folds. It is resolved cohort-wide here and recorded as provenance. The raw value stays
    in the DB; train applies the cut per-patient. Categorical and log1p need no statistic."""

    def provenance(self, name: str, field, ctx: "_BuildCtx") -> Optional[dict]:
        transform = field.source.transform
        if transform is None or (transform.binarize is None and transform.discretize is None):
            return None  # only binarize/discretize carry a fold-invariant cut; log1p is elementwise
        column = field.source.column
        values = []
        for row in ctx.patient_labels.values():
            raw = row.get(column)
            try:
                parsed = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isnan(parsed):
                values.append(parsed)
        if not values:
            return None
        arr = np.asarray(values, dtype=float)
        if transform.binarize is not None:
            b = transform.binarize
            threshold = b.value if b.method == StratifyMethod.threshold else float(np.median(arr))
            return {"transform": "binarize", "column": column, "threshold": threshold, "n": len(values)}
        d = transform.discretize
        edges = continuous_cuts(arr, method=d.method, k=d.k, edges=d.edges)
        return {"transform": "discretize", "column": column, "edges": edges, "n": len(values)}

    def coverage(self, name: str, field, ctx: "_BuildCtx") -> Optional[set[str]]:
        return None  # every manifest patient is a candidate; splits use the whole cohort


class _BulkRNABuilder:
    """RNA expression rider: target is the external matrix, so coverage is which patients it holds.

    Verifies the matrix exists and reports the covered patients, the usable-N for the endpoint.
    The adapter reads the matrix directly at train time. The DB only records coverage (via the splits)."""

    def provenance(self, name: str, field, ctx: "_BuildCtx") -> Optional[dict]:
        return None  # no fold-invariant scalar stat; z-scoring is per-fold per-gene at train time

    def coverage(self, name: str, field, ctx: "_BuildCtx") -> Optional[set[str]]:
        covered = _modality_coverage("bulk_rna", ctx)
        print(f"[build_db] expression '{name}': {len(covered)}/{len(ctx.manifest_codes)} patients have RNA.")
        return covered


_SHEET = _SheetBuilder()
_BULK_RNA = _BulkRNABuilder()
_BUILDERS = {"sheet": _SHEET, "bulk_rna": _BULK_RNA}

# Modality -> patient-coverage resolver, derived from the modality registry: a modality can gate splits
# (Study.require_modalities) exactly when it declares a `coverage` capability. One vocabulary, adding a
# modality with coverage makes it gateable, with nothing to register here.
MODALITY_COVERAGE = {name: modality.coverage for name, modality in MODALITIES.items() if hasattr(modality, "coverage")}


def _modality_coverage(name: str, ctx: "_BuildCtx") -> set[str]:
    """Patients covered by ``name``, adapting the build context to the modality's explicit arguments."""
    return MODALITY_COVERAGE[name](
        cohorts_dir=ctx.cohorts_dir, cohort_name=ctx.cohort_name, manifest_codes=ctx.manifest_codes
    )


def _resolve_require_coverage(require_modalities: list[str], ctx: "_BuildCtx") -> Optional[set[str]]:
    """Intersection of the required modalities' patient coverage (None when none required), the fusion
    fair-comparison gate handed to generate_splits so every field's splits share one patient set."""
    if not require_modalities:
        return None
    covered: Optional[set[str]] = None
    for modality in require_modalities:
        if modality not in MODALITY_COVERAGE:
            raise BuildDbError(
                f"require_modalities '{modality}' cannot gate splits (gateable modalities: {sorted(MODALITY_COVERAGE)})"
            )
        patients = _modality_coverage(modality, ctx)
        covered = patients if covered is None else (covered & patients)
    print(f"[build_db] require_modalities {require_modalities}: {len(covered or set())} patients gated into splits.")
    return covered


def _builder(field):
    return _BUILDERS[field.family]


def _resolve_provenance(cohort: Cohort, ctx: "_BuildCtx") -> dict:
    """Fold-invariant target stats per endpoint (empty for purely-categorical cohorts)."""
    stats: dict[str, dict] = {}
    for name, field in cohort.contract.items():
        entry = _builder(field).provenance(name, field, ctx)
        if entry is not None:
            stats[name] = entry
    return stats


def _resolve_coverage(cohort: Cohort, ctx: "_BuildCtx") -> dict[str, set[str]]:
    """Per-endpoint patient coverage; only endpoints that restrict their population contribute."""
    covered: dict[str, set[str]] = {}
    for name, field in cohort.contract.items():
        patients = _builder(field).coverage(name, field, ctx)
        if patients is not None:
            covered[name] = patients
    return covered


def _apply_admin_censoring(patients: pd.DataFrame, cohort: Cohort, admin_censor: dict[str, float]) -> pd.DataFrame:
    """Cap each survival target's follow-up at its horizon on the sheet, before any label or split is
    derived. A patient beyond the horizon becomes ``event=0`` at ``time=horizon``. Because the capped
    frame feeds both the stored labels and ``generate_splits``, the DB and its stratified splits agree
    on the capped endpoint, and every downstream reader (time bins, task labels, scoring) inherits it."""
    for target, horizon in admin_censor.items():
        field = cohort.contract.get(target)
        if field is None or field.objective != Objective.survival:
            raise BuildDbError(f"admin_censor target '{target}' is not a survival endpoint of cohort '{cohort.name}'.")
        event_col, time_col = field.source.column, field.source.time_column
        time = pd.to_numeric(patients[time_col], errors="coerce")
        beyond = time > horizon  # NaN (missing time) compares False, so missing stays missing
        events_lost = int(((pd.to_numeric(patients[event_col], errors="coerce") == 1) & beyond).sum())
        patients.loc[beyond, event_col] = 0
        patients.loc[beyond, time_col] = horizon
        print(
            f"[build_db] admin_censor {target}={horizon} ({time_col}/{event_col}): "
            f"{int(beyond.sum())} patient(s) capped, {events_lost} event(s) -> censored."
        )
    return patients


def build_db(
    cohort: Cohort,
    splits: Splits,
    database_uri: str,
    patients_csv: Path,
    slides_csv: Path,
    probe: str = "metadata",
    probe_grid: Optional[dict] = None,
    overwrite: bool = False,
    analyze: bool = True,
    cohorts_dir: Optional[Path] = None,
    admin_censor: Optional[dict[str, float]] = None,
    import_splits: Optional[Path] = None,
    import_allow_uncovered: bool = False,
) -> None:
    mode = ProbeMode(probe)
    grid = probe_grid or DEFAULT_PROBE_GRID

    # Pre-flight, before any slow work (sheet parse, probing): fail fast + clean.
    for label, csv_path in (("patients_csv", patients_csv), ("slides_csv", slides_csv)):
        if not csv_path.exists():
            raise BuildDbError(f"{label} not found: {csv_path}")

    db_path = Path(database_uri.replace("sqlite:///", ""))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        if not overwrite:
            raise BuildDbError(
                f"database already exists: {db_path}. Pass overwrite=true to replace it, or point at a new path."
            )
        print(f"[build_db] overwrite=true → removing existing {db_path}")
        db_path.unlink()

    patients = sheets.read_patients(patients_csv)
    slides = sheets.read_slides(slides_csv)
    sheets.validate_sheets(patients, slides, cohort)
    if admin_censor:
        patients = _apply_admin_censoring(patients, cohort, admin_censor)
    # Masks live in their own sheet, keyed by image_path, so generating them never rewrites slides.csv.
    masks_by_image = sheets.read_masks(
        Path(slides_csv).with_name(sheets.MASKS_SHEET), known_slides=set(slides[sheets.IMAGE_PATH].astype(str))
    )

    image_dir = Path(cohort.storage.image_dir)
    mask_dir = Path(cohort.storage.mask_dir) if cohort.storage.mask_dir else None

    label_columns = [c for c in patients.columns if c != sheets.PATIENT_ID]
    has_reader_column = sheets.READER in slides.columns
    has_staining_column = sheets.STAINING in slides.columns

    with open_db(database_uri, ensure_exists=False) as session:
        manifest = Manifest(name=cohort.name)
        session.add(manifest)
        session.flush()

        patient_cache: dict[str, Patient] = {}
        patient_labels: dict[str, dict[str, str]] = {}
        for _, row in tqdm(patients.iterrows(), total=len(patients), desc="patients"):
            code = str(row[sheets.PATIENT_ID])
            patient = Patient(patient_code=code, manifest=manifest)
            session.add(patient)
            session.flush()
            patient_cache[code] = patient
            values: dict[str, str] = {}
            for column in label_columns:
                value = row[column]
                if _is_missing(value):
                    continue
                session.add(PatientLabels(key=column, value=str(value), patient=patient))
                values[column] = str(value)
            patient_labels[code] = values

        ctx = _BuildCtx(
            patient_labels=patient_labels,
            manifest_codes=set(patient_labels),
            cohort_name=cohort.name,
            cohorts_dir=Path(cohorts_dir) if cohorts_dir is not None else None,
        )
        # Resolve target-defining population stats once (binarize threshold / target zscore μ,σ).
        contract_stats = _resolve_provenance(cohort, ctx)

        n_slides = 0
        slide_desc = f"slides ({mode.value} probe)" if mode != ProbeMode.none else "slides"
        for _, row in tqdm(slides.iterrows(), total=len(slides), desc=slide_desc):
            code = str(row[sheets.PATIENT_ID])
            patient = patient_cache[code]
            image_rel = str(row[sheets.IMAGE_PATH])

            reader = cohort.storage.default_reader
            if has_reader_column and not _is_missing(row.get(sheets.READER)):
                reader = str(row[sheets.READER]).upper()

            staining = cohort.storage.default_staining
            if has_staining_column and not _is_missing(row.get(sheets.STAINING)):
                staining = str(row[sheets.STAINING])

            mask_rel = masks_by_image.get(image_rel)
            has_mask = mask_dir is not None and not _is_missing(mask_rel)
            mask_full = (mask_dir / str(mask_rel)) if has_mask else None

            geometry: dict[str, Optional[float]] = {"height": None, "width": None, "mpp": None}
            if mode != ProbeMode.none:
                probed = _probe_slide(image_dir / image_rel, reader, mode, mask_full, grid)
                if probed is None:
                    print(f"[build_db] skipping slide (probe failed): {image_rel}")
                    continue
                geometry = probed  # type: ignore[assignment]

            image = Image(filename=image_rel, reader=reader, staining=staining, patient=patient, **geometry)
            session.add(image)
            session.flush()

            for column in slides.columns:
                if column in sheets.RESERVED_SLIDE_COLUMNS:
                    continue
                value = row[column]
                if _is_missing(value):
                    continue
                session.add(ImageLabels(key=column, value=str(value), image=image))

            if has_mask:
                session.add(Mask(filename=str(mask_rel), reader="IMAGEIO", image=image))
            n_slides += 1

        session.flush()

        rnaseq_covered = _resolve_coverage(cohort, ctx)
        require_coverage = _resolve_require_coverage(splits.require_modalities, ctx)
        if import_splits is not None:
            # Predefined split: write the supplied assignment verbatim instead of drawing one. The split
            # math (generate_splits) is untouched and stays the default; this only swaps the source.
            versions = import_splits_csv(import_splits)
            valid_by_field = valid_patients_by_field(cohort, patient_labels, rnaseq_covered, require_coverage)
            versions = validate_imported_splits(
                versions,
                splits,
                cohort_patients=set(patient_cache),
                valid_by_field=valid_by_field,
                allow_uncovered=import_allow_uncovered,
            )
            print(f"[build_db] imported {len(versions)} split version(s) from {import_splits}")
        else:
            versions = generate_splits(
                cohort, patient_labels, splits, rnaseq_covered=rnaseq_covered, require_coverage=require_coverage
            )
        for split_version in versions:
            split_def = SplitDefinitions(version=split_version.version, description=split_version.description)
            session.add(split_def)
            session.flush()
            for code, category in split_version.assignments:
                session.add(
                    Split(category=_CATEGORY[category], patient=patient_cache[code], split_definition=split_def)
                )
            session.flush()

        session.commit()

    # Contract-resolution provenance (fold-invariant target stats). Empty for purely-categorical
    # cohorts; train reads it to apply binarize/numeric target transforms consistently.
    stats_path = db_path.parent / f"{cohort.name}_contract_stats.json"
    stats_path.write_text(json.dumps(contract_stats, indent=2))

    # Always emit the resulting split as a portable CSV, the canonical export, and the exact format a
    # sibling cohort imports. When imported, record where it came from so a reader never mistakes an
    # imported split for a freshly drawn one.
    export_splits_csv(versions, db_path.parent / f"{cohort.name}_splits.csv")
    if import_splits is not None:
        source_sha = hashlib.sha256(Path(import_splits).read_bytes()).hexdigest()[:16]
        (db_path.parent / f"{cohort.name}_splits_source.json").write_text(
            json.dumps({"imported_from": str(import_splits), "sha256_16": source_sha}, indent=2)
        )

    print(
        f"[build_db] wrote {db_path}: {len(patient_cache)} patients, {n_slides} slides, "
        f"{len(versions)} split versions, {len(contract_stats)} resolved target stat(s)"
    )

    if analyze:
        from dlux.data.analyze import analyze_db  # local import: pulls matplotlib/seaborn only when needed

        analyze_db(cohort, database_uri, db_path.parent / f"{cohort.name}_analysis", cohorts_dir=cohorts_dir)
