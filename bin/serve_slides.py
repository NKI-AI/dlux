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
"""Hydra entry point: browse a cohort's slides in the browser.

    serve_slides cohort=tcga_brca_coad_christiana
    serve_slides cohort=tcga_brca_coad_mskcc port=8080
    serve_slides cohort=tcga_brca_coad_christiana 'only=[<slide-filename>.svs]'

Reads the cohort's ``sheets/slides.csv`` (no manifest DB needed, so it works before build_db) and
serves each slide as an XYZ tile pyramid with an OpenLayers viewer. Runs on a CPU node. Reach it with
an SSH tunnel to ``port``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import uvicorn
from dlux.config.cohort import Cohort
from dlux.data import layout
from dlux.eval.run_records import RunRecords, class_names, load_run_records
from dlux.viewer import create_app, resolve_entries
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from ahcore.manifest import DataManager
from ahcore.utils.io import get_logger, print_config

logger = get_logger(__name__)


def _in_study_slides(cfg: DictConfig, cohort: Cohort, study: str, records: RunRecords) -> set[str] | None:
    """Slide ids the study's splits admit for this endpoint, or None if the DB cannot answer.

    Read from the splits table through the same manager training uses, so the viewer and training agree
    on who is in the study. Nested CV partitions the study population across outer folds, so one split
    version's fit + validate + test is already the whole admitted set. Hence a single query, on a version
    the loaded run proves exists.
    """
    database = layout.db_path(cfg.paths.studies_dir, study, cohort.name)
    if not database.exists():
        logger.warning("no manifest DB at %s; cannot tell missing predictions from out-of-study.", database)
        return None
    outer = records.folds[0]
    split_version = f"{outer}_i{records.inners_by_outer[outer][0]}"
    manager = DataManager(f"sqlite:///{database}")
    try:
        patients = manager.get_records_by_split(manifest_name=cohort.name, split_version=split_version)
        return {str(image.filename) for patient in patients for image in patient.images}
    except ValueError as exc:  # no such manifest or split version in this DB
        logger.warning("splits unavailable for %s (%s): %s", split_version, database, exc)
        return None


def _resolve_run(cfg: DictConfig, cohort: Cohort) -> tuple[RunRecords | None, str, dict[int, str]]:
    """Loads the model named by the ``run`` config group, if it is set.

    Returns:
        The records and a label identifying the model, or ``(None, "")`` when the group is unset.

    Raises:
        SystemExit: the group is set only partially, or names a sweep with no attention artifacts.
    """
    group = cfg.get("run") or {}
    named = {key: group.get(key) for key in ("study", "experiment", "field")}
    if not any(named.values()):
        return None, "", {}
    missing = [key for key, value in named.items() if not value]
    if missing:
        sys.exit(f"\n[serve-slides] run needs study, experiment and field together; missing: {missing}")

    raw_inner = group.get("inner")
    inner = None if raw_inner is None else int(raw_inner)
    runs_dir = Path(cfg.paths.studies_dir) / str(named["study"]) / "runs" / str(named["experiment"]) / cohort.name
    if not runs_dir.exists():
        sys.exit(f"\n[serve-slides] no such sweep: {runs_dir}")
    records = load_run_records(runs_dir, str(named["field"]), inner)
    if records is None:
        sys.exit(
            f"\n[serve-slides] no attention.npz under {runs_dir} for {named['field']}_cv_o*_i{'*' if inner is None else inner} — "
            "retrain that sweep, or pick another inner replicate."
        )
    field_name = str(named["field"])
    contract_field = cohort.contract.get(field_name)
    mapping = getattr(getattr(contract_field, "source", None), "transform", None)
    names = class_names(field_name, getattr(mapping, "categorical", None))
    which = "ensemble" if inner is None else f"i{inner}"
    return records, f"{named['experiment']}/{field_name} {which}", names


@hydra.main(version_base=None, config_path="../config", config_name="serve_slides")
def main(cfg: DictConfig) -> None:
    print_config(cfg, fields=("cohort", "paths", "host", "port", "tile_size", "only", "run"))

    try:
        cohort = Cohort(**OmegaConf.to_container(cfg.cohort, resolve=True))  # type: ignore[arg-type]
        only = list(OmegaConf.to_container(cfg.only, resolve=True)) if cfg.get("only") else None  # type: ignore[arg-type]
        entries = resolve_entries(cohort, Path(cfg.paths.cohorts_dir), only=only)
    except (FileNotFoundError, ValidationError, ValueError) as exc:
        sys.exit(f"\n[serve-slides] {exc}")

    if not entries:
        sys.exit(f"\n[serve-slides] cohort '{cohort.name}' has no slides to serve.")

    run, run_label, run_classes = _resolve_run(cfg, cohort)

    n_masked = sum(1 for e in entries if e.mask_path is not None)
    logger.info(
        "cohort %s: %d slides (%d with a mask) via %s",
        cohort.name,
        len(entries),
        n_masked,
        cohort.storage.default_reader,
    )
    in_study = None
    if run is not None:
        in_study = _in_study_slides(cfg, cohort, str(cfg.run.study), run)
        covered = sum(1 for e in entries if e.slide_id in run.attention)
        logger.info(
            "run: %s | %d/%d slides covered | folds %s",
            run_label,
            covered,
            len(entries),
            ",".join(run.folds),
        )
        if in_study is not None:
            admitted = sum(1 for e in entries if e.slide_id in in_study)
            gaps = sum(1 for e in entries if e.slide_id in in_study and e.slide_id not in run.attention)
            logger.info(
                "study admits %d/%d slides for this endpoint; %d admitted without a prediction",
                admitted,
                len(entries),
                gaps,
            )
    logger.info(
        "open http://127.0.0.1:%d after tunnelling: ssh -N -L %d:%s:%d <this-host>",
        cfg.port,
        cfg.port,
        cfg.host,
        cfg.port,
    )

    app = create_app(
        entries,
        cohort_name=cohort.name,
        run=run,
        run_label=run_label,
        run_classes=run_classes,
        in_study=in_study,
        image_dir=Path(cohort.storage.image_dir),
        mask_dir=Path(cohort.storage.mask_dir) if cohort.storage.mask_dir else None,
        tile_size=int(cfg.tile_size),
        jpeg_quality=int(cfg.jpeg_quality),
    )
    uvicorn.run(app, host=str(cfg.host), port=int(cfg.port), log_level="info")


if __name__ == "__main__":
    main()
