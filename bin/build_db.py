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
"""Hydra entry point: materialise one cohort's DB for a study.

    build_db study=tcga_subtyping cohort=tcga_brca_coad_christiana \
             patients_csv=/path/patients.csv slides_csv=/path/slides.csv

The ``study`` assigns the cohort its role (→ split strategy) and owns the CV params. The ``cohort``
describes the data source (inputs + contract). Both are plain structured configs (no ``_target_``),
constructed as typed ``Study``/``Cohort`` here so they are validated. The DB is written to
``{paths.studies_dir}/{study}/db/{cohort}.db``. Run once per cohort in the study.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from dlux.config.cohort import Cohort, Study
from dlux.data import layout
from dlux.data.build_db import build_db
from dlux.data.errors import BuildDbError
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from ahcore.utils.io import print_config


@hydra.main(version_base=None, config_path="../config", config_name="build_db")
def main(cfg: DictConfig) -> None:
    print_config(
        cfg,
        fields=(
            "study",
            "cohort",
            "paths",
            "patients_csv",
            "slides_csv",
            "probe",
            "probe_grid",
            "splits",
            "overwrite",
            "analyze",
        ),
    )
    try:
        study = Study(**OmegaConf.to_container(cfg.study, resolve=True))  # type: ignore[arg-type]
        cohort = Cohort(**OmegaConf.to_container(cfg.cohort, resolve=True))  # type: ignore[arg-type]
        if cohort.name not in study.cohorts:
            raise BuildDbError(
                f"cohort '{cohort.name}' is not part of study '{study.name}' (has: {sorted(study.cohorts)}). "
                f"Pass a cohort listed in the study, or add it to study/{study.name}.yaml."
            )
        cohort = study.filter_contract(cohort)  # scope to the study's targets (splits + analysis)
        splits = study.splits_for(cohort.name)  # role -> strategy, composed with the study's params
        database_uri = layout.db_uri(cfg.paths.studies_dir, study.name, cohort.name)

        probe_grid = OmegaConf.to_container(cfg.probe_grid, resolve=True) if cfg.get("probe_grid") is not None else None
        splits_cfg = cfg.get("splits")
        import_from = splits_cfg.get("import_from") if splits_cfg is not None else None
        build_db(
            cohort,
            splits,
            database_uri,
            patients_csv=Path(cfg.patients_csv),
            slides_csv=Path(cfg.slides_csv),
            probe=str(cfg.probe),
            probe_grid=probe_grid,  # type: ignore[arg-type]
            overwrite=bool(cfg.overwrite),
            analyze=bool(cfg.analyze),
            cohorts_dir=Path(cfg.paths.cohorts_dir),
            admin_censor=study.admin_censor,
            import_splits=Path(import_from) if import_from else None,
            import_allow_uncovered=bool(splits_cfg.get("allow_uncovered", False)) if splits_cfg is not None else False,
        )
    except (BuildDbError, ValidationError) as exc:
        # Expected, user-facing failure: print a clean message, no traceback.
        sys.exit(f"\n[build-db] {exc}")


if __name__ == "__main__":
    main()
