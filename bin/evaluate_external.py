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
"""Externally validate a development sweep on a study's validation cohort.

    evaluate_external study=tcga_subtyping cohort=tcga_brca_coad_mskcc experiment_name=baseline
    evaluate_external study=tcga_subtyping cohort=tcga_brca_coad_mskcc experiment_name=baseline only_fields=[cancer_type]

Discovers every scorable endpoint the sweep trained (``runs/<study>/<experiment>/<dev_cohort>/
<field>_cv_*/best.ckpt``) that the validation ``cohort`` also declares. Scorable means it has a handler
in ``dlux.eval.external``; the rest are skipped with a warning. Scores each on the cohort's
``all_test_<field>`` split and writes one combined report (grand-ensemble headline + per-outer
stability), the same way ``aggregate`` handles every endpoint at once. ``only_fields`` narrows to a
subset. The study says which cohort is `development` (the models) and which is `validation` (the target).
A validation cohort that shares a patient with the development cohort is refused.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from ahcore.utils.io import get_logger, print_config

from dlux.config.cohort import Cohort, Role, Study
from dlux.config.resolvers import register_resolvers
from dlux.data import layout
from dlux.data.splits import format_all_test
from dlux.eval import trained_folds
from dlux.eval.external import ExternalResult, headline, supported_objectives, write_external_report

logger = get_logger(__name__)

register_resolvers()


@hydra.main(version_base=None, config_path="../config", config_name="evaluate_external")
def main(cfg: DictConfig) -> None:
    print_config(
        cfg,
        fields=(
            "study",
            "cohort",
            "paths",
            "datamodule",
            "experiment_name",
        ),
    )
    study = Study(**OmegaConf.to_container(cfg.study, resolve=True))  # type: ignore[arg-type]
    val_cohort = Cohort(**OmegaConf.to_container(cfg.cohort, resolve=True))  # type: ignore[arg-type]
    experiment_name = str(cfg.experiment_name)
    try:
        # The study declares which endpoints exist. The handler registry declares which can be reported on.
        plan = trained_folds.plan(
            study,
            val_cohort,
            role=Role.validation,
            flow="evaluate_external",
            experiment=experiment_name,
            studies_dir=cfg.paths.studies_dir,
            strict=bool(cfg.strict),
            only_fields=[str(f) for f in cfg.get("only_fields") or []],
            reportable=supported_objectives(),
        )
        for field in plan.fields:
            trained_folds.assert_disjoint_from_development(
                field, plan.dev_db_uri, plan.dev_cohort, plan.db_uri, val_cohort.name, format_all_test(field)
            )
        random_folds = trained_folds.random_baseline_folds(
            study,
            val_cohort,
            plan,
            cfg.get("random_experiment"),
            studies_dir=cfg.paths.studies_dir,
            strict=bool(cfg.strict),
        )
    except ValueError as exc:
        sys.exit(f"[evaluate_external] {exc}")

    logger.info(
        "Scoring %d endpoint(s) %s on '%s' (dev cohort '%s', experiment '%s').",
        len(plan.fields),
        plan.fields,
        val_cohort.name,
        plan.dev_cohort,
        experiment_name,
    )
    results: list[ExternalResult] = []
    for field in plan.fields:
        run_cfg = trained_folds.load_run_cfg(plan.folds[field][0][2].parent, cfg.paths)  # folds share the recipe
        res = trained_folds.evaluate_field(
            field,
            plan,
            val_cohort,
            run_cfg,
            cohorts_dir=Path(cfg.paths.cohorts_dir),
            trainer_cfg=cfg.trainer,
            num_workers=int(cfg.num_workers),
            random_folds=random_folds.get(field),
        )
        results.append(res)
        logger.info("%s", headline(res))

    out_dir = layout.results_dir(cfg.paths.studies_dir, study.name, experiment_name, val_cohort.name)
    write_external_report(results, experiment_name, out_dir)
    logger.info("Wrote external report (%d endpoint(s)) -> %s", len(results), out_dir)


if __name__ == "__main__":
    main()
