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
"""Run a development sweep's models over an unlabeled cohort and persist predictions — no metrics.

    predict study=tcga_subtyping cohort=some_unlabeled_cohort experiment_name=baseline
    predict study=... cohort=... experiment_name=baseline only_fields=[subtype]

The sibling of ``evaluate_external``: same load-checkpoints-and-run spine, but the target cohort has a
``predict`` role (``all_predict`` split, every patient, no labels), so there is nothing to score. For
each endpoint the dev sweep trained that the cohort also declares, this grand-ensembles the outer folds'
per-patient predictions. It writes ``results/<experiment>/<cohort>/``: ``per_patient_<field>.npz``, a csv,
the first fold's attention, and summary.md. Whether a prediction is any good is unknowable here. ``evaluate_external`` answers that on
a labeled cohort.
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
from dlux.eval import trained_folds

logger = get_logger(__name__)

register_resolvers()


@hydra.main(version_base=None, config_path="../config", config_name="predict")
def main(cfg: DictConfig) -> None:
    print_config(
        cfg,
        fields=("study", "cohort", "paths", "datamodule", "experiment_name"),
    )
    study = Study(**OmegaConf.to_container(cfg.study, resolve=True))  # type: ignore[arg-type]
    cohort = Cohort(**OmegaConf.to_container(cfg.cohort, resolve=True))  # type: ignore[arg-type]
    experiment_name = str(cfg.experiment_name)
    try:
        # The cohort's contract must declare each target's type and space even though it carries no values.
        # Inference needs no metric, so every objective the task supports can be predicted.
        plan = trained_folds.plan(
            study,
            cohort,
            role=Role.predict,
            flow="predict",
            experiment=experiment_name,
            studies_dir=cfg.paths.studies_dir,
            strict=bool(cfg.strict),
            only_fields=[str(f) for f in cfg.get("only_fields") or []],
        )
    except ValueError as exc:
        sys.exit(f"[predict] {exc}")
    out_dir = layout.results_dir(cfg.paths.studies_dir, study.name, experiment_name, cohort.name)

    logger.info(
        "Predicting %d endpoint(s) %s on '%s' (dev cohort '%s', experiment '%s').",
        len(plan.fields),
        plan.fields,
        cohort.name,
        plan.dev_cohort,
        experiment_name,
    )
    predictions = []
    for field in plan.fields:
        run_cfg = trained_folds.load_run_cfg(plan.folds[field][0][2].parent, cfg.paths)  # folds share the recipe
        predictions.append(
            trained_folds.predict_field(
                field,
                plan,
                cohort,
                run_cfg,
                cohorts_dir=Path(cfg.paths.cohorts_dir),
                trainer_cfg=cfg.trainer,
                num_workers=int(cfg.num_workers),
            )
        )
    trained_folds.write_cohort_predictions(predictions, experiment_name, out_dir)
    for prediction in predictions:
        logger.info("[%s] wrote predictions for %d patients -> %s", prediction.field, len(prediction.patients), out_dir)


if __name__ == "__main__":
    main()
