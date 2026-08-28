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
"""Aggregate a nested-CV sweep into per-outer-fold metrics + figures.

    aggregate experiment_name=tcga_uni2_v1

Scans ``${paths.studies_dir}/<study>/runs/<experiment_name>`` for completed folds (each fold's
``test_predictions.npz`` gated by its ``metadata.json`` sentinel). Ensembles the inner replicates per
outer fold, then writes CSVs, summary.md, and figures into
``${paths.studies_dir}/<study>/results/<experiment_name>``. Figures are ROC + AUROC-by-fold for
classification, predicted-vs-actual scatter + R²-by-fold for regression.
"""

from __future__ import annotations

from pathlib import Path

import hydra
from dlux.config.cohort import Study
from dlux.eval.aggregate import aggregate_experiment, endpoint_headline, write_reports
from omegaconf import DictConfig, OmegaConf

from ahcore.utils.io import get_logger

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="aggregate")
def main(cfg: DictConfig) -> None:
    # The study declares what this sweep was supposed to produce. Discovery only says what it did.
    study = Study(**OmegaConf.to_container(cfg.study, resolve=True))  # type: ignore[arg-type]
    study_name, cohort_name = study.name, str(cfg.cohort.name)
    experiment_name = str(cfg.experiment_name)
    label = f"{study_name}/{experiment_name}/{cohort_name}"
    experiment_dir = Path(cfg.paths.studies_dir) / study_name / "runs" / experiment_name / cohort_name
    out_dir = Path(cfg.paths.studies_dir) / study_name / "results" / experiment_name / cohort_name

    # Optional random-baseline experiment (expression) -> SEQUOIA conservative significant-gene count.
    random_experiment = cfg.get("random_experiment", None)
    random_dir = (
        Path(cfg.paths.studies_dir) / study_name / "runs" / str(random_experiment) / cohort_name
        if random_experiment
        else None
    )
    results = aggregate_experiment(
        experiment_dir,
        expected_fields=study.targets,
        n_outer=study.splits.n_outer,
        n_inner=study.splits.n_inner,
        strict=bool(cfg.get("strict", True)),
        random_experiment_dir=random_dir,
    )
    write_reports(results, label, out_dir)

    for res in results:
        logger.info(endpoint_headline(res))
    logger.info(f"Wrote aggregate reports -> {out_dir}")


if __name__ == "__main__":
    main()
