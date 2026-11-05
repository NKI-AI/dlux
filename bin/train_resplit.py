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
"""Hydra entry point: one resplit trial — one config, one seed-drawn split, one replicate.

    train_resplit study=tcga_subtyping cohort=tcga_brca_coad_christiana \
        experiment=tcga_subtyping/baseline task.target.field=cancer_type seed=0 rep=0

A comparison needs many of these. The sweep is a slurm array, not a loop here. Run identity is
``(experiment, seed, rep)``, and the results row is the resumability sentinel: adding seeds later just
submits more array tasks, recomputing nothing. See ``docs/specs/COMPARE_SPEC.md`` §1.4.

A separate bin from ``train.py`` because four of its seven phases differ. It draws the split in memory,
keys identity on (seed, rep), and persists one row plus the per-patient NPZ instead of a run directory.
It reuses the task and model builders, the fit and test (``dlux.training.fit_and_test``), the prediction
collector, and the metric function ``aggregate`` uses.

Nothing here writes to the database. The split lives only for the run. It is a pure function of ``seed``,
and ``split_hash`` in the row turns silent drift in that function into a visible mismatch.
"""

from __future__ import annotations

import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch import seed_everything
from omegaconf import DictConfig

from ahcore.manifest import DataManager, get_labels_from_record
from ahcore.utils.io import get_logger, print_config

from dlux.config.cohort import Objective, SplitCategory
from dlux.config.resolvers import register_resolvers
from dlux.data import layout
from dlux.data.splits import draw_split, field_codes_and_strata, format_cv_split, split_hash
from dlux.eval.compare import resolve_metric, score_predictions
from dlux.eval.persistence import get_git_sha
from dlux.eval.predictions import write_predictions_npz
from dlux.eval.resplit import row_path, train_seed, write_row
from dlux.sample_units import slide_predictions
from dlux.training import (
    build_setup,
    check_experiment_choice,
    check_target_choice,
    fit_and_test,
    load_study_and_cohort,
)

logger = get_logger(__name__)

warnings.filterwarnings("ignore", message=r".*does not have many workers.*")


def _admissible(database_uri: str, manifest: str, field_name: str, field):
    """The endpoint's admissible patients + their stratification labels.

    Population comes from the union of FIT/VALIDATE/TEST of one existing nested-CV split version:
    nested CV partitions the study population, so any single version's union IS the admitted set — and
    it inherits the study's ``require_modalities`` gate, with no re-derivation here.

    Strata come from the same resolver ``build_db`` uses, so a drawn split is stratified exactly the
    way a stored one is. Diverging here would make the comparison not like-for-like."""
    version = format_cv_split(field_name, 0, 0)
    records: dict[str, Any] = {}
    with DataManager(database_uri) as dm:
        for category in (SplitCategory.FIT, SplitCategory.VALIDATE, SplitCategory.TEST):
            for patient in dm.get_records_by_split(
                manifest_name=manifest, split_version=version, split_category=category
            ):
                records[str(patient.patient_code)] = patient
        if not records:
            sys.exit(f"[train_resplit] no patients found for split version '{version}'. Has build_db run?")
        patient_labels = {code: dict(get_labels_from_record(p) or []) for code, p in records.items()}
    # regression_vector's target is the external RNA matrix, not a sheet column, so its admissible set is
    # read from RNA coverage rather than labels. build_db already gated this per-field split version on
    # that coverage, so the split's patients are the covered set — hand them in instead of re-deriving.
    rnaseq_covered = {field_name: set(records)} if field.objective == Objective.regression_vector else None
    return field_codes_and_strata(field, field_name, patient_labels, rnaseq_covered)


register_resolvers()


@hydra.main(version_base=None, config_path="../config", config_name="train_resplit")
def main(cfg: DictConfig) -> None:
    print_config(
        cfg, fields=("study", "cohort", "experiment_name", "resplit_name", "seed", "rep", "ratios", "datamodule")
    )

    started = datetime.now(timezone.utc)

    study, cohort = load_study_and_cohort(cfg, "train_resplit")
    target = str(cfg.task.target.field)
    # The same invocation checks as train, before the skip below: a mislabelled arm would otherwise write
    # its rows under another arm's name, and the skip would then pass over that arm's own trials.
    check_experiment_choice(
        str(HydraConfig.get().runtime.choices.get("experiment", "")), str(cfg.experiment_name), study.name
    )
    check_target_choice(target, format_cv_split(target, 0, 0), study.targets)
    contract_field = cohort.contract[target]
    seed, rep = int(cfg.seed), int(cfg.rep)

    rows_dir = layout.resplit_rows_dir(cfg.paths.studies_dir, study.name, cfg.resplit_name, cohort.name)
    row_file = row_path(rows_dir, str(cfg.experiment_name), target, seed, rep)
    if row_file.exists():
        logger.info("[skip] %s seed=%d rep=%d already recorded.", cfg.experiment_name, seed, rep)
        return

    database_uri = layout.db_uri(cfg.paths.studies_dir, study.name, cohort.name)
    codes, strata = _admissible(database_uri, cohort.name, target, contract_field)
    drawn = draw_split(codes, strata, ratios=tuple(cfg.ratios), seed=seed)
    digest = split_hash(drawn)
    logger.info(
        "seed=%d rep=%d -> fit %d / validate %d / test %d (split_hash %s)",
        seed,
        rep,
        len(drawn["fit"]),
        len(drawn["validate"]),
        len(drawn["test"]),
        digest,
    )

    trial_seed = train_seed(str(cfg.experiment_name), seed, rep)  # see dlux.eval.resplit.train_seed
    # Same reach as the ordinary train entry point: python, numpy and torch, plus a derived seed per
    # dataloader worker, so a trial's data order repeats and not just its weight init.
    seed_everything(trial_seed, workers=True)

    # The drawn `fit` set is the authority for what the model trains on, so the fit-time statistics read
    # from it too.
    setup = build_setup(cfg, study, cohort, target, drawn=drawn, tool="train_resplit")
    task, lit_module, datamodule = setup.task, setup.lit_module, setup.datamodule

    # No mlflow: R x arms x reps runs would bury the study's real experiments.
    fit_and_test(
        cfg,
        task,
        lit_module,
        datamodule,
        run_logger=False,
        random_baseline=bool(cfg.random_baseline),
        output_dir=Path(HydraConfig.get().runtime.output_dir),
    )
    fit_state = task.fit_state()
    preds = slide_predictions(
        task,
        lit_module,
        datamodule,
        "test",
        trainer_cfg=cfg.trainer,
        endpoint_type=contract_field.objective,
        target_key=f"patient.{target}",
        fit_state=fit_state,
    )
    predictions_dir = layout.resplit_predictions_dir(cfg.paths.studies_dir, study.name, cfg.resplit_name, cohort.name)
    npz = predictions_dir / f"{cfg.experiment_name}__{target}__s{seed:04d}_r{rep}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    write_predictions_npz(preds, npz)

    objective = contract_field.objective.value
    metric_spec = resolve_metric(objective, cfg.get("metric"))  # the metric to score every trial on
    metric, n_test, n_pos = score_predictions(preds, objective, metric_spec.key)
    write_row(
        row_file,
        {
            "experiment": str(cfg.experiment_name),
            "field": target,
            "seed": seed,
            "rep": rep,
            "metric_name": metric_spec.label,
            "metric_key": metric_spec.key,
            "metric": metric,
            # Declared by the draw vs actually scored. They differ when a patient's slides all fall
            # below min_tiles at this grid — the same coverage gap aggregate reports, kept per trial
            # so a systematic loss cannot hide inside the cloud.
            "n_declared": len(drawn["test"]),
            "n_test": n_test,
            "n_pos": n_pos,
            "split_hash": digest,
            "train_seed": trial_seed,
            "git_sha": get_git_sha(os.environ.get("BUILD_WORKING_DIRECTORY", os.getcwd())),
            "wall_s": int((datetime.now(timezone.utc) - started).total_seconds()),
        },
    )
    logger.info(
        "[%s] seed=%d rep=%d %s = %.4f (n=%d) -> %s",
        cfg.experiment_name,
        seed,
        rep,
        metric_spec.label,
        metric,
        n_test,
        row_file,
    )


if __name__ == "__main__":
    main()
