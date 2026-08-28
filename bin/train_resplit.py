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
submits more array tasks, recomputing nothing. See ``docs/specs/COMPARE_SPEC.md`` §6b.

A separate bin from ``train.py`` because four of its seven phases differ. It draws the split in memory,
keys identity on (seed, rep), and persists one row plus the per-patient NPZ instead of a run directory.
It reuses the same task/model builders, prediction collector, and metric function ``aggregate`` uses.

Nothing here writes to the database. The split lives only for the run. It is a pure function of ``seed``,
and ``split_hash`` in the row turns silent drift in that function into a visible mismatch.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from dlux.config.cohort import Cohort, Objective, SplitCategory, Study
from dlux.config.resolvers import register_resolvers
from dlux.data import layout
from dlux.data.splits import draw_split, field_codes_and_strata, format_cv_split, split_hash
from dlux.eval.compare import resolve_metric, score_predictions
from dlux.eval.persistence import get_git_sha
from dlux.eval.predictions import collect_predictions, write_predictions_npz
from dlux.models import build_lit_module
from dlux.tasks.build import build_task
from lightning.pytorch.trainer import Trainer
from omegaconf import DictConfig, OmegaConf

from ahcore.data.mm_dataset import MultiModalDataModule
from ahcore.manifest import DataManager, get_labels_from_record
from ahcore.transforms.tile import TileTransformFactory
from ahcore.utils.io import get_logger, print_config

logger = get_logger(__name__)

warnings.filterwarnings("ignore", message=r".*does not have many workers.*")


def _streamed_headline(streamed: dict) -> float:
    """The task's own streamed metric for this objective, or NaN when it logs none under that name."""
    for key in ("test/auroc", "test/c_index", "test/r2", "test/mse"):
        if key in streamed:
            return streamed[key]
    return float("nan")


# One JSON file per trial, keyed by (experiment, seed, rep). `metric` is the objective's headline
# scalar on the drawn TEST set — the per-patient one; `metric_slide` is the streamed per-slide value
# beside it, and their gap measures how much multi-slide patients pull the result.


def _trial_dir(cfg: DictConfig, study_name: str, cohort_name: str) -> Path:
    return Path(cfg.paths.studies_dir) / study_name / "resplits" / str(cfg.resplit_name) / cohort_name


def _row_path(rows_dir: Path, experiment: str, field: str, seed: int, rep: int) -> Path:
    """One file per trial, named by the run identity.

    Not one shared table: R x arms x reps array tasks finish concurrently, and appending to a single
    file would put hundreds of writers on it. A torn row would be silent and tedious to find. One file
    per trial has no writers in common, and existence IS the resumability sentinel — so re-submitting
    an array to add seeds costs a stat() per task instead of parsing a growing table."""
    return rows_dir / f"{experiment}__{field}__s{seed:04d}_r{rep}.json"


def _write_row(path: Path, row: dict) -> None:
    """Write via a temp file + atomic rename, so a killed task leaves no half-written row behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(row, indent=2))
    tmp.rename(path)


def _admissible(database_uri: str, manifest: str, field_name: str, field, require: set[str] | None):
    """The endpoint's admissible patients + their stratification labels.

    Population comes from the union of FIT/VALIDATE/TEST of one existing nested-CV split version:
    nested CV partitions the study population, so any single version's union IS the admitted set — and
    it inherits the study's ``require_modalities`` gate, with no re-derivation here.

    Strata come from the same resolver ``build_db`` uses, so a drawn split is stratified exactly the
    way a stored one is. Diverging here would make the comparison not like-for-like."""
    dm = DataManager(database_uri)
    version = format_cv_split(field_name, 0, 0)
    records: dict[str, Any] = {}
    for category in (SplitCategory.FIT, SplitCategory.VALIDATE, SplitCategory.TEST):
        for patient in dm.get_records_by_split(manifest_name=manifest, split_version=version, split_category=category):
            records[str(patient.patient_code)] = patient
    if not records:
        sys.exit(f"[train_resplit] no patients found for split version '{version}' — has build_db run?")
    patient_labels = {code: dict(get_labels_from_record(p) or []) for code, p in records.items()}
    # regression_vector's target is the external RNA matrix, not a sheet column, so its admissible set is
    # read from RNA coverage rather than labels. build_db already gated this per-field split version on
    # that coverage, so the split's patients are the covered set — hand them in instead of re-deriving.
    rnaseq_covered = {field_name: set(records)} if field.objective == Objective.regression_vector else None
    return field_codes_and_strata(field, field_name, patient_labels, rnaseq_covered, require)


register_resolvers()


@hydra.main(version_base=None, config_path="../config", config_name="train_resplit")
def main(cfg: DictConfig) -> None:
    print_config(
        cfg, fields=("study", "cohort", "experiment_name", "resplit_name", "seed", "rep", "ratios", "datamodule")
    )
    from datetime import datetime, timezone

    started = datetime.now(timezone.utc)

    study = Study(**OmegaConf.to_container(cfg.study, resolve=True))  # type: ignore[arg-type]
    cohort = Cohort(**OmegaConf.to_container(cfg.cohort, resolve=True))  # type: ignore[arg-type]
    target = str(cfg.task.target.field)
    contract_field = cohort.contract[target]
    seed, rep = int(cfg.seed), int(cfg.rep)

    rows_dir = _trial_dir(cfg, study.name, cohort.name) / "rows"
    row_path = _row_path(rows_dir, str(cfg.experiment_name), target, seed, rep)
    if row_path.exists():
        logger.info("[skip] %s seed=%d rep=%d already recorded.", cfg.experiment_name, seed, rep)
        return

    database_uri = layout.db_uri(cfg.paths.studies_dir, study.name, cohort.name)
    codes, strata = _admissible(database_uri, cohort.name, target, contract_field, None)
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

    # The split is seeded by `seed` alone, so every arm at one seed sees identical data. That is what
    # makes the comparison paired. Training randomness is seeded by (experiment, seed, rep), and all
    # three axes are required:
    #   rep        — without it rep=1 reproduces rep=0 and the null cloud collapses to zeros;
    #   experiment — without it arm A and arm B share a torch seed at every seed, which makes
    #                delta_real quieter than delta_null and biases the comparison toward declaring an
    #                effect (COMPARE_SPEC "The trap: identical randomness structure"). It would not
    #                even help: two architectures consume the RNG differently, so a shared seed buys
    #                no matched randomness — but two arms differing only in a knob (the lr +/-1%
    #                negative control) do share an architecture, and there it bites for real.
    # sha256 over the arm name rather than hash(): Python salts str hashing per process, so hash()
    # would give a different training seed on every invocation and nothing would be reproducible.
    arm_salt = int(hashlib.sha256(str(cfg.experiment_name).encode()).hexdigest()[:8], 16)
    train_seed = (arm_salt + seed * 1_000_003 + rep * 7_919) % (2**31 - 1)
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)

    # Splits live only in memory here, so the compiled DataDescription must carry split_version=None
    # and the membership travels as an explicit per-stage patient set.
    tiling = OmegaConf.to_container(cfg.tiling, resolve=True)
    # The drawn `fit` set is already the authority for what the model trains on (it reaches the
    # datamodule as patient_codes_by_stage below), so the fit-time statistics read from it too. There
    # is no stored split to name here.
    task = build_task(
        cfg, cohort, contract_field, tiling, database_uri, split_version=None, fit_patient_codes=drawn["fit"]
    )
    task.register_adapter_augmentations(hydra.utils.instantiate(cfg.augmentations))
    lit_module = build_lit_module(cfg, task)
    datamodule = MultiModalDataModule(
        data_description=task.data_description,
        task=task,
        tile_transform=TileTransformFactory.for_mil_classification(),
        batch_size=cfg.datamodule.batch_size,
        eval_batch_size=cfg.datamodule.eval_batch_size,
        num_workers=cfg.datamodule.num_workers,
        min_tiles=cfg.datamodule.min_tiles,
        patient_codes_by_stage={k: set(v) for k, v in drawn.items()},
    )

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        logger=False,  # no mlflow: R x arms x reps runs would bury the study's real experiments
        callbacks=list(hydra.utils.instantiate(cfg.callbacks).values()),
        enable_checkpointing=True,
    )
    trainer.fit(lit_module, datamodule)
    best = getattr(trainer.checkpoint_callback, "best_model_path", "") or ""
    if best and Path(best).exists():
        lit_module.load_state_dict(torch.load(best, weights_only=False, map_location="cpu")["state_dict"])
    else:
        logger.warning("No best checkpoint; scoring last-epoch weights.")

    # Runs the test loop exactly as train.py does: it builds the test dataset, and its streamed
    # torchmetrics value is recorded as provenance. It is not the headline metric — torchmetrics scores
    # per slide, and every reported dlux number is per patient. On a cohort with multi-slide patients
    # the two differ, so taking it here would let a resplit trial disagree with the stage-1 comparison
    # it feeds. Keeping both is diagnostic: a systematic gap between the columns is a bug signature.
    trainer.test(lit_module, datamodule)
    streamed = {k: float(v) for k, v in trainer.callback_metrics.items() if str(k).startswith("test/")}
    preds = collect_predictions(
        lit_module,
        datamodule.test_dataloader(),
        endpoint_type=contract_field.objective,
        num_classes=task.num_classes,
        target_key=f"patient.{contract_field.source.column}",
        target_mean=getattr(task, "target_mean", 0.0),
        target_std=getattr(task, "target_std", 1.0),
        time_edges=getattr(task, "time_edges", None),
    )
    out_dir = _trial_dir(cfg, study.name, cohort.name)
    npz = out_dir / "predictions" / f"{cfg.experiment_name}__{target}__s{seed:04d}_r{rep}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    write_predictions_npz(preds, npz)

    objective = contract_field.objective.value
    metric_spec = resolve_metric(objective, cfg.get("metric"))  # the metric to score every trial on
    metric, n_test, n_pos = score_predictions(preds, objective, metric_spec.key)
    _write_row(
        row_path,
        {
            "experiment": str(cfg.experiment_name),
            "field": target,
            "seed": seed,
            "rep": rep,
            "metric_name": metric_spec.label,
            "metric_key": metric_spec.key,
            "metric": metric,
            # Streamed per slide. `metric` above is the per-patient one aggregate reports; their gap
            # measures how much multi-slide patients pull the result.
            "metric_slide": _streamed_headline(streamed),
            # Declared by the draw vs actually scored. They differ when a patient's slides all fall
            # below min_tiles at this grid — the same coverage gap aggregate reports, kept per trial
            # so a systematic loss cannot hide inside the cloud.
            "n_declared": len(drawn["test"]),
            "n_test": n_test,
            "n_pos": n_pos,
            "split_hash": digest,
            "train_seed": train_seed,
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
        row_path,
    )


if __name__ == "__main__":
    main()
