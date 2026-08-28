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
"""Train one fold of WSI classification on precomputed features.

    train study=tcga_subtyping cohort=tcga_brca_coad_christiana task.target.field=cancer_type \
          fold=0 experiment=tcga_subtyping/baseline

`fold` is the flat nested-CV index, resolved against the study's n_outer x n_inner grid.
`split_version=cancer_type_cv_o0_i0` names a stored split directly, and is the only way to reach the
non-CV families. Pass exactly one of the two.

Reads the feature cache from `extract_features` (same cohort + tiling + feature_extractor => same
cache), aggregates each slide's tile-feature bag into one slide vector with attention-MIL, fits on the
fold, then tests the best-by-`validate/loss` checkpoint. Persists {best.ckpt, predictions, metadata}
into the canonical runs dir for cross-fold aggregation.

Hydra run dirs are disposable. What is kept is copied into
`${paths.studies_dir}/<study>/runs/<experiment_name>/<cohort>/<split_version>/`.

`main` is a thin spine: resolve -> build -> mlflow -> fit/test -> write -> persist, one `_`-helper per
phase. `RunContext` (the run's identity) and `TrainingSetup` (task, model, data) carry state between them.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import hydra
import torch
from dlux.config.cohort import Cohort, Objective, SplitCategory, Study
from dlux.config.resolvers import register_resolvers
from dlux.data import layout
from dlux.data.splits import format_cv_split, parse_cv_split
from dlux.eval.attention import write_attention_npz
from dlux.eval.compare import METRIC_KEYS, score_predictions
from dlux.eval.persistence import (
    claim_hydra_run_dir,
    get_git_sha,
    persist_run_artifacts,
    persistent_run_dir,
    run_already_persisted,
)
from dlux.eval.predictions import collect_predictions, write_predictions_csv, write_predictions_npz
from dlux.modalities.state import STATE_FILENAME, write_stream_state
from dlux.models import build_lit_module
from dlux.tasks.build import build_task
from dlux.tracking import build_mlflow_logger
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.trainer import Trainer
from omegaconf import DictConfig, OmegaConf

from ahcore.data.mm_dataset import MultiModalDataModule
from ahcore.manifest import DataManager
from ahcore.transforms.tile import TileTransformFactory
from ahcore.utils.io import get_logger, print_config

logger = get_logger(__name__)

warnings.filterwarnings("ignore", message=r".*does not have many workers.*")


@dataclass(frozen=True)
class RunContext:
    """The run's identity: every name and path that pins down which fold of which experiment this is.
    Resolved once from cfg + Hydra, then read by every phase. Frozen, so no phase can redefine it."""

    output_dir: Path  # disposable hydra dir
    run_dir: Path  # canonical persistent run dir
    study_name: str
    cohort_name: str
    experiment_name: str
    split_version: str
    exp_choice: str  # the chosen experiment config (recipe provenance)
    target: str
    target_key: str
    target_normalize: str
    force: bool
    random_baseline: bool
    train_start_iso: str


@dataclass
class TrainingSetup:
    """The constructed task, lightning module and datamodule, plus the identity bits that only the build
    can resolve: the endpoint's objective, and which feature extractor produced the cache being read."""

    task: Any
    lit_module: Any
    datamodule: Any
    study: Study
    objective: Objective
    model_name: str
    model_hash: str
    database_uri: str


@dataclass
class PredictionArtifacts:
    """What the prediction pass produced: the files to persist, plus the patients actually scored."""

    npz_path: Path
    csv_path: Path
    writes_csv: bool
    attention_path: Optional[Path]
    scored_patients: set[str]
    patient_metrics: dict  # test/<name>_patient — the number aggregate reports (see _patient_test_metric)


def _run_key(split_version: Optional[str], field: str, fold: Optional[int]) -> str:
    """The fold's name for the hydra scratch path, resolved at compose time.

    Hydra builds ``hydra.run.dir`` before ``main`` runs, so the split name that ``_resolve_split_version``
    derives from ``fold`` does not exist yet and cannot be interpolated into the path. ``fold`` and
    ``field`` are known at compose time, so this name is resolvable.

    Only names a disposable directory. Collisions are prevented by ``claim_hydra_run_dir``'s O_EXCL
    marker, not this string.
    """
    if split_version is not None:
        return str(split_version)
    return f"{field}_fold{fold}"


OmegaConf.register_new_resolver("dlux.run_key", _run_key, replace=True)
register_resolvers()


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    print_config(
        cfg,
        fields=(
            "study",
            "cohort",
            "tiling",
            "feature_extractor",
            "task",
            "lit_module",
            "augmentations",
            "callbacks",
            "paths",
            "datamodule",
            "trainer",
            "experiment_name",
            "split_version",
            "force",
        ),
    )
    # -- 1. work out which run this is ----------------------------------------
    ctx = _resolve_run_context(cfg)
    if ctx is None:  # already-persisted fold -> clean skip
        return

    # -- 2. the run itself ----------------------------------------------------
    # All the actual work is these four calls: assemble the model and this fold's data, open the
    # mlflow run that will receive the metrics, fit and test the best checkpoint, then make a final
    # pass to collect the per-slide predictions. Steps 1 and 3 are naming and bookkeeping around them.
    #
    # The build has to come first: it resolves the model widths into cfg, and mlflow snapshots cfg as
    # the run's config.yaml artifact — in this order, that snapshot is the config the run really used.
    setup = _build_model_and_data(cfg, ctx)
    mlflow_logger = _configure_mlflow(cfg, ctx, setup)
    final_epoch, best_ckpt_path, test_metrics = _fit_and_test(cfg, ctx, setup, mlflow_logger)
    preds = _write_predictions(ctx, setup, OmegaConf.to_container(cfg.tiling, resolve=True))
    # Only now do both test numbers exist: the streamed per-slide one from trainer.test, and the
    # per-patient one that needed the collected predictions. Logged together, without a step, so each
    # renders as a single bar rather than a point pinned to the final training global_step.
    test_metrics.update(preds.patient_metrics)
    for key, value in test_metrics.items():
        mlflow_logger.experiment.log_metric(mlflow_logger.run_id, key, value)

    # -- 3. persist to the canonical run dir (metadata.json written last) ------
    # The hydra dir is disposable, so kept artifacts are copied out here. metadata.json is
    # written last and acts as the sentinel: a fold without it counts as unfinished, both for the
    # already-persisted check in step 1 and for the cross-fold aggregate.
    metadata = _run_metadata(
        cfg,
        ctx,
        setup,
        final_epoch=final_epoch,
        best_ckpt_path=best_ckpt_path,
        test_metrics=test_metrics,
        test_coverage=_test_coverage(ctx, setup, preds.scored_patients),
    )
    OmegaConf.save(cfg, ctx.output_dir / "hparams.yaml")
    artifacts = {
        "best.ckpt": best_ckpt_path or (ctx.output_dir / "checkpoints" / "missing.ckpt"),
        "hparams.yaml": ctx.output_dir / "hparams.yaml",
        "test_predictions.npz": preds.npz_path,
    }
    if preds.writes_csv:  # only require the CSV where it is actually produced (scalar endpoints)
        artifacts["test_predictions.csv"] = preds.csv_path
    if preds.attention_path is not None:  # only models with an attention branch produce one
        artifacts["attention.npz"] = preds.attention_path
    # Fit-derived stream state (bulk RNA's per-gene μ/σ): a cohort with no fit split cannot recompute it,
    # so the run records it. Only streams with such state produce a file, hence the None check. It must
    # be registered here, because persist_run_artifacts copies this map and nothing else.
    state_path = write_stream_state(setup.task, ctx.output_dir)
    if state_path is not None:
        artifacts[STATE_FILENAME] = state_path
    persist_run_artifacts(run_dir=ctx.run_dir, artifacts=artifacts, metadata=metadata, force=ctx.force)


def _resolve_run_context(cfg: DictConfig) -> Optional[RunContext]:
    """Resolve the run identity, validate the experiment choice, and claim the disposable hydra dir.
    Returns None (caller exits 0) when the fold is already persisted and not forced — a clean resumable
    skip, not an error."""
    output_dir = Path(HydraConfig.get().runtime.output_dir)  # disposable
    study_name, cohort_name = str(cfg.study.name), str(cfg.cohort.name)
    experiment_name = str(cfg.experiment_name)
    exp_choice = str(HydraConfig.get().runtime.choices.get("experiment", ""))
    _check_experiment_choice(exp_choice, experiment_name, study_name)
    # Validate the invocation before the already-persisted skip: a mismatched target on a fold that
    # happens to exist must be refused, not silently reported as "already done".
    target = str(cfg.task.target.field)
    split_version = _resolve_split_version(cfg, target)
    _check_target_choice(target, split_version, cfg.study.get("targets") or [])

    force = bool(cfg.get("force", False))
    run_dir = persistent_run_dir(cfg.paths.studies_dir, study_name, experiment_name, cohort_name, split_version)
    rel = f"{study_name}/{experiment_name}/{cohort_name}/{split_version}"
    if run_already_persisted(run_dir) and not force:
        logger.info(f"[skip] {rel}: already persisted — pass force=true to overwrite.")
        return None
    claim_hydra_run_dir(output_dir, experiment_name=experiment_name, split_version=split_version)  # collision guard

    return RunContext(
        output_dir=output_dir,
        run_dir=run_dir,
        study_name=study_name,
        cohort_name=cohort_name,
        experiment_name=experiment_name,
        split_version=split_version,
        exp_choice=exp_choice,
        target=target,
        target_key=f"patient.{target}",
        target_normalize=str(cfg.task.get("target_normalize", "none")),  # provenance (metadata); applied in build_task
        force=force,
        random_baseline=bool(cfg.get("random_baseline", False)),
        train_start_iso=datetime.now(timezone.utc).isoformat(),
    )


def _resolve_split_version(cfg: DictConfig, target: str) -> str:
    """The fold to train, named either by ``fold`` (a flat nested-CV index) or ``split_version``.

    A sweep passes ``fold=k``: ``divmod`` against the study's ``n_inner`` gives (outer, inner), and
    :func:`format_cv_split` names it. The grid is stated once, in the study config. ``split_version=``
    names a stored split directly, for rerunning one fold by name. Persisted runs are keyed by split
    (``runs/<exp>/<cohort>/er_cv_o3_i2/``), so that is the name a gap appears under, and passing it back
    skips resolving the flat index."""
    fold, split_version = cfg.get("fold"), cfg.get("split_version")
    if (fold is None) == (split_version is None):
        sys.exit("[train] pass exactly one of fold=<int> (nested CV) or split_version=<name>.")
    if split_version is not None:
        return str(split_version)

    n_outer, n_inner = int(cfg.study.splits.n_outer), int(cfg.study.splits.n_inner)
    fold = int(fold)
    if not 0 <= fold < n_outer * n_inner:
        sys.exit(
            f"[train] fold={fold} is outside study '{cfg.study.name}'s {n_outer}x{n_inner} grid "
            f"(0..{n_outer * n_inner - 1}) — adjust the sbatch --array bound to match the study."
        )
    outer, inner = divmod(fold, n_inner)
    return format_cv_split(target, outer, inner)


def _check_target_choice(target: str, split_version: str, study_targets: Sequence[str]) -> None:
    """Guard the endpoint. A run names it three times: the study declares it in ``targets``, the split
    encodes it (``os_cv_o0_i0``), and ``task.target.field`` selects it. Nothing forces the three to
    agree, so this cross-checks them. ``task.target.field=grade split_version=os_cv_o0_i0`` would train
    grade labels on the os folds under an os-named directory: wrong labels, no error. Non-CV splits
    (``all_test_<field>``) don't parse and are not checked here."""
    parsed = parse_cv_split(split_version)
    if parsed is not None and parsed[0] != target:
        sys.exit(
            f"[train] task.target.field='{target}' but split_version '{split_version}' holds the folds for "
            f"'{parsed[0]}' — the labels and the folds would come from different endpoints."
        )
    if study_targets and target not in study_targets:
        sys.exit(
            f"[train] task.target.field='{target}' is not one of study.targets {sorted(study_targets)} — "
            f"either target a declared endpoint or declare this one on the study."
        )


def _check_experiment_choice(exp_choice: str, experiment_name: str, study_name: str) -> None:
    """Guard the study-nested experiment config (catalog/experiment/<study>/<name>.yaml): the chosen
    config's leaf must match experiment_name, and a study-nested experiment must match the study — so a
    recipe can't be silently mislabelled or applied to the wrong study."""
    exp_leaf = exp_choice.rsplit("/", 1)[-1]
    if exp_leaf and exp_leaf != experiment_name:
        sys.exit(f"[train] experiment_name '{experiment_name}' != chosen experiment leaf '{exp_leaf}' ({exp_choice}).")
    if "/" in exp_choice and exp_choice.rsplit("/", 1)[0] != study_name:
        sys.exit(
            f"[train] experiment '{exp_choice}' is nested under a different study than '{study_name}' — "
            f"move it or select the matching study."
        )


def _build_model_and_data(cfg: DictConfig, ctx: RunContext) -> TrainingSetup:
    """Build the task, the lightning module, and the datamodule for this fold.

    The task holds the endpoint's objective->task dispatch and its fit-time setup (target statistics
    computed on the fit split only, so no validation/test leakage). Building mutates ``cfg`` with the
    runtime-resolved model widths, which then travel into the config.yaml artifact."""
    study = Study(**OmegaConf.to_container(cfg.study, resolve=True))  # type: ignore[arg-type]
    cohort = Cohort(**OmegaConf.to_container(cfg.cohort, resolve=True))  # type: ignore[arg-type]
    if cohort.name not in study.cohorts:
        sys.exit(f"[train] cohort '{cohort.name}' is not part of study '{study.name}' (has: {sorted(study.cohorts)}).")
    database_uri = layout.db_uri(cfg.paths.studies_dir, study.name, cohort.name)

    tiling = OmegaConf.to_container(cfg.tiling, resolve=True)
    contract_field = cohort.contract[ctx.target]

    try:
        task = build_task(cfg, cohort, contract_field, tiling, database_uri, ctx.split_version)
    except ValueError as exc:
        sys.exit(f"[train] {exc}")
    task.register_adapter_augmentations(hydra.utils.instantiate(cfg.augmentations))

    tiles = task.inputs.get("tile_features")
    lit_module = build_lit_module(cfg, task)

    datamodule = MultiModalDataModule(
        data_description=task.data_description,
        task=task,
        tile_transform=TileTransformFactory.for_mil_classification(),
        batch_size=cfg.datamodule.batch_size,
        eval_batch_size=cfg.datamodule.eval_batch_size,
        num_workers=cfg.datamodule.num_workers,
        min_tiles=cfg.datamodule.min_tiles,
    )
    return TrainingSetup(
        task=task,
        lit_module=lit_module,
        datamodule=datamodule,
        study=study,
        objective=contract_field.objective,
        model_name=tiles.model_name if tiles else "",
        model_hash=tiles.model_hash if tiles else "",
        database_uri=database_uri,
    )


def _gene_panel(cfg: DictConfig) -> str:
    """The active gene panel, for the mlflow column — read from wherever this arm keeps it.

    A panel parameterises the bulk-RNA stream, so it lives on whichever side that stream is used: on an
    input for a fusion arm, on the target for an arm predicting the profile. It is not a task-level key,
    so this looks in both places."""
    inputs = OmegaConf.select(cfg, "task.inputs") or {}
    for spec in inputs.values():
        panel = spec.get("gene_panel") if hasattr(spec, "get") else None
        if panel:
            return str(panel)
    return str(OmegaConf.select(cfg, "task.target.gene_panel") or "none")


def _configure_mlflow(cfg: DictConfig, ctx: RunContext, setup: TrainingSetup) -> MLFlowLogger:
    """mlflow wiring. The experiment groups the sweep's folds; run_name is the split_version (this fold).
    Tags are organizational slices (fold / target / encoder). Curated params are the handful of knobs we
    group/scatter results by. The full resolved config is logged as a config.yaml artifact, so provenance
    stays complete without flattening a deeply nested config into hundreds of params."""
    tags = {
        "task_name": str(cfg.task_name),
        "target": ctx.target,
        "endpoint_type": setup.objective.value,
        "model_name": setup.model_name,
        "model_hash": setup.model_hash,
    }
    cv = parse_cv_split(ctx.split_version)  # (field, outer, inner) | None for non-CV splits
    if cv is not None:
        tags["outer_fold"], tags["inner_fold"] = str(cv[1]), str(cv[2])
    mlflow_logger = build_mlflow_logger(
        mlflow_dir=cfg.paths.mlflow_dir,
        experiment_name=f"{ctx.study_name}/{ctx.experiment_name}",
        run_name=f"{ctx.cohort_name}/{ctx.split_version}",
        tags=tags,
    )
    # Curated params (clean names, consistent across experiments so mlflow's Compare columns line up):
    # the knobs we group/scatter results by. Absent knobs get a neutral default so the column exists for
    # every experiment. Ordered by how central they are — the arm-specific ones come last.
    mlflow_logger.log_hyperparams(
        {
            "arm": ctx.experiment_name,
            "lr": OmegaConf.select(cfg, "lit_module.optimizer.lr"),
            "max_epochs": cfg.trainer.max_epochs,
            "modality_dropout": OmegaConf.select(cfg, "lit_module.model.modality_dropout") or 0,
            "gene_panel": _gene_panel(cfg),
        }
    )
    # Full resolved config (minus filesystem paths) as one browsable artifact — provenance, no param noise.
    cfg_no_paths = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_no_paths.pop("paths", None)
    mlflow_logger.experiment.log_text(mlflow_logger.run_id, OmegaConf.to_yaml(cfg_no_paths), "config.yaml")
    return mlflow_logger


def _fit_and_test(
    cfg: DictConfig, ctx: RunContext, setup: TrainingSetup, mlflow_logger: MLFlowLogger
) -> tuple[int, str, dict]:
    """Fit (unless random_baseline), load + test the best-by-monitor checkpoint, and log the held-out
    test metrics to mlflow at step 0. Returns ``(final_epoch, best_ckpt_path, test_metrics)``."""
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        logger=mlflow_logger,
        callbacks=list(hydra.utils.instantiate(cfg.callbacks).values()),
    )
    lit_module, datamodule = setup.lit_module, setup.datamodule
    # random_baseline: skip fitting entirely and test the untrained, randomly-initialised weights.
    # That is a null model. For the expression endpoint, a gene counts as predicted only when the
    # trained model beats this baseline's correlation on it.
    if ctx.random_baseline:
        logger.info("random_baseline=true -> testing untrained (random-init) weights (no fit).")
        best_ckpt_path = ""
    else:
        trainer.fit(lit_module, datamodule)
        # Load the best-by-monitor checkpoint and test those weights, so what is tested and persisted is
        # never ambiguous. weights_only=False: the ckpt embeds the functools.partial optimizer hparam,
        # which torch's safe-load would reject.
        best_ckpt_path = getattr(trainer.checkpoint_callback, "best_model_path", "") or ""
        if best_ckpt_path and Path(best_ckpt_path).exists():
            state = torch.load(best_ckpt_path, weights_only=False, map_location="cpu")
            lit_module.load_state_dict(state["state_dict"])
            logger.info(f"Loaded best checkpoint for test/predict: {best_ckpt_path}")
        else:
            logger.warning("No best checkpoint found; testing last-epoch weights.")

    trainer.test(lit_module, datamodule)

    if ctx.random_baseline:  # persist the (now trainer-connected) random-init weights as the run artifact
        best_ckpt_path = str(ctx.output_dir / "checkpoints" / "random.ckpt")
        Path(best_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(best_ckpt_path)

    # The lit module logs test metrics with logger=False to keep them off the step axis. That empties
    # trainer.test()'s return (it only carries logger=True metrics), so read them from callback_metrics.
    # These are streamed per slide. Logging happens in main() instead of here, because the patient-level
    # counterpart cannot exist until predictions have been collected — see `_patient_test_metric`.
    test_metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if str(k).startswith("test/")}
    return int(trainer.current_epoch), best_ckpt_path, test_metrics


def _write_predictions(ctx: RunContext, setup: TrainingSetup, tiling: dict) -> PredictionArtifacts:
    """Own pass over the test loader -> per-slide predictions, de-standardised back to interpretable units
    (scalar (μ,σ) for regression, per-gene (μ,σ) for expression z-space -> log1p; identity for
    classification). Writes the lossless NPZ always + a human CSV for scalar endpoints."""
    task, objective = setup.task, setup.objective
    if objective == Objective.regression_vector:
        pred_mean, pred_std = task.gene_means.numpy(), task.gene_stds.numpy()
    else:
        pred_mean, pred_std = getattr(task, "target_mean", 0.0), getattr(task, "target_std", 1.0)
    test_preds = collect_predictions(
        setup.lit_module,
        setup.datamodule.test_dataloader(),
        endpoint_type=objective,
        num_classes=task.num_classes,
        target_key=ctx.target_key,
        target_mean=pred_mean,
        target_std=pred_std,
        time_edges=getattr(task, "time_edges", None),
    )
    npz_path = ctx.output_dir / "test_predictions.npz"
    write_predictions_npz(test_preds, npz_path)
    # Per-tile attention logits; only models with an attention branch produce them.
    attention_path = None
    if test_preds.get("attention"):
        attention_path = write_attention_npz(test_preds["attention"], ctx.output_dir / "attention.npz", tiling=tiling)
    # CSV is human-readable only (the NPZ is the lossless artifact aggregate reads). Written for the
    # endpoints whose per-slide output fits a handful of named columns — multiclass (per-class columns
    # deferred) and expression (a 20k-gene vector) intentionally have none.
    writes_csv = objective in (Objective.binary, Objective.regression, Objective.survival)
    csv_path = ctx.output_dir / "test_predictions.csv"
    if writes_csv:
        write_predictions_csv(test_preds, csv_path)
    return PredictionArtifacts(
        npz_path=npz_path,
        csv_path=csv_path,
        writes_csv=writes_csv,
        attention_path=attention_path,
        scored_patients={str(pc) for pc in test_preds["patient_codes"]},
        patient_metrics=_patient_test_metric(test_preds, objective),
    )


def _test_coverage(ctx: RunContext, setup: TrainingSetup, scored_patients: set[str]) -> dict:
    """How many of this fold's TEST patients actually got a prediction, and which did not.

    The split table is the study's statement of who should be scored. The predictions say who was.
    The two come apart when every slide a patient has falls below ``datamodule.min_tiles`` at this
    grid, or when none of them has cached features. Both are logged during setup and nowhere else,
    so without this the run dir — and therefore the aggregate — reports a silently reduced N."""
    records = DataManager(setup.database_uri).get_records_by_split(
        manifest_name=ctx.cohort_name,
        split_version=ctx.split_version,
        split_category=SplitCategory.TEST,
    )
    declared = {str(patient.patient_code) for patient in records}
    lost = sorted(declared - scored_patients)
    if lost:
        logger.warning(
            "[test] %d/%d declared patient(s) were not scored on this fold: %s",
            len(lost),
            len(declared),
            lost,
        )
    return {"declared": len(declared), "scored": len(scored_patients), "lost": lost}


def _as_floats(values) -> Optional[list]:
    """A sequence of weights as plain floats for JSON, or None when there is no weighting."""
    return None if values is None else [float(v) for v in values]


def _patient_test_metric(preds: dict, objective: Objective) -> dict:
    """The headline metric recomputed from the collected predictions, keyed ``test/<name>_patient``.

    The streamed `test/<name>` beside it is computed per slide by torchmetrics, so on a cohort where
    patients have several slides it is the same quantity weighted by scan count. This per-patient value
    is the one `aggregate` reports, scored through the same `METRICS` table."""
    if objective == Objective.regression_vector:
        return {}  # a vector target has no per-patient scalar; the expression report handles it
    metric, _n, _n_pos = score_predictions(preds, objective.value)
    return {f"test/{METRIC_KEYS[objective.value]}_patient": metric}


def _run_metadata(
    cfg: DictConfig,
    ctx: RunContext,
    setup: TrainingSetup,
    *,
    final_epoch: int,
    best_ckpt_path: str,
    test_metrics: dict,
    test_coverage: dict,
) -> dict:
    """The run's metadata.json payload: provenance + the CV grid aggregate's strict completeness check reads."""
    task, study = setup.task, setup.study
    # Survival only: this fold's discrete-time bin edges, recorded next to the other fit-derived state
    # (target_mean/std) so the run dir describes what its per-bin hazards are indexed by.
    time_edges = getattr(task, "time_edges", None)
    return {
        "experiment_name": ctx.experiment_name,
        "split_version": ctx.split_version,
        "task_name": str(cfg.task_name),
        "target": ctx.target,
        "endpoint_type": setup.objective.value,
        "num_classes": task.num_classes,
        "experiment": ctx.exp_choice,  # the chosen experiment config (recipe provenance)
        "random_baseline": ctx.random_baseline,
        "target_normalize": ctx.target_normalize,
        "target_mean": getattr(task, "target_mean", 0.0),
        "target_std": getattr(task, "target_std", 1.0),
        # Classification only, and only when weighting is on. Under `balanced` these are derived from
        # this fold's fit split, so the run has to record them or the numbers survive nowhere but a log.
        "pos_weight": getattr(task, "pos_weight", None),
        "class_weights": _as_floats(getattr(task, "class_weights", None)),
        "time_edges": None if time_edges is None else [float(v) for v in time_edges],
        "n_outer": study.splits.n_outer,  # expected CV grid -> aggregate's strict completeness check
        "n_inner": study.splits.n_inner,
        "model_name": setup.model_name,
        "model_hash": setup.model_hash,
        "git_sha": get_git_sha(os.environ.get("BUILD_WORKING_DIRECTORY", os.getcwd())),
        "train_start": ctx.train_start_iso,
        "train_end": datetime.now(timezone.utc).isoformat(),
        "final_epoch": final_epoch,
        "best_ckpt_filename": Path(best_ckpt_path).name if best_ckpt_path else None,
        "test_metrics": test_metrics,
        "test_coverage": test_coverage,  # declared vs scored TEST patients -> aggregate's coverage line
        "cli": " ".join(sys.argv),
        "hydra_output_dir": str(ctx.output_dir),
    }


if __name__ == "__main__":
    main()
