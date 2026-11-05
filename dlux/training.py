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
"""The steps of a training run: resolve which fold this is, build the task, model and data, open the mlflow run,
fit and test the best checkpoint, write the per-slide predictions, and describe the run for its metadata.

``bin/train.py`` runs them in order. ``bin/train_resplit.py`` shares ``load_study_and_cohort``, ``build_setup``
and ``fit_and_test``. ``RunContext`` (the run's identity) and ``TrainingSetup`` (task, model, data) carry
state between them. What differs per sample unit is in ``dlux.sample_units``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger, MLFlowLogger
from lightning.pytorch.trainer import Trainer
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from ahcore.manifest import DataManager
from ahcore.utils.io import get_logger

from dlux.callbacks import FailOnNonFiniteLoss, has_progress_bar, select_callbacks
from dlux.config.cohort import Cohort, Objective, SplitCategory, Study
from dlux.data import layout
from dlux.data.splits import format_cv_split, parse_cv_split
from dlux.eval.attention import write_attention_npz
from dlux.eval.compare import default_metric_key, score_predictions
from dlux.eval.persistence import claim_hydra_run_dir, get_git_sha, persistent_run_dir, run_already_persisted
from dlux.eval.predictions import write_predictions_csv, write_predictions_npz, writes_csv
from dlux.models import build_lit_module
from dlux.sample_units import build_datamodule, encoder_identity, slide_predictions, unit_callbacks
from dlux.tasks.build import build_task
from dlux.tracking import build_mlflow_logger

logger = get_logger(__name__)


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
    target_normalize: Optional[str]
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


def resolve_run_context(cfg: DictConfig) -> Optional[RunContext]:
    """Resolve the run identity, validate the experiment choice, and claim the disposable hydra dir.
    Returns None (caller exits 0) when the fold is already persisted and not forced — a clean resumable
    skip, not an error."""
    output_dir = Path(HydraConfig.get().runtime.output_dir)  # disposable
    study_name, cohort_name = str(cfg.study.name), str(cfg.cohort.name)
    experiment_name = str(cfg.experiment_name)
    exp_choice = str(HydraConfig.get().runtime.choices.get("experiment", ""))
    check_experiment_choice(exp_choice, experiment_name, study_name)
    # Validate the invocation before the already-persisted skip: a mismatched target on a fold that
    # happens to exist must be refused, not silently reported as "already done".
    target = str(cfg.task.target.field)
    split_version = _resolve_split_version(cfg, target)
    check_target_choice(target, split_version, cfg.study.get("targets") or [])

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
        # Provenance for metadata.json. Only the regression task declares it.
        target_normalize=OmegaConf.select(cfg, "task.target_normalize"),
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
            f"(0..{n_outer * n_inner - 1}) — pass a fold index in that range."
        )
    outer, inner = divmod(fold, n_inner)
    return format_cv_split(target, outer, inner)


def check_target_choice(target: str, split_version: str, study_targets: Sequence[str]) -> None:
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


def check_experiment_choice(exp_choice: str, experiment_name: str, study_name: str) -> None:
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


def load_study_and_cohort(cfg: DictConfig, tool: str) -> tuple[Study, Cohort]:
    """The run's study and cohort, validated, with the cohort checked to be part of the study. Exits with a
    message prefixed ``[tool]`` on a bad config."""
    try:
        study = Study(**OmegaConf.to_container(cfg.study, resolve=True))  # type: ignore[arg-type]
        cohort = Cohort(**OmegaConf.to_container(cfg.cohort, resolve=True))  # type: ignore[arg-type]
    except ValidationError as exc:
        sys.exit(f"[{tool}] {exc}")
    if cohort.name not in study.cohorts:
        sys.exit(f"[{tool}] cohort '{cohort.name}' is not part of study '{study.name}' (has: {sorted(study.cohorts)}).")
    return study, cohort


def build_setup(
    cfg: DictConfig,
    study: Study,
    cohort: Cohort,
    target: str,
    *,
    split_version: Optional[str] = None,
    drawn: Optional[dict[str, list[str]]] = None,
    tool: str = "train",
) -> TrainingSetup:
    """Build the task, the lightning module, and the datamodule for one fold.

    The fold is either a split stored in the DB (``split_version``) or one drawn in memory (``drawn``, the
    patient codes per stage), and exactly one is given. The task holds the endpoint's fit-time setup (target
    statistics computed on the fit split only, so no validation/test leakage). Building mutates ``cfg`` with
    the runtime-resolved model widths, which then travel into the config.yaml artifact."""
    if (split_version is None) == (drawn is None):
        raise ValueError("build_setup needs exactly one of split_version or drawn.")
    database_uri = layout.db_uri(cfg.paths.studies_dir, study.name, cohort.name)

    tiling = OmegaConf.to_container(cfg.tiling, resolve=True)
    contract_field = cohort.contract[target]

    try:
        task = build_task(
            cfg,
            cohort,
            contract_field,
            tiling,
            database_uri,
            split_version,
            fit_patient_codes=drawn["fit"] if drawn is not None else None,
        )
    except ValueError as exc:
        sys.exit(f"[{tool}] {exc}")
    task.register_adapter_augmentations(hydra.utils.instantiate(cfg.augmentations))

    lit_module = build_lit_module(cfg, task)

    # A drawn split lives only in memory, so the data description carries no split version and the
    # membership travels as an explicit per-stage patient set.
    selection = {} if drawn is None else {"patient_codes_by_stage": {k: set(v) for k, v in drawn.items()}}
    datamodule = build_datamodule(task, task.data_description, cfg.datamodule, **selection)
    model_name, model_hash = encoder_identity(
        task, cfg.lit_module, str(HydraConfig.get().runtime.choices.get("lit_module", ""))
    )
    return TrainingSetup(
        task=task,
        lit_module=lit_module,
        datamodule=datamodule,
        study=study,
        objective=contract_field.objective,
        model_name=model_name,
        model_hash=model_hash,
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


def configure_mlflow(cfg: DictConfig, ctx: RunContext, setup: TrainingSetup) -> MLFlowLogger:
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
    assert isinstance(cfg_no_paths, DictConfig)
    cfg_no_paths.pop("paths", None)
    mlflow_logger.experiment.log_text(mlflow_logger.run_id, OmegaConf.to_yaml(cfg_no_paths), "config.yaml")
    return mlflow_logger


def fit_and_test(
    cfg: DictConfig,
    task: Any,
    lit_module: Any,
    datamodule: Any,
    *,
    run_logger: Logger | bool,
    random_baseline: bool,
    output_dir: Path,
) -> tuple[int, str, dict]:
    """Fit (unless ``random_baseline``), load and test the best-by-monitor checkpoint. Returns
    ``(final_epoch, best_ckpt_path, test_metrics)``, with the streamed per-slide ``test/`` metrics.

    ``run_logger`` receives the fit's metrics, ``False`` logs none. ``random_baseline`` skips the fit and
    tests the untrained, randomly initialised weights, saved to ``output_dir/checkpoints/random.ckpt``. That
    is a null model: for the expression endpoint, a gene counts as predicted only when the trained model
    beats this baseline's correlation on it."""
    # The unit's callbacks come first, so the metrics they log at the end of a validation reach LogMetrics.
    callbacks = [*unit_callbacks(task), *select_callbacks(cfg), FailOnNonFiniteLoss()]
    # Every key of the `trainer` config block is a Trainer argument. An unknown key fails in Lightning.
    trainer = Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),  # type: ignore[arg-type]
        logger=run_logger,
        callbacks=callbacks,
        enable_progress_bar=has_progress_bar(callbacks),
    )
    if random_baseline:
        logger.info("random_baseline=true -> testing untrained (random-init) weights (no fit).")
    else:
        trainer.fit(lit_module, datamodule)
        best_ckpt_path = load_best_checkpoint(trainer, lit_module)

    trainer.test(lit_module, datamodule)

    if random_baseline:  # persist the (now trainer-connected) random-init weights as the run artifact
        best_ckpt_path = str(output_dir / "checkpoints" / "random.ckpt")
        Path(best_ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(best_ckpt_path)

    # The lit module logs test metrics with logger=False to keep them off the step axis. That empties
    # trainer.test()'s return (it only carries logger=True metrics), so they are read from callback_metrics.
    # These are streamed per slide. The per-patient counterpart needs the collected predictions, so the
    # caller logs both (see `_patient_test_metric`).
    test_metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if str(k).startswith("test/")}
    return int(trainer.current_epoch), best_ckpt_path, test_metrics


def load_best_checkpoint(trainer: Trainer, lit_module: torch.nn.Module) -> str:
    """Load the fit's best checkpoint into ``lit_module`` and return its path, so what is tested and persisted
    is never ambiguous. A fit that left no best checkpoint raises.

    ``weights_only=False``: the checkpoint embeds the functools.partial optimizer hparam, which torch's safe
    load rejects."""
    callback = trainer.checkpoint_callback
    if not isinstance(callback, ModelCheckpoint):
        raise RuntimeError("The trainer has no ModelCheckpoint callback, so the fit has no best checkpoint to test.")
    best = str(callback.best_model_path)
    if not best or not Path(best).exists():
        raise RuntimeError(f"The fit left no best checkpoint (best_model_path={best!r}).")
    lit_module.load_state_dict(torch.load(best, weights_only=False, map_location="cpu")["state_dict"])
    logger.info("Loaded best checkpoint for test/predict: %s", best)
    return best


def write_predictions(
    ctx: RunContext, setup: TrainingSetup, tiling: dict, trainer_cfg: DictConfig
) -> PredictionArtifacts:
    """Own pass over the test loader -> per-slide predictions, de-standardised back to interpretable units
    (scalar (μ,σ) for regression, per-gene (μ,σ) for expression z-space -> log1p; identity for
    classification). Writes the lossless NPZ always + a human CSV for scalar endpoints."""
    task, objective = setup.task, setup.objective
    fit_state = task.fit_state()
    test_preds = slide_predictions(
        task,
        setup.lit_module,
        setup.datamodule,
        "test",
        trainer_cfg=trainer_cfg,
        endpoint_type=objective,
        target_key=ctx.target_key,
        fit_state=fit_state,
    )
    npz_path = ctx.output_dir / "test_predictions.npz"
    write_predictions_npz(test_preds, npz_path)
    # Per-tile attention logits; only models with an attention branch produce them.
    attention_path = None
    if test_preds.get("attention"):
        attention_path = write_attention_npz(test_preds["attention"], ctx.output_dir / "attention.npz", tiling=tiling)
    # CSV is human-readable only (the NPZ is the lossless artifact aggregate reads). Expression has none:
    # its per-slide output is a gene vector.
    has_csv = writes_csv(objective)
    csv_path = ctx.output_dir / "test_predictions.csv"
    if has_csv:
        write_predictions_csv(test_preds, csv_path)
    return PredictionArtifacts(
        npz_path=npz_path,
        csv_path=csv_path,
        writes_csv=has_csv,
        attention_path=attention_path,
        scored_patients={str(pc) for pc in test_preds["patient_codes"]},
        patient_metrics=_patient_test_metric(test_preds, objective),
    )


def scored_test_coverage(ctx: RunContext, setup: TrainingSetup, scored_patients: set[str]) -> dict:
    """How many of this fold's TEST patients actually got a prediction, and which did not.

    The split table is the study's statement of who should be scored. The predictions say who was.
    The two come apart when every slide a patient has falls below ``datamodule.min_tiles`` at this
    grid, or when none of them has cached features. Both are logged during setup and nowhere else,
    so without this the run dir — and therefore the aggregate — reports a silently reduced N."""
    with DataManager(setup.database_uri) as data_manager:
        records = data_manager.get_records_by_split(
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


def _patient_test_metric(preds: dict, objective: Objective) -> dict:
    """The headline metric recomputed from the collected predictions, keyed ``test/<name>_patient``.

    The streamed `test/<name>` beside it is computed per slide by torchmetrics, so on a cohort where
    patients have several slides it is the same quantity weighted by scan count. This per-patient value
    is the one `aggregate` reports, scored through the same ``compare.score_predictions``."""
    if objective == Objective.regression_vector:
        return {}  # a vector target has no per-patient scalar; the expression report handles it
    metric, _n, _n_pos = score_predictions(preds, objective.value)
    return {f"test/{default_metric_key(objective.value)}_patient": metric}


def run_metadata(
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
        # The fold's fit-derived numbers (the target scale, the class weights, the survival bin edges), so
        # the run dir describes what its predictions and per-bin hazards mean.
        **task.fit_state().recorded(),
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
