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
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from dlux.config.cohort import Cohort, Objective, Role, Study
from dlux.data import bulk_rna_matrix, layout
from dlux.data.compile import compile_data_description
from dlux.data.splits import format_all_test, parse_cv_split
from dlux.eval._common import _roll_external
from dlux.eval.external import (
    ExternalResult,
    cast_scalar_label,
    external_ensemble,
    headline,
    supported_objectives,
    uses_vector_labels,
    uses_vector_probs,
    write_external_report,
)
from dlux.eval.gene_metrics import conservative_significant_gene_count
from dlux.eval.predictions import collect_predictions
from dlux.modalities.context import ModalityContext
from dlux.modalities.state import apply_stream_state, read_stream_state
from dlux.models import build_lit_module
from dlux.tasks.build import build_inputs, resolve_replay_extras
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

from ahcore.data.mm_dataset import MultiModalDataModule
from ahcore.transforms.tile import TileTransformFactory
from ahcore.utils.io import get_logger, print_config

logger = get_logger(__name__)


def _load_run_cfg(fold_dir: Path, paths: DictConfig) -> DictConfig:
    """The config the run recorded, not the one we happen to be invoked with.

    A run persists its fully-resolved config. Reconstructing from it scores the model that was actually
    trained instead of whatever the current defaults compose to."""
    run_cfg = OmegaConf.load(fold_dir / "hparams.yaml")
    # A run's config records two different kinds of thing: what it did (task, model, extractor identity)
    # and where that machine kept its files. Only the first is run identity — site paths must come from
    # now, or interpolations like ${paths.models_dir} resolve to wherever the sweep happened to run.
    run_cfg.paths = paths
    task_cfg = run_cfg.get("task") or {}
    if not task_cfg.get("_target_") or not task_cfg.get("inputs"):
        sys.exit(
            f"[evaluate_external] {fold_dir / 'hparams.yaml'} carries no task _target_ / inputs, so the "
            f"task cannot be reconstructed. Re-run that sweep — external eval rebuilds the task from what "
            f"the run recorded and will not guess."
        )
    return run_cfg


def _fold_target_stats(fold_dir: Path) -> tuple[float, float]:
    """The fit-split target μ/σ this fold trained with, read from its persisted record.

    Only a z-scored regression head needs them. Every other endpoint predicts in raw units, for which
    (0, 1) is the identity. A run that recorded ``target_normalize: zscore`` without the statistics
    cannot be de-standardised, and guessing would silently report z-space predictions against raw
    labels — so that is an error, not a fallback."""
    meta = json.loads((fold_dir / "metadata.json").read_text())
    if str(meta.get("target_normalize", "none")) != "zscore":
        return 0.0, 1.0
    if meta.get("target_mean") is None or meta.get("target_std") is None:
        sys.exit(
            f"[evaluate_external] {fold_dir / 'metadata.json'} says target_normalize=zscore but records "
            f"no target_mean/target_std, so its predictions cannot be returned to raw units. Re-run that "
            f"fold — external eval will not guess the fit-split statistics."
        )
    return float(meta["target_mean"]), float(meta["target_std"])


def _discover_dev_folds(exp_dir: Path) -> dict[str, list[tuple[int, int, Path]]]:
    """Map each endpoint the dev sweep trained -> its completed folds ``[(outer, inner, best.ckpt)]``."""
    by_field: dict[str, list[tuple[int, int, Path]]] = {}
    for d in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
        parsed = parse_cv_split(d.name)
        if parsed is None:
            continue
        if not (d / "metadata.json").exists() or not (d / "best.ckpt").exists():
            continue
        field, outer, inner = parsed
        by_field.setdefault(field, []).append((outer, inner, d / "best.ckpt"))
    return by_field


def _assert_stream_state_consistent(field: str, fold_state: dict) -> None:
    """A fold that recorded nothing corrupts scoring. ``apply_stream_state`` with an empty payload is a
    no-op, so the per-fold swap leaves the previous fold's statistics installed and scores this fold's
    model against another fold's standardisation, silently. Either every fold records a stream's state or
    none does. Anything between is inconsistent and must be re-run."""
    recorded = {(o, i): set(state) for (o, i), state in fold_state.items()}
    every_key: set[str] = set().union(*recorded.values()) if recorded else set()
    short = {f"o{o}_i{i}": sorted(every_key - keys) for (o, i), keys in sorted(recorded.items()) if keys != every_key}
    if short:
        raise ValueError(
            f"[{field}] the sweep records fit-derived stream state inconsistently: "
            f"{len(short)}/{len(recorded)} fold(s) are missing state that others recorded "
            f"(e.g. {dict(list(short.items())[:4])}). Scoring cannot proceed, because a fold with no "
            f"recorded state would silently reuse the previous fold's. Re-run those folds."
        )


def _score_field(
    field: str,
    cfg: DictConfig,
    val_cohort: Cohort,
    val_db_uri: str,
    tiling: dict,
    device: torch.device,
    folds: list[tuple[int, int, Path]],
    run_cfg: DictConfig,
    random_folds: list[tuple[int, int, Path]] | None = None,
) -> ExternalResult:
    """Score one endpoint on the validation cohort's ``all_test_<field>`` split with each dev model.

    ``random_folds`` (expression only) is a random-baseline sweep's folds: when given, the trained model's
    per-gene correlation is compared against that random-init model's on the same external patients
    (SEQUOIA's conservative significant-gene count). The random arm reuses this field's built model and
    dataloader — only its checkpoints and recorded per-fold state differ."""
    val_field = val_cohort.contract[field]
    target_key = f"patient.{field}"
    val_dd = compile_data_description(val_cohort, tiling, val_db_uri, split_version=format_all_test(field))

    # The task, input streams, feature extractor, and model architecture all come from the run's config.
    # Only the cohort (and so the data description and label map) is swapped to the validation one.
    # Fit-derived stream state is replayed from what each run recorded, never refitted here: refitting on
    # the scored cohort would change the input distribution the model was trained for.
    # Construction takes the first fold's record, which fixes the fold-independent parts (gene ids, panel,
    # vector width) and validates them against this cohort's data. The per-fold statistics are swapped in
    # below, before each fold's inference, including the first's.
    fold_state = {(o, i): read_stream_state(ckpt.parent) for o, i, ckpt in folds}
    _assert_stream_state_consistent(field, fold_state)

    def _no_fit_split():
        raise ValueError(
            f"input for '{field}' needs fit-split statistics that the run did not record, and an external "
            f"cohort has no fit split to refit them from (it is scored end-to-end on all_test). Re-run "
            f"that sweep so it records its stream state."
        )

    inputs = build_inputs(
        run_cfg,
        ModalityContext(
            cohort_name=val_cohort.name,
            cohorts_dir=Path(cfg.paths.cohorts_dir),
            data_description=val_dd,
            feature_extractor=run_cfg.feature_extractor,
            fit_patient_codes=_no_fit_split,
            recorded_state=fold_state[(folds[0][0], folds[0][1])].get,
        ),
    )
    # Reconstruct the task from what the run recorded. Beyond the four structural arguments, a task may
    # declare its own config scalars (survival's `n_bins`). Pass through exactly those the class accepts,
    # so this stays generic instead of branching on the task type. Fit-split-derived arguments are not
    # passed — an external cohort has no fit split, and a task that needs such state to score (rather
    # than only to train) will say so when it is used.
    task_cls = get_class(str(run_cfg.task._target_))
    accepted = set(inspect.signature(task_cls.__init__).parameters)
    structural = {"self", "target", "contract_field", "data_description", "inputs"}
    extras = {k: v for k, v in run_cfg.task.items() if k in accepted - structural}
    # Target-side fit state, replayed rather than refitted. The input streams get theirs through the
    # ModalityContext above. A task whose target carries such state (expression's per-gene μ/σ) has no
    # such route, because it builds that stream itself. Construction takes the first fold's record —
    # it fixes the fold-independent parts (gene ids, panel, vector width) and validates them against
    # this cohort's matrix. The per-fold statistics are swapped in below, before each fold's inference.
    extras.update(
        resolve_replay_extras(
            val_field.objective,
            fold_state[(folds[0][0], folds[0][1])],
            matrix_path=bulk_rna_matrix.matrix_path(cfg.paths.cohorts_dir, val_cohort.name),
            gene_panel=str(run_cfg.task.target.get("gene_panel", "full")),  # a read parameter of the target
        )
    )
    task = task_cls(
        target=field,
        contract_field=val_field,  # the validation cohort's map (its raw labels -> the shared {0,1} space)
        data_description=val_dd,
        inputs=inputs,
        **extras,
    )
    task.register_adapter_augmentations(hydra.utils.instantiate(cfg.augmentations))
    lit_module = build_lit_module(run_cfg, task).to(device)

    datamodule = MultiModalDataModule(
        data_description=val_dd,
        task=task,
        tile_transform=TileTransformFactory.for_mil_classification(),
        batch_size=cfg.datamodule.batch_size,
        eval_batch_size=cfg.datamodule.eval_batch_size,
        num_workers=cfg.datamodule.num_workers,
        min_tiles=cfg.datamodule.min_tiles,
    )
    datamodule.setup("test")  # constructs _ds_test (no Trainer here to do it for us)
    test_loader = datamodule.test_dataloader()  # all_test_<field> -> only valid-label patients

    objective = val_field.objective  # Objective enum; eval.external.supported_objectives() is the authority
    vec_p, vec_l = uses_vector_probs(objective), uses_vector_labels(objective)

    def _collect_arm(arm_folds, arm_state, tag):
        """Every fold of one arm through the (shared) model + dataloader -> per-model patient probs.

        Reused for the trained arm and, when scoring the expression null, the random-baseline arm — the
        model, task and test_loader are built once above; only the checkpoint weights and the swapped-in
        per-fold stream statistics differ. Returns (per_model, labels); labels are identical across folds."""
        _assert_stream_state_consistent(field, arm_state)
        per_model: dict[tuple[int, int], dict[str, object]] = {}
        labels: dict[str, object] = {}
        for outer, inner, ckpt in arm_folds:
            state = torch.load(ckpt, weights_only=False, map_location=device)
            lit_module.load_state_dict(state["state_dict"])
            # This fold's fit-derived stream statistics, so the model is fed inputs standardised the way it
            # was trained. Only the statistics move. The dataset and its adapters are fold-independent.
            apply_stream_state(task, arm_state[(outer, inner)])
            # A z-scored regression head trains in standardised space, so map its predictions back to raw
            # units with the same fit-split statistics that fold trained with (per-fold, from metadata.json).
            mean, std = _fold_target_stats(ckpt.parent)
            preds = collect_predictions(
                lit_module,
                test_loader,
                endpoint_type=objective,
                num_classes=task.num_classes,
                target_key=target_key,
                target_mean=mean,
                target_std=std,
            )
            # Prediction shape and label shape are independent: multiclass has (N, K) probs with a scalar
            # class label, survival a scalar risk with a coupled (time, event) label. The handler owns both.
            patient_probs, patient_label = _roll_external(
                preds["patient_codes"], preds["probs"], preds["labels"], vector_preds=vec_p, vector_labels=vec_l
            )
            if not vec_l:  # a scalar label is a class index or a raw value
                patient_label = {pc: cast_scalar_label(objective, v) for pc, v in patient_label.items()}
            per_model[(outer, inner)] = patient_probs
            labels = patient_label  # identical across models (same external patients)
            logger.info("  [%s] %s o%d_i%d: %d patients", field, tag, outer, inner, len(patient_probs))
        return per_model, labels

    per_model, labels = _collect_arm(folds, fold_state, "scored")
    result = external_ensemble(
        field, val_cohort.name, per_model, labels, endpoint_type=objective, num_classes=task.num_classes
    )

    # Expression null: the trained model must beat a random-init model of the same architecture per gene,
    # not merely beat zero correlation (SEQUOIA). Both arms see every external patient, so it is a paired
    # comparison on identical patients. Williams has df = n-3, so
    # on a small external cohort the count is underpowered (says "too few patients", not "does not transfer").
    if random_folds and objective == Objective.regression_vector:
        random_state = {(o, i): read_stream_state(ckpt.parent) for o, i, ckpt in random_folds}
        random_per_model, _ = _collect_arm(random_folds, random_state, "random")
        pats = sorted(result.labels)
        random_grand = {
            p: np.mean(np.stack([np.asarray(random_per_model[m][p]) for m in random_per_model]), axis=0) for p in pats
        }
        trained_mat = np.stack([np.asarray(result.grand_patient_probs[p]) for p in pats])
        random_mat = np.stack([random_grand[p] for p in pats])
        label_mat = np.stack([np.asarray(result.labels[p]) for p in pats])
        result.conservative = conservative_significant_gene_count(trained_mat, random_mat, label_mat)
        logger.info(
            "  [%s] significant genes (trained vs random-init): %d/%d",
            field,
            result.conservative["n_significant"],
            result.conservative["n_scored"],
        )
    return result


@hydra.main(version_base=None, config_path="../config", config_name="evaluate_external")
def main(cfg: DictConfig) -> None:
    print_config(
        cfg,
        fields=(
            "study",
            "cohort",
            "tiling",
            "augmentations",
            "paths",
            "datamodule",
            "experiment_name",
        ),
    )
    study = Study(**OmegaConf.to_container(cfg.study, resolve=True))  # type: ignore[arg-type]
    val_cohort = Cohort(**OmegaConf.to_container(cfg.cohort, resolve=True))  # type: ignore[arg-type]
    experiment_name = str(cfg.experiment_name)
    only = cfg.get("only_fields")
    only_fields = [str(f) for f in only] if only else []  # empty -> score every scorable endpoint

    # The cohort must be a `validation` cohort of this study (development supplies the models).
    role = study.cohorts.get(val_cohort.name)
    if role != Role.validation:
        sys.exit(
            f"[evaluate_external] cohort '{val_cohort.name}' is not a 'validation' cohort of study "
            f"'{study.name}' (role={role.value if role else None}). Development cohorts supply the models."
        )
    dev_cohorts = sorted(c for c, r in study.cohorts.items() if r == Role.development)
    if len(dev_cohorts) != 1:
        sys.exit(f"[evaluate_external] expected exactly one development cohort in '{study.name}', got {dev_cohorts}.")
    dev_cohort = dev_cohorts[0]

    exp_dir = Path(cfg.paths.studies_dir) / study.name / "runs" / experiment_name / dev_cohort
    if not exp_dir.is_dir():
        sys.exit(f"[evaluate_external] development runs dir not found: {exp_dir} (train the sweep first).")
    dev_folds = _discover_dev_folds(exp_dir)
    if not dev_folds:
        sys.exit(f"[evaluate_external] no completed folds (best.ckpt + metadata.json) under {exp_dir}.")

    # The study declares which endpoints exist. The handler registry declares which can be reported on.
    # Neither is inferred from what happens to be on disk.
    scorable = supported_objectives()
    fields: list[str] = []
    for f in study.targets:
        if f not in dev_folds:
            logger.warning("skipping '%s': the development sweep has no completed folds for it.", f)
            continue
        if f not in val_cohort.contract:
            logger.warning("skipping '%s': not in validation cohort '%s' contract.", f, val_cohort.name)
            continue
        objective = val_cohort.contract[f].objective
        if objective not in scorable:
            logger.warning(
                "skipping '%s': no external handler for objective '%s' (have %s).", f, objective.value, scorable
            )
            continue
        fields.append(f)
    fields.sort()  # declared set, sorted order — which endpoints are scored is the study's call, the
    # order they appear in the report is not something this bin should change.
    if only_fields:
        unknown = [f for f in only_fields if f not in fields]
        if unknown:
            sys.exit(f"[evaluate_external] only_fields={unknown} are not scorable endpoints (scorable: {fields}).")
        fields = [f for f in fields if f in only_fields]
    if not fields:
        sys.exit(f"[evaluate_external] no scorable endpoints shared by the dev sweep and '{val_cohort.name}'.")

    val_db_uri = layout.db_uri(cfg.paths.studies_dir, study.name, val_cohort.name)
    tiling = OmegaConf.to_container(cfg.tiling, resolve=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optional random-baseline arm for the expression significance null — a sweep trained with
    # `random_baseline: true` (untrained model), mirroring aggregate's `random_experiment_dir`. Only the
    # expression endpoint consumes it (other objectives have a fixed chance floor at 0.5). _score_field
    # ignores it for those. Discovered by field, same as the trained arm.
    random_experiment = cfg.get("random_experiment")
    random_folds: dict[str, list[tuple[int, int, Path]]] = {}
    if random_experiment:
        rand_dir = Path(cfg.paths.studies_dir) / study.name / "runs" / str(random_experiment) / dev_cohort
        if not rand_dir.is_dir():
            sys.exit(f"[evaluate_external] random_experiment runs dir not found: {rand_dir}.")
        random_folds = _discover_dev_folds(rand_dir)
        if not random_folds:
            sys.exit(f"[evaluate_external] no completed folds under random_experiment dir {rand_dir}.")

    logger.info(
        "Scoring %d endpoint(s) %s on '%s' (dev cohort '%s', experiment '%s').",
        len(fields),
        fields,
        val_cohort.name,
        dev_cohort,
        experiment_name,
    )

    results: list[ExternalResult] = []
    for field in fields:
        run_cfg = _load_run_cfg(dev_folds[field][0][2].parent, cfg.paths)  # folds of a field share the recipe
        res = _score_field(
            field,
            cfg,
            val_cohort,
            val_db_uri,
            tiling,
            device,
            dev_folds[field],
            run_cfg,
            random_folds=random_folds.get(field),
        )
        results.append(res)
        logger.info("%s", headline(res))

    out_dir = Path(cfg.paths.studies_dir) / study.name / "results" / experiment_name / val_cohort.name
    write_external_report(results, experiment_name, out_dir)
    logger.info("Wrote external report (%d endpoint(s)) -> %s", len(results), out_dir)


if __name__ == "__main__":
    main()
