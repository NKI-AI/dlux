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

    predict study=dcis_recurrence cohort=some_unlabeled_cohort experiment_name=baseline
    predict study=... cohort=... experiment_name=baseline only_fields=[recurrence]

The sibling of ``evaluate_external``: same load-checkpoints-and-run spine, but the target cohort has a
``predict`` role (``all_predict`` split, every patient, no labels), so there is nothing to score. For
each endpoint the dev sweep trained that the cohort also declares, this grand-ensembles the outer folds'
per-patient predictions and writes ``predictions_<field>.npz`` (+ a csv, + the first fold's attention for
the viewer). Whether a prediction is any good is unknowable here. ``evaluate_external`` answers that on
a labeled cohort.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from dlux.config.cohort import Cohort, Role, Study
from dlux.data import bulk_rna_matrix, layout
from dlux.data.compile import compile_data_description
from dlux.data.splits import format_all_predict, parse_cv_split
from dlux.eval._common import _roll_preds_to_patients
from dlux.eval.attention import write_attention_npz
from dlux.eval.external import uses_vector_probs
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
    """The config the run recorded (site paths swapped for now's), so we reconstruct the model that was
    actually trained rather than whatever the current defaults compose to. Same contract as external eval."""
    run_cfg = OmegaConf.load(fold_dir / "hparams.yaml")
    run_cfg.paths = paths
    task_cfg = run_cfg.get("task") or {}
    if not task_cfg.get("_target_") or not task_cfg.get("inputs"):
        sys.exit(
            f"[predict] {fold_dir / 'hparams.yaml'} carries no task _target_ / inputs, so the task cannot be "
            f"reconstructed. Re-run that sweep — predict rebuilds the task from what the run recorded and will not guess."
        )
    return run_cfg


def _fold_target_stats(fold_dir: Path) -> tuple[float, float]:
    """The fit-split target μ/σ a z-scored regression head trained with; (0, 1) identity otherwise."""
    meta = json.loads((fold_dir / "metadata.json").read_text())
    if str(meta.get("target_normalize", "none")) != "zscore":
        return 0.0, 1.0
    if meta.get("target_mean") is None or meta.get("target_std") is None:
        sys.exit(
            f"[predict] {fold_dir / 'metadata.json'} says target_normalize=zscore but records no "
            f"target_mean/target_std, so its predictions cannot be returned to raw units. Re-run that fold."
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


def _predict_field(
    field: str,
    cfg: DictConfig,
    cohort: Cohort,
    db_uri: str,
    tiling: dict,
    device: torch.device,
    folds: list[tuple[int, int, Path]],
    run_cfg: DictConfig,
) -> dict:
    """Grand-ensemble every dev fold's per-patient prediction on the cohort's ``all_predict`` split.

    Everything about the model (task, inputs, extractor, architecture, per-fold fit-derived stream state)
    is reconstructed from the run's record, as external eval does. Only the cohort and the split
    (all_predict, no labels) differ. Returns a dict ready for ``_write_predictions``.
    """
    pred_field = cohort.contract[field]
    pdd = compile_data_description(cohort, tiling, db_uri, split_version=format_all_predict())

    fold_state = {(o, i): read_stream_state(ckpt.parent) for o, i, ckpt in folds}
    # A fold that recorded nothing would make the per-fold state swap below silently reuse the previous
    # fold's statistics — so demand all-or-nothing, exactly as external eval does.
    recorded = {(o, i): set(state) for (o, i), state in fold_state.items()}
    every_key: set[str] = set().union(*recorded.values()) if recorded else set()
    short = {f"o{o}_i{i}": sorted(every_key - keys) for (o, i), keys in sorted(recorded.items()) if keys != every_key}
    if short:
        raise ValueError(
            f"[{field}] the development sweep records fit-derived stream state inconsistently: "
            f"{len(short)}/{len(recorded)} fold(s) are missing state that others recorded (e.g. "
            f"{dict(list(short.items())[:4])}). Re-run those folds — a fold with no state reuses another's."
        )

    def _no_fit_split():
        raise ValueError(
            f"input for '{field}' needs fit-split statistics the run did not record, and a predict cohort "
            f"has no fit split to refit them from. Re-run that sweep so it records its stream state."
        )

    first = fold_state[(folds[0][0], folds[0][1])]
    inputs = build_inputs(
        run_cfg,
        ModalityContext(
            cohort_name=cohort.name,
            cohorts_dir=Path(cfg.paths.cohorts_dir),
            data_description=pdd,
            feature_extractor=run_cfg.feature_extractor,
            fit_patient_codes=_no_fit_split,
            recorded_state=first.get,
        ),
    )
    task_cls = get_class(str(run_cfg.task._target_))
    accepted = set(inspect.signature(task_cls.__init__).parameters)
    structural = {"self", "target", "contract_field", "data_description", "inputs"}
    extras = {k: v for k, v in run_cfg.task.items() if k in accepted - structural}
    extras.update(
        resolve_replay_extras(
            pred_field.objective,
            first,
            matrix_path=bulk_rna_matrix.matrix_path(cfg.paths.cohorts_dir, cohort.name),
            gene_panel=str(run_cfg.task.target.get("gene_panel", "full")),
        )
    )
    task = task_cls(
        target=field,
        contract_field=pred_field,
        data_description=pdd,
        inputs=inputs,
        **extras,
    )
    task.register_adapter_augmentations(hydra.utils.instantiate(cfg.augmentations))
    lit_module = build_lit_module(run_cfg, task).to(device)

    datamodule = MultiModalDataModule(
        data_description=pdd,
        task=task,
        tile_transform=TileTransformFactory.for_mil_classification(),
        batch_size=cfg.datamodule.batch_size,
        eval_batch_size=cfg.datamodule.eval_batch_size,
        num_workers=cfg.datamodule.num_workers,
        min_tiles=cfg.datamodule.min_tiles,
    )
    datamodule.setup("predict")  # constructs _ds_predict over the all_predict (PREDICT-category) split
    predict_loader = datamodule.predict_dataloader()

    objective = pred_field.objective
    vector = uses_vector_probs(objective)
    per_model: dict[tuple[int, int], dict[str, object]] = {}
    attention: list | None = None
    for idx, (outer, inner, ckpt) in enumerate(folds):
        state = torch.load(ckpt, weights_only=False, map_location=device)
        lit_module.load_state_dict(state["state_dict"])
        apply_stream_state(task, fold_state[(outer, inner)])
        mean, std = _fold_target_stats(ckpt.parent)
        preds = collect_predictions(
            lit_module,
            predict_loader,
            endpoint_type=objective,
            num_classes=task.num_classes,
            target_key=f"patient.{field}",
            target_mean=mean,
            target_std=std,
            require_labels=False,
        )
        per_model[(outer, inner)] = _roll_preds_to_patients(
            np.asarray(preds["patient_codes"]), preds["probs"], vector=vector
        )
        if idx == 0 and preds["attention"]:  # one representative attention map (the first fold's model)
            attention = preds["attention"]
        logger.info("  [%s] o%d_i%d: %d patients", field, outer, inner, len(per_model[(outer, inner)]))

    # Grand-ensemble: mean of the per-fold patient predictions. Every fold sees every patient (the split
    # is label-agnostic), so the patient set is fold-independent; guard anyway.
    patients = sorted(set().union(*(set(m) for m in per_model.values())))
    ensemble: dict[str, object] = {}
    for pc in patients:
        vals = [per_model[k][pc] for k in per_model if pc in per_model[k]]
        ensemble[pc] = np.mean(vals, axis=0) if vector else float(np.mean(vals))
    return {
        "field": field,
        "objective": objective,
        "num_classes": task.num_classes,
        "patients": patients,
        "ensemble": ensemble,
        "vector": vector,
        "attention": attention,
    }


def _write_predictions(out_dir: Path, result: dict, tiling: dict) -> None:
    """``predictions_<field>.npz`` (+ csv, + attention) — the ensembled per-patient prediction, no labels."""
    out_dir.mkdir(parents=True, exist_ok=True)
    field = result["field"]
    patients = result["patients"]
    codes = np.asarray(patients)
    if result["vector"]:
        preds = np.stack([np.asarray(result["ensemble"][p], dtype=np.float64) for p in patients])
    else:
        preds = np.asarray([result["ensemble"][p] for p in patients], dtype=np.float64)
    np.savez_compressed(
        out_dir / f"predictions_{field}.npz",
        patient_codes=codes,
        predictions=preds,
        objective=result["objective"].value,
        num_classes=int(result["num_classes"]),
    )
    # A flat csv for eyeballing: one column for a scalar prediction, pred_0..pred_{K-1} for a vector.
    if result["vector"]:
        header = "patient_code," + ",".join(f"pred_{k}" for k in range(preds.shape[1]))
        rows = [f"{p}," + ",".join(f"{v:.6g}" for v in preds[i]) for i, p in enumerate(patients)]
    else:
        header = "patient_code,prediction"
        rows = [f"{p},{preds[i]:.6g}" for i, p in enumerate(patients)]
    (out_dir / f"predictions_{field}.csv").write_text("\n".join([header, *rows]) + "\n")
    if result["attention"]:
        write_attention_npz(result["attention"], out_dir / f"attention_{field}.npz", tiling=tiling)


@hydra.main(version_base=None, config_path="../config", config_name="predict")
def main(cfg: DictConfig) -> None:
    print_config(
        cfg,
        fields=("study", "cohort", "tiling", "augmentations", "paths", "datamodule", "experiment_name"),
    )
    study = Study(**OmegaConf.to_container(cfg.study, resolve=True))  # type: ignore[arg-type]
    cohort = Cohort(**OmegaConf.to_container(cfg.cohort, resolve=True))  # type: ignore[arg-type]
    experiment_name = str(cfg.experiment_name)
    only = cfg.get("only_fields")
    only_fields = [str(f) for f in only] if only else []

    role = study.cohorts.get(cohort.name)
    if role != Role.predict:
        sys.exit(
            f"[predict] cohort '{cohort.name}' is not a 'predict' cohort of study '{study.name}' "
            f"(role={role.value if role else None}). Give it `role: predict` (all_predict, unlabeled)."
        )
    dev_cohorts = sorted(c for c, r in study.cohorts.items() if r == Role.development)
    if len(dev_cohorts) != 1:
        sys.exit(f"[predict] expected exactly one development cohort in '{study.name}', got {dev_cohorts}.")
    dev_cohort = dev_cohorts[0]

    exp_dir = Path(cfg.paths.studies_dir) / study.name / "runs" / experiment_name / dev_cohort
    if not exp_dir.is_dir():
        sys.exit(f"[predict] development runs dir not found: {exp_dir} (train the sweep first).")
    dev_folds = _discover_dev_folds(exp_dir)
    if not dev_folds:
        sys.exit(f"[predict] no completed folds (best.ckpt + metadata.json) under {exp_dir}.")

    # Which endpoints to run: declared by the study, trained by the sweep, and present in the cohort's
    # contract (it must declare the target's type/space even though it carries no values). No handler
    # filter — inference needs no metric, so every objective the task supports is predictable.
    fields: list[str] = []
    for f in study.targets:
        if f not in dev_folds:
            logger.warning("skipping '%s': the development sweep has no completed folds for it.", f)
            continue
        if f not in cohort.contract:
            logger.warning("skipping '%s': not in predict cohort '%s' contract.", f, cohort.name)
            continue
        fields.append(f)
    fields.sort()
    if only_fields:
        unknown = [f for f in only_fields if f not in fields]
        if unknown:
            sys.exit(f"[predict] only_fields={unknown} are not predictable endpoints (available: {fields}).")
        fields = [f for f in fields if f in only_fields]
    if not fields:
        sys.exit(f"[predict] no endpoints shared by the dev sweep and cohort '{cohort.name}'.")

    db_uri = layout.db_uri(cfg.paths.studies_dir, study.name, cohort.name)
    tiling = OmegaConf.to_container(cfg.tiling, resolve=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.paths.studies_dir) / study.name / "runs" / experiment_name / cohort.name / "predict"

    logger.info(
        "Predicting %d endpoint(s) %s on '%s' (dev cohort '%s', experiment '%s').",
        len(fields),
        fields,
        cohort.name,
        dev_cohort,
        experiment_name,
    )
    for field in fields:
        run_cfg = _load_run_cfg(dev_folds[field][0][2].parent, cfg.paths)  # folds of a field share the recipe
        result = _predict_field(field, cfg, cohort, db_uri, tiling, device, dev_folds[field], run_cfg)
        _write_predictions(out_dir, result, tiling)
        logger.info("[%s] wrote predictions for %d patients -> %s", field, len(result["patients"]), out_dir)


if __name__ == "__main__":
    main()
