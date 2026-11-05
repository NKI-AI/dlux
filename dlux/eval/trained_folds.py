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
"""A development sweep's trained folds, rebuilt to run on a cohort they were not trained on.

A fold is rebuilt from what its run recorded: the config in ``hparams.yaml``, the fit-derived stream state,
and the fit-split target statistics and survival time bins in ``metadata.json``. Only the data description
and the scored cohort's own build-db target statistics (a binarize or discretize cut) are new. Nothing is
refitted on the data the model runs over, because that would change the input distribution the model was
trained for.

``plan`` finds the sweep and the endpoints a flow runs on a cohort. ``rebuild`` builds one task, model and
datamodule over the cohort's single split, and ``predictions`` loads each fold's weights and stream state
into them (``load``) and yields that fold's per-slide predictions. ``evaluate_external`` and ``predict``
are these steps plus their own scoring or output. A flow that runs each fold on its own split builds a
task per fold with that fold's state.

A trained fold is scored only on patients it never saw. ``assert_disjoint_from_development`` checks that for
a cohort scored with every fold of a sweep.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import lightning.pytorch as pl
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from ahcore.data.interfaces import Stage
from ahcore.manifest import DataManager
from ahcore.utils.io import get_logger

from dlux.config.cohort import Cohort, Objective, Role, SplitCategory, Study
from dlux.data import bulk_rna_matrix, layout
from dlux.data.compile import compile_data_description
from dlux.data.splits import format_all_predict, format_all_test, format_cv_split, parse_cv_split
from dlux.eval._common import ensemble_models, require_same_patients, roll_slides_to_patients
from dlux.eval.attention import write_attention_npz
from dlux.eval.external import ExternalResult, external_ensemble, label_model
from dlux.eval.gene_metrics import conservative_significant_gene_count
from dlux.modalities.context import ModalityContext
from dlux.modalities.state import apply_stream_state, read_stream_state
from dlux.models import build_lit_module
from dlux.sample_units import build_datamodule, slide_predictions
from dlux.tasks.build import build_inputs, construct_task, resolve_recorded_extras
from dlux.tasks.fit_state import FitState
from dlux.tasks.target import load_target_stats

logger = get_logger(__name__)

# A development fold: (outer, inner, path of its best.ckpt). The checkpoint's directory is the run's.
DevFold = tuple[int, int, Path]

_CV_CATEGORIES = (SplitCategory.FIT, SplitCategory.VALIDATE, SplitCategory.TEST)


def discover(exp_dir: Path) -> dict[str, list[DevFold]]:
    """Each endpoint a development sweep trained, with its completed folds. A fold is complete when its run
    directory holds both ``metadata.json`` and ``best.ckpt``."""
    by_field: dict[str, list[DevFold]] = {}
    for d in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
        parsed = parse_cv_split(d.name)
        if parsed is None:
            continue
        if not (d / "metadata.json").exists() or not (d / "best.ckpt").exists():
            continue
        field, outer, inner = parsed
        by_field.setdefault(field, []).append((outer, inner, d / "best.ckpt"))
    return by_field


def require_complete(field: str, folds: list[DevFold], n_outer: int, n_inner: int, *, strict: bool) -> None:
    """Check that ``folds`` hold every model of the ``n_outer x n_inner`` sweep before they are ensembled.
    A missing model raises, or with ``strict=False`` is logged and the ensemble uses the models present."""
    present = {(o, i) for o, i, _ in folds}
    missing = sorted((o, i) for o in range(n_outer) for i in range(n_inner) if (o, i) not in present)
    if not missing:
        return
    msg = (
        f"[{field}] {len(missing)} of {n_outer * n_inner} development models are missing "
        f"({['o%d_i%d' % m for m in missing[:12]]}{'...' if len(missing) > 12 else ''})"
    )
    if strict:
        raise ValueError(msg + ". Re-run those folds, or pass strict=false to ensemble the models present.")
    logger.warning(msg + ". Ensembling the %d models present because strict=false.", len(folds))


def load_run_cfg(run_dir: Path, paths: DictConfig) -> DictConfig:
    """The config a run recorded, with ``paths`` in place of the recorded site paths.

    The recorded config fixes what the run did (task, model, extractor identity). The site paths come from
    the caller, or interpolations like ``${paths.models_dir}`` resolve to wherever the sweep ran."""
    path = run_dir / "hparams.yaml"
    run_cfg = OmegaConf.load(path)
    if not isinstance(run_cfg, DictConfig):
        raise ValueError(f"{path} is a sequence, not a run config.")
    run_cfg.paths = paths
    task_cfg = run_cfg.get("task") or {}
    if not task_cfg.get("_target_") or not task_cfg.get("inputs"):
        raise ValueError(
            f"{path} carries no task _target_ / inputs, so the task cannot be rebuilt from it. Re-run that sweep."
        )
    return run_cfg


def recorded_tiling(run_cfg: DictConfig) -> dict:
    """The tiling the run trained on, which fixes the grid its model reads."""
    tiling = OmegaConf.to_container(run_cfg.tiling, resolve=True)
    if not isinstance(tiling, dict):
        raise ValueError(f"the run's tiling is {type(tiling).__name__}, not a mapping.")
    return tiling


def fold_target_stats(run_dir: Path) -> tuple[float, float]:
    """The scalar target scale a fold recorded in its ``metadata.json``: the fit-split mean and standard
    deviation of a z-scored regression head, the identity (0, 1) for every other scalar endpoint. A run that
    recorded no scalar scale raises."""
    path = run_dir / "metadata.json"
    meta = json.loads(path.read_text())
    mean, std = meta["target_mean"], meta["target_std"]
    if mean is None or std is None:
        raise ValueError(f"{path} records no scalar target_mean/target_std. Re-run that fold.")
    return float(mean), float(std)


def fold_time_edges(run_dir: Path) -> np.ndarray | None:
    """The interior survival time-bin edges a fold recorded in its ``metadata.json``, or None for an endpoint
    without them."""
    edges = json.loads((run_dir / "metadata.json").read_text()).get("time_edges")
    return None if edges is None else np.asarray(edges, dtype=np.float64)


def read_fold_states(field: str, folds: list[DevFold]) -> dict[tuple[int, int], dict[str, dict]]:
    """Each fold's recorded stream state, keyed by ``(outer, inner)``.

    Either every fold records a stream's state or none does. ``apply_stream_state`` with an empty payload is
    a no-op, so a fold that recorded nothing would silently run with another fold's statistics."""
    fold_state = {(o, i): read_stream_state(ckpt.parent) for o, i, ckpt in folds}
    recorded = {(o, i): set(state) for (o, i), state in fold_state.items()}
    every_key: set[str] = set().union(*recorded.values()) if recorded else set()
    short = {f"o{o}_i{i}": sorted(every_key - keys) for (o, i), keys in sorted(recorded.items()) if keys != every_key}
    if short:
        raise ValueError(
            f"[{field}] the sweep records fit-derived stream state inconsistently: "
            f"{len(short)}/{len(recorded)} fold(s) are missing state that others recorded "
            f"(e.g. {dict(list(short.items())[:4])}). Re-run those folds."
        )
    return fold_state


def rebuild_task(
    field: str,
    run_cfg: DictConfig,
    cohort: Cohort,
    data_description: Any,
    *,
    cohorts_dir: Path,
    contract_stats: Path,
    recorded_state: dict[str, dict],
) -> Any:
    """The run's task for ``field``, rebuilt over ``cohort``'s ``data_description``.

    The task class, its input streams, the feature extractor, the augmentations and the task's own options
    come from ``run_cfg``, and ``construct_task`` builds it as training does. Build ``data_description`` with
    ``recorded_tiling(run_cfg)``. The cohort's contract supplies the label map. Fit-derived state comes from
    ``recorded_state`` (one fold's record) and is never refitted. A stream that needs fit-split statistics
    the run did not record raises. ``contract_stats`` is the scored cohort's build-db statistics file, so a
    binarize or discretize target is cut at that cohort's own statistic (CONTRACT_SPEC §1.5)."""
    contract_field = cohort.contract[field]

    def _no_fit_split() -> list[str]:
        raise ValueError(
            f"input for '{field}' needs fit-split statistics that the run did not record, and they are not "
            f"refitted on the data being scored. Re-run that sweep so it records its stream state."
        )

    inputs = build_inputs(
        run_cfg,
        ModalityContext(
            cohort_name=cohort.name,
            cohorts_dir=cohorts_dir,
            data_description=data_description,
            feature_extractor=run_cfg.feature_extractor,
            fit_patient_codes=_no_fit_split,
            recorded_state=recorded_state.get,
        ),
    )
    # Target-side fit state (expression's per-gene statistics). The input streams get theirs through the
    # ModalityContext above.
    extras = resolve_recorded_extras(
        contract_field.objective,
        recorded_state,
        matrix_path=bulk_rna_matrix.matrix_path(cohorts_dir, cohort.name),
        gene_panel=OmegaConf.select(run_cfg, "task.target.gene_panel"),
    )
    task = construct_task(
        run_cfg.task,
        target=field,
        contract_field=contract_field,
        data_description=data_description,
        inputs=inputs,
        target_stats=load_target_stats(contract_stats, field),
        fit_extras=extras,
    )
    task.register_adapter_augmentations(hydra.utils.instantiate(run_cfg.augmentations))
    return task


def load(
    lit_module: torch.nn.Module,
    task: Any,
    checkpoint: Path,
    state: dict[str, dict],
    objective: Objective,
    device: torch.device,
) -> FitState:
    """Load a fold's weights into ``lit_module`` and install its stream ``state`` on ``task``. Returns what
    maps the fold's predictions back to label units: the per-gene statistics the fold trained with for an
    expression target, else ``fold_target_stats``, plus the fold's survival time bins."""
    checkpoint_state = torch.load(checkpoint, weights_only=False, map_location=device)
    lit_module.load_state_dict(checkpoint_state["state_dict"])
    apply_stream_state(task, state)
    time_edges = fold_time_edges(checkpoint.parent)
    if objective == Objective.regression_vector:
        fit_state = task.fit_state()
        return FitState(pred_mean=fit_state.pred_mean, pred_std=fit_state.pred_std, time_edges=time_edges)
    mean, std = fold_target_stats(checkpoint.parent)
    return FitState(pred_mean=mean, pred_std=std, time_edges=time_edges)


def split_patients(db_uri: str, manifest_name: str, split_version: str, categories: tuple[str, ...]) -> set[str]:
    """The patient codes that ``split_version`` of ``manifest_name`` assigns to ``categories``."""
    with DataManager(db_uri) as manager:
        return {
            str(record.patient_code)
            for category in categories
            for record in manager.get_records_by_split(
                manifest_name=manifest_name, split_version=split_version, split_category=category
            )
        }


def assert_disjoint_from_development(
    field: str, dev_db_uri: str, dev_cohort: str, scored_db_uri: str, scored_cohort: str, scored_split: str
) -> None:
    """Raise when ``scored_split`` of ``scored_cohort`` holds a patient of the development cohort.

    Every fold of a sweep scores the whole split, and each development patient was in the fit or validate
    split of most folds, so a shared patient would be scored by models that trained on it. Patients are
    matched by patient code across the two cohorts."""
    development = split_patients(dev_db_uri, dev_cohort, format_cv_split(field, 0, 0), _CV_CATEGORIES)
    scored = split_patients(scored_db_uri, scored_cohort, scored_split, (SplitCategory.TEST,))
    for name, patients, where in (
        ("development", development, f"'{dev_cohort}' {format_cv_split(field, 0, 0)}"),
        ("scored", scored, f"'{scored_cohort}' {scored_split}"),
    ):
        if not patients:
            raise ValueError(f"[{field}] the {name} split {where} has no patients, so disjointness cannot be checked.")
    shared = sorted(development & scored)
    if shared:
        raise ValueError(
            f"[{field}] {len(shared)} of the {len(scored)} patients in '{scored_cohort}' {scored_split} are also "
            f"patients of the development cohort '{dev_cohort}' (e.g. {shared[:5]}). The development models "
            f"trained on them, so they cannot validate those models. Score such a cohort fold by fold, each "
            f"fold only on its own held-out patients."
        )


@dataclass(frozen=True)
class Plan:
    """What a flow runs on a cohort: the development sweep's folds of each endpoint to run, and where the
    cohort's and the development cohort's DBs are."""

    dev_cohort: str
    exp_dir: Path  # runs/<experiment>/<dev_cohort>/
    fields: list[str]  # sorted
    folds: dict[str, list[DevFold]]  # field -> its complete folds, for every field in ``fields``
    db_uri: str  # the cohort's DB
    contract_stats: Path  # the cohort's build-db target statistics
    dev_db_uri: str
    device: torch.device


def plan(
    study: Study,
    cohort: Cohort,
    *,
    role: Role,
    flow: str,
    experiment: str,
    studies_dir: str | Path,
    strict: bool,
    only_fields: Collection[str] = (),
    reportable: Collection[str] | None = None,
) -> Plan:
    """The endpoints ``flow`` runs on ``cohort``, with the development sweep ``experiment`` that trained them.

    ``cohort`` must have ``role`` in ``study``, and the study must have exactly one development cohort. An
    endpoint runs when the study declares it, the sweep trained it, the cohort's contract declares it and,
    when ``reportable`` is given, its objective is in ``reportable``. Other endpoints are skipped with a
    warning. ``only_fields`` narrows that set, and a name outside it raises. Every endpoint's sweep must be
    complete (see ``require_complete``). Raises ValueError for each refusal."""
    found = study.cohorts.get(cohort.name)
    if found != role:
        raise ValueError(
            f"cohort '{cohort.name}' is not a '{role.value}' cohort of study '{study.name}' "
            f"(role={found.value if found else None})."
        )
    dev_cohorts = sorted(c for c, r in study.cohorts.items() if r == Role.development)
    if len(dev_cohorts) != 1:
        raise ValueError(f"expected exactly one development cohort in '{study.name}', got {dev_cohorts}.")
    dev_cohort = dev_cohorts[0]

    exp_dir = layout.runs_dir(studies_dir, study.name, experiment, dev_cohort)
    if not exp_dir.is_dir():
        raise ValueError(f"development runs dir not found: {exp_dir} (train the sweep first).")
    trained = discover(exp_dir)
    if not trained:
        raise ValueError(f"no completed folds (best.ckpt + metadata.json) under {exp_dir}.")

    fields: list[str] = []
    for field in study.targets:
        if field not in trained:
            logger.warning("skipping '%s': the development sweep has no completed folds for it.", field)
        elif field not in cohort.contract:
            logger.warning("skipping '%s': not in the contract of '%s'.", field, cohort.name)
        elif reportable is not None and cohort.contract[field].objective not in reportable:
            logger.warning(
                "skipping '%s': %s cannot report objective '%s' (it reports %s).",
                field,
                flow,
                cohort.contract[field].objective.value,
                sorted(reportable),
            )
        else:
            fields.append(field)
    fields.sort()
    if only_fields:
        unknown = sorted(set(only_fields) - set(fields))
        if unknown:
            raise ValueError(f"only_fields={unknown} are not endpoints {flow} can run (it can run {fields}).")
        fields = [f for f in fields if f in only_fields]
    if not fields:
        raise ValueError(f"no endpoint is shared by the development sweep and '{cohort.name}'.")

    for field in fields:
        require_complete(field, trained[field], study.splits.n_outer, study.splits.n_inner, strict=strict)
    return Plan(
        dev_cohort=dev_cohort,
        exp_dir=exp_dir,
        fields=fields,
        folds={f: trained[f] for f in fields},
        db_uri=layout.db_uri(studies_dir, study.name, cohort.name),
        contract_stats=layout.contract_stats_path(studies_dir, study.name, cohort.name),
        dev_db_uri=layout.db_uri(studies_dir, study.name, dev_cohort),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


@dataclass
class Rebuilt:
    """One endpoint's task, model and datamodule, rebuilt over a cohort. Every fold of a sweep runs through
    them, because only the weights and the stream state differ between folds."""

    field: str
    objective: Objective
    task: Any
    lit_module: pl.LightningModule
    datamodule: Any


def rebuild(
    field: str,
    run_cfg: DictConfig,
    cohort: Cohort,
    data_description: Any,
    recorded_state: dict[str, dict],
    *,
    cohorts_dir: Path,
    contract_stats: Path,
    num_workers: int,
    device: torch.device,
) -> Rebuilt:
    """The run's task (see ``rebuild_task``), model and datamodule for ``field`` over ``data_description``.

    ``recorded_state`` is one fold's record. It fixes the fold-independent parts (gene ids, panel, vector
    width) and they are validated against the cohort's data. ``predictions`` installs each fold's own state
    before that fold runs. The datamodule is the run's, with ``num_workers`` loader workers."""
    task = rebuild_task(
        field,
        run_cfg,
        cohort,
        data_description,
        cohorts_dir=cohorts_dir,
        contract_stats=contract_stats,
        recorded_state=recorded_state,
    )
    datamodule_cfg = OmegaConf.merge(run_cfg.datamodule, {"num_workers": num_workers})
    assert isinstance(datamodule_cfg, DictConfig)
    return Rebuilt(
        field=field,
        objective=cohort.contract[field].objective,
        task=task,
        lit_module=build_lit_module(run_cfg, task).to(device),
        datamodule=build_datamodule(task, data_description, datamodule_cfg),
    )


def predictions(
    rebuilt: Rebuilt,
    folds: list[DevFold],
    fold_states: dict[tuple[int, int], dict[str, dict]],
    *,
    stage: Stage,
    trainer_cfg: DictConfig,
    device: torch.device,
    require_labels: bool = True,
) -> Iterator[tuple[int, int, dict[str, Any]]]:
    """``(outer, inner, per-slide predictions)`` of each fold on the ``stage`` split, in fold order.

    Each fold's weights and stream state are loaded into ``rebuilt`` (``load``) before it runs, and its
    predictions are mapped back to label units with that fold's own target scale."""
    for outer, inner, checkpoint in folds:
        fit_state = load(
            rebuilt.lit_module, rebuilt.task, checkpoint, fold_states[(outer, inner)], rebuilt.objective, device
        )
        yield (
            outer,
            inner,
            slide_predictions(
                rebuilt.task,
                rebuilt.lit_module,
                rebuilt.datamodule,
                stage,
                trainer_cfg=trainer_cfg,
                endpoint_type=rebuilt.objective,
                target_key=f"patient.{rebuilt.field}",
                fit_state=fit_state,
                require_labels=require_labels,
            ),
        )


PerModel = dict[tuple[int, int], dict[str, Any]]


def cohort_predictions(
    rebuilt: Rebuilt,
    folds: list[DevFold],
    fold_states: dict[tuple[int, int], dict[str, dict]],
    *,
    stage: Stage,
    trainer_cfg: DictConfig,
    device: torch.device,
    require_labels: bool = True,
    tag: str = "",
) -> tuple[PerModel, dict[str, Any] | None, list | None]:
    """Every fold's per-patient predictions on the cohort ``rebuilt`` runs over, the per-patient labels (the
    same for every fold, or None with ``require_labels=False``), and the first fold's attention."""
    model = label_model(rebuilt.objective)
    per_model: PerModel = {}
    labels: dict[str, Any] | None = None
    attention: list | None = None
    for outer, inner, preds in predictions(
        rebuilt, folds, fold_states, stage=stage, trainer_cfg=trainer_cfg, device=device, require_labels=require_labels
    ):
        per_patient, patient_label = roll_slides_to_patients(
            np.asarray(preds["patient_codes"]),
            preds["probs"],
            preds["labels"] if require_labels else None,
            vector_preds=model.vector_preds,
            vector_labels=model.vector_labels,
        )
        if patient_label is not None:
            labels = (
                patient_label
                if model.vector_labels
                else {pc: model.scalar_label_cast(v) for pc, v in patient_label.items()}
            )
        per_model[(outer, inner)] = per_patient
        if attention is None and preds.get("attention"):
            attention = preds["attention"]
        logger.info("  [%s] %s o%d_i%d: %d patients", rebuilt.field, tag, outer, inner, len(per_patient))
    return per_model, labels, attention


def random_baseline_folds(
    study: Study, cohort: Cohort, plan: Plan, random_experiment: str | None, *, studies_dir: str | Path, strict: bool
) -> dict[str, list[DevFold]]:
    """The folds of the random-baseline sweep ``random_experiment`` for each expression endpoint of ``plan``.

    A sweep trained with ``random_baseline: true`` (an untrained model), the same arm aggregate reads as
    ``random_experiment_dir``. Only the expression endpoint uses it. Other objectives have a fixed chance
    level. Empty when ``random_experiment`` is None."""
    if not random_experiment:
        return {}
    rand_dir = layout.runs_dir(studies_dir, study.name, random_experiment, plan.dev_cohort)
    if not rand_dir.is_dir():
        raise ValueError(f"random_experiment runs dir not found: {rand_dir}.")
    found = discover(rand_dir)
    if not found:
        raise ValueError(f"no completed folds under random_experiment dir {rand_dir}.")
    random_folds: dict[str, list[DevFold]] = {}
    for field in plan.fields:
        if cohort.contract[field].objective != Objective.regression_vector:
            continue
        if field not in found:
            raise ValueError(f"[{field}] random_experiment {random_experiment!r} has no completed folds for it.")
        require_complete(field, found[field], study.splits.n_outer, study.splits.n_inner, strict=strict)
        random_folds[field] = found[field]
    return random_folds


def evaluate_field(
    field: str,
    plan: Plan,
    cohort: Cohort,
    run_cfg: DictConfig,
    *,
    cohorts_dir: Path,
    trainer_cfg: DictConfig,
    num_workers: int,
    random_folds: list[DevFold] | None = None,
) -> ExternalResult:
    """Score one endpoint on the validation cohort's ``all_test_<field>`` split with every development fold.

    The task, input streams, feature extractor and model come from the run's record. Only the cohort, and so
    the data description and the label map, is the validation one.

    ``random_folds`` (expression only) is a random-baseline sweep's folds. When given, the trained model's
    per-gene correlation is compared against that random-init model's on the same external patients
    (SEQUOIA's conservative significant-gene count). The random arm runs through the same rebuilt model
    and dataloader. Only its checkpoints and recorded per-fold state differ."""
    folds = plan.folds[field]
    data_description = compile_data_description(
        cohort, recorded_tiling(run_cfg), plan.db_uri, split_version=format_all_test(field)
    )
    fold_states = read_fold_states(field, folds)
    rebuilt = rebuild(
        field,
        run_cfg,
        cohort,
        data_description,
        fold_states[(folds[0][0], folds[0][1])],
        cohorts_dir=cohorts_dir,
        contract_stats=plan.contract_stats,
        num_workers=num_workers,
        device=plan.device,
    )
    per_model, labels, _ = cohort_predictions(
        rebuilt, folds, fold_states, stage="test", trainer_cfg=trainer_cfg, device=plan.device, tag="scored"
    )
    assert labels is not None
    result = external_ensemble(
        field, cohort.name, per_model, labels, endpoint_type=rebuilt.objective, num_classes=rebuilt.task.num_classes
    )

    # Expression null: the trained model must beat a random-init model of the same architecture per gene,
    # not merely beat zero correlation (SEQUOIA). Both arms see every external patient, so it is a paired
    # comparison on identical patients. Williams has df = n-3, so on a small external cohort the count is
    # underpowered (says "too few patients", not "does not transfer").
    if random_folds and rebuilt.objective == Objective.regression_vector:
        random_per_model, _, _ = cohort_predictions(
            rebuilt,
            random_folds,
            read_fold_states(field, random_folds),
            stage="test",
            trainer_cfg=trainer_cfg,
            device=plan.device,
            tag="random",
        )
        patients = sorted(result.labels)
        random_grand = ensemble_models(random_per_model, list(random_per_model), patients, vector=True)
        trained = np.stack([np.asarray(result.grand_patient_probs[p]) for p in patients])
        random = np.stack([random_grand[p] for p in patients])
        measured = np.stack([np.asarray(result.labels[p]) for p in patients])
        result.conservative = conservative_significant_gene_count(trained, random, measured)
        logger.info(
            "  [%s] significant genes (trained vs random-init): %d/%d",
            field,
            result.conservative["n_significant"],
            result.conservative["n_scored"],
        )
    return result


@dataclass
class CohortPrediction:
    """One endpoint's grand-ensemble prediction for every patient of an unlabelled cohort."""

    field: str
    objective: Objective
    num_classes: int
    patients: list[str]
    predictions: np.ndarray  # (N,), or (N, K|G) for a vector prediction, in ``patients`` order
    attention: list | None  # the first fold's per-tile attention, when the model has an attention branch
    tiling: dict  # the run's tiling, which places the attention tiles


def predict_field(
    field: str,
    plan: Plan,
    cohort: Cohort,
    run_cfg: DictConfig,
    *,
    cohorts_dir: Path,
    trainer_cfg: DictConfig,
    num_workers: int,
) -> CohortPrediction:
    """Grand-ensemble every development fold's per-patient prediction on the cohort's ``all_predict`` split.

    Everything about the model (task, inputs, extractor, architecture, per-fold fit-derived stream state)
    is rebuilt from the run's record, as for external scoring. Only the cohort and the split (all_predict, no
    labels) differ."""
    folds = plan.folds[field]
    data_description = compile_data_description(
        cohort, recorded_tiling(run_cfg), plan.db_uri, split_version=format_all_predict()
    )
    fold_states = read_fold_states(field, folds)
    rebuilt = rebuild(
        field,
        run_cfg,
        cohort,
        data_description,
        fold_states[(folds[0][0], folds[0][1])],
        cohorts_dir=cohorts_dir,
        contract_stats=plan.contract_stats,
        num_workers=num_workers,
        device=plan.device,
    )
    per_model, _, attention = cohort_predictions(
        rebuilt,
        folds,
        fold_states,
        stage="predict",
        trainer_cfg=trainer_cfg,
        device=plan.device,
        require_labels=False,
    )
    # The split and the datamodule do not depend on the fold, so every fold predicts the same patients.
    patients = require_same_patients(field, per_model)
    vector = label_model(rebuilt.objective).vector_preds
    grand = ensemble_models(per_model, list(per_model), patients, vector=vector)
    return CohortPrediction(
        field=field,
        objective=rebuilt.objective,
        num_classes=int(rebuilt.task.num_classes),
        patients=patients,
        predictions=np.asarray([grand[p] for p in patients], dtype=np.float64),
        attention=attention,
        tiling=recorded_tiling(run_cfg),
    )


def write_cohort_predictions(predictions: list[CohortPrediction], experiment_name: str, out_dir: Path) -> None:
    """For each endpoint ``per_patient_<field>.npz`` (the record: ``patient_codes``, ``preds``,
    ``endpoint_type``), a per-patient CSV for people and the attention when there is one, plus summary.md.
    The folder is written in full before it replaces the previous one (``layout.replaced_dir``)."""
    with layout.replaced_dir(out_dir) as tmp:
        lines = [f"# Predictions — {experiment_name}", "", "| endpoint | patients | prediction |", "|---|--:|---|"]
        for prediction in predictions:
            field, patients, preds = prediction.field, prediction.patients, prediction.predictions
            np.savez_compressed(
                layout.per_patient_npz(tmp, field),
                patient_codes=np.asarray(patients),
                preds=preds,
                endpoint_type=np.asarray(prediction.objective.value),
            )
            # One column for a scalar prediction, pred_0..pred_{K-1} for a vector.
            if preds.ndim == 2:
                header = "patient_code," + ",".join(f"pred_{k}" for k in range(preds.shape[1]))
                rows = [f"{p}," + ",".join(f"{v:.6g}" for v in preds[i]) for i, p in enumerate(patients)]
            else:
                header = "patient_code,prediction"
                rows = [f"{p},{preds[i]:.6g}" for i, p in enumerate(patients)]
            layout.per_patient_csv(tmp, field).write_text("\n".join([header, *rows]) + "\n")
            if prediction.attention:
                write_attention_npz(prediction.attention, tmp / f"attention_{field}.npz", tiling=prediction.tiling)
            shape = "scalar" if preds.ndim == 1 else f"vector of {preds.shape[1]}"
            lines.append(f"| {field} ({prediction.objective.value}) | {len(patients)} | {shape} |")
        lines += ["", "_The grand ensemble over every development fold. No labels, so no metrics._", ""]
        layout.summary_md(tmp).write_text("\n".join(lines))
