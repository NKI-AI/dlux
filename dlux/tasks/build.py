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
"""Task construction: instantiate the class the task config names, and resolve each endpoint's
fold-specific, leakage-clean fit-time setup (class weighting / scalar target z-score / per-gene z-score
+ gene panel) + the head-width. All of it reads the DB's ``fit`` split only, so it never sees
validate/test.

Kept here rather than in the train bin so ``train.py`` stays a generic orchestrator: it calls
``build_task`` and knows nothing about pos-weights, target statistics, or gene panels.
"""

from __future__ import annotations

import inspect
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from hydra.utils import get_class
from omegaconf import DictConfig, OmegaConf

from ahcore.manifest import DataManager, get_labels_from_record
from ahcore.utils.io import get_logger

from dlux.config.cohort import Objective, SplitCategory
from dlux.config.task import LossSpec
from dlux.data import bulk_rna_matrix, layout
from dlux.data.compile import compile_data_description
from dlux.modalities import MODALITIES, BulkRNA
from dlux.modalities.context import ModalityContext
from dlux.tasks.classification import resolve_class_weights, resolve_pos_weight
from dlux.tasks.target import build_field_transform, load_target_stats

logger = get_logger(__name__)


def _fit_records(database_uri, manifest_name, split_version, fit_patient_codes=None):
    """This fold's ``fit`` patient records, the leakage boundary shared by every ``_fit_*`` helper
    below. Kept lazy: each helper calls it only when it actually needs the stats.

    Two ways to name the fit set, and exactly one must be given. ``split_version`` reads a split
    stored in the DB. ``fit_patient_codes`` names the patients directly, for callers that drew the
    split in memory and never wrote it (``bin/train_resplit.py``), where the same codes are already
    the authority for what the model trains on.
    """
    if (split_version is None) == (fit_patient_codes is None):
        raise ValueError(
            "fit records need exactly one of split_version or fit_patient_codes, got "
            f"split_version={split_version!r} and "
            f"fit_patient_codes={'a set' if fit_patient_codes is not None else None}. Supplying both "
            "leaves it ambiguous which patients the fit-time statistics were computed over."
        )
    manager = DataManager(database_uri)
    if fit_patient_codes is not None:
        wanted = set(fit_patient_codes)
        return (p for p in manager.get_all_records(manifest_name) if str(p.patient_code) in wanted)
    return manager.get_records_by_split(
        manifest_name=manifest_name, split_version=split_version, split_category=SplitCategory.FIT
    )


def _fit_class_indices(
    database_uri, manifest_name, split_version, contract_field, target_stats=None, fit_patient_codes=None
):
    """This fold's ``fit``-split class indices, mapped through the same contract transform the model
    uses. Patients whose label is absent are skipped; out-of-map values arrive as ``EXCLUDED``."""
    mapping = build_field_transform(contract_field, target_stats)
    indices: list[int] = []
    for patient in _fit_records(database_uri, manifest_name, split_version, fit_patient_codes):
        raw = dict(get_labels_from_record(patient) or []).get(contract_field.source.column)
        if raw is not None:
            indices.append(mapping[str(raw)])
    return indices


def _fit_loss_weights(
    database_uri,
    manifest_name,
    split_version,
    contract_field,
    weight_spec,
    target_stats=None,
    fit_patient_codes=None,
):
    """Resolve the task's class weighting to the classification task's weighting kwargs:
    ``pos_weight`` (binary BCE, a scalar) or ``class_weights`` (multiclass CE, a length-K vector).

    ``"balanced"`` is inverse-frequency from this fold's ``fit`` split only, so it never sees
    validate/test (leakage-clean). Manual weights need no data."""
    is_binary = contract_field.objective == Objective.binary
    if weight_spec == "none":
        return {"pos_weight": None, "class_weights": None}

    # Only "balanced" needs the train-split class frequencies.
    fit_indices = (
        _fit_class_indices(database_uri, manifest_name, split_version, contract_field, target_stats, fit_patient_codes)
        if weight_spec == "balanced"
        else []
    )
    if is_binary:
        pos_weight = resolve_pos_weight(weight_spec, fit_indices)
        balance = (
            f" (fit: {sum(c == 1 for c in fit_indices)} pos / {sum(c == 0 for c in fit_indices)} neg)"
            if fit_indices
            else ""
        )
        logger.info("loss.weight=%r -> pos_weight=%s%s", weight_spec, pos_weight, balance)
        return {"pos_weight": pos_weight, "class_weights": None}

    num_classes = contract_field.num_outputs
    class_weights = resolve_class_weights(weight_spec, fit_indices, num_classes)
    support = [sum(c == k for c in fit_indices) for k in range(num_classes)]
    balance = f" (fit support: {support})" if fit_indices else ""
    logger.info("loss.weight=%r -> class_weights=%s%s", weight_spec, [round(w, 3) for w in class_weights], balance)
    return {"pos_weight": None, "class_weights": class_weights}


def _fit_target_stats(
    database_uri, manifest_name, split_version, contract_field, target_stats=None, fit_patient_codes=None
):
    """Mean/std of the continuous target over this fold's ``fit`` split only (leakage-clean, like
    ``_fit_loss_weights``), for optional z-score standardisation. Missing/NaN excluded; population std;
    degenerate (n<2 or std=0) falls back to (0, 1) = identity."""
    transform = build_field_transform(contract_field, target_stats)
    records = _fit_records(database_uri, manifest_name, split_version, fit_patient_codes)
    values: list[float] = []
    for patient in records:
        raw = dict(get_labels_from_record(patient) or []).get(contract_field.source.column)
        if raw is None:
            continue
        value = float(transform(raw))
        if not (value != value):  # skip NaN (missing/out-of-contract)
            values.append(value)
    if len(values) < 2:
        return 0.0, 1.0
    mean = statistics.fmean(values)
    std = statistics.pstdev(values, mu=mean)
    return (mean, std) if std > 0.0 else (mean, 1.0)


def _fit_gene_stats(
    matrix_path, database_uri, manifest_name, split_version, gene_panel, panels_dir, fit_patient_codes=None
):
    """Active gene panel + per-gene μ/σ of log1p expression over this fold's ``fit`` split only
    (leakage-clean, like ``_fit_target_stats``).

    The statistics themselves belong to the bulk_rna modality. What stays here is the leakage boundary:
    reading the ``fit`` split and nothing else, then handing over just those patient codes."""
    records = _fit_records(database_uri, manifest_name, split_version, fit_patient_codes)
    resolved_codes = [str(patient.patient_code) for patient in records]
    try:
        return BulkRNA.fit_stats(
            matrix_path=matrix_path,
            fit_patient_codes=resolved_codes,
            gene_panel=gene_panel,
            panels_dir=panels_dir,
        )
    except ValueError as exc:
        raise ValueError(f"{exc} (split '{split_version}')") from exc


def _fit_time_bins(database_uri, manifest_name, split_version, event_col, time_col, n_bins, fit_patient_codes=None):
    """Interior quantile edges (``n_bins - 1`` cut points) for discretising follow-up time, from this
    fold's ``fit`` split only (leakage-clean). Edges are quantiles of the observed-event times (the
    convention for discrete-time survival), falling back to all follow-up times when there are too
    few events to bin. Raises if the fit split has fewer than ``n_bins`` usable follow-up times."""
    records = _fit_records(database_uri, manifest_name, split_version, fit_patient_codes)
    times: list[float] = []
    event_times: list[float] = []
    for patient in records:
        labels = dict(get_labels_from_record(patient) or [])
        try:
            time = float(labels.get(time_col))
            event = float(labels.get(event_col))
        except (TypeError, ValueError):
            continue
        if time <= 0.0:
            continue
        times.append(time)
        if event == 1.0:
            event_times.append(time)
    basis = event_times if len(event_times) >= n_bins else times
    if len(basis) < n_bins:
        raise ValueError(
            f"survival needs >= {n_bins} fit-split patients with follow-up time, got {len(basis)} "
            f"for split '{split_version}'."
        )
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.quantile(np.asarray(basis, dtype=float), quantiles).astype(np.float32)
    logger.info("survival: %d-bin time edges from %d fit-split event times: %s", n_bins, len(event_times), edges)
    return edges


# The top-level task-config keys that task construction reads itself. Every other key is an option of the
# task class and must be a parameter of its constructor.
_BUILD_KEYS = {"_target_", "target", "target_normalize", "loss", "inputs"}

# The constructor arguments construction supplies itself, never from the task config.
_SUPPLIED = {"self", "target", "contract_field", "data_description", "inputs", "target_stats"}


def task_class_and_options(task_cfg) -> tuple[type, dict]:
    """The task class the config's ``_target_`` names, and the config's options for its constructor.

    Every task-config key outside the keys construction reads itself is an option, so a task class takes its
    own options from YAML. A key the constructor does not accept raises ``ValueError``, so a typo cannot
    pass without effect."""
    task_path = task_cfg.get("_target_")
    if not task_path:
        raise ValueError("task config must declare `_target_` (the task class), e.g. config/task/wsi_survival.yaml.")
    task_cls = get_class(str(task_path))
    accepted = set(inspect.signature(task_cls).parameters) - _SUPPLIED
    options = {key: task_cfg[key] for key in task_cfg if key not in _BUILD_KEYS}
    unknown = sorted(set(options) - accepted)
    if unknown:
        raise ValueError(
            f"task: unknown key(s) {unknown}. {task_cls.__name__} takes {sorted(accepted)}, and task "
            f"construction reads {sorted(_BUILD_KEYS)}."
        )
    options = {
        key: OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
        for key, value in options.items()
    }
    return task_cls, options


def construct_task(task_cfg, *, target, contract_field, data_description, inputs, target_stats, fit_extras) -> Any:
    """The task the config names, built over ``data_description``.

    Training and scoring both build a task here, and differ only in where ``inputs`` and ``fit_extras`` come
    from: fitted on a fold's fit split, or restored from what a run recorded."""
    task_cls, options = task_class_and_options(task_cfg)
    return task_cls(
        target=target,
        contract_field=contract_field,
        data_description=data_description,
        inputs=inputs,
        target_stats=target_stats,
        **options,
        **fit_extras,
    )


@dataclass(frozen=True)
class _FitRequest:
    """What every endpoint's fit-time setup may read. Each ``_*_extras`` function below uses its own part."""

    cfg: Any
    manifest_name: str
    contract_field: Any
    target_normalize: str
    loss_weight: Any
    database_uri: str
    split_version: Optional[str]
    matrix_path: Path
    gene_panel: str
    target_stats: Any
    fit_patient_codes: Any


def _warn_loss_weight_ignored(request: _FitRequest) -> None:
    # Warned rather than raised because a study can share one experiment recipe across endpoints of
    # different objectives.
    if request.loss_weight != "none":
        logger.warning(
            "task.loss.weight=%r ignored for %s endpoints (classification only).",
            request.loss_weight,
            request.contract_field.objective.value,
        )


def _warn_target_normalize_ignored(request: _FitRequest, reason: str) -> None:
    if request.target_normalize != "none":
        logger.warning(
            "task.target_normalize=%r ignored for %s endpoints (%s).",
            request.target_normalize,
            request.contract_field.objective.value,
            reason,
        )


def _classification_extras(request: _FitRequest) -> dict:
    """``pos_weight`` for binary, ``class_weights`` for multiclass (see ``_fit_loss_weights``)."""
    _warn_target_normalize_ignored(request, "regression only")
    return _fit_loss_weights(
        request.database_uri,
        request.manifest_name,
        request.split_version,
        request.contract_field,
        request.loss_weight,
        request.target_stats,
        request.fit_patient_codes,
    )


def _regression_extras(request: _FitRequest) -> dict:
    """The scalar target's fit-split (μ, σ) when ``target_normalize`` is ``zscore``, else the identity (0, 1)."""
    _warn_loss_weight_ignored(request)
    if request.target_normalize != "zscore":
        return {"target_mean": 0.0, "target_std": 1.0}
    mean, std = _fit_target_stats(
        request.database_uri,
        request.manifest_name,
        request.split_version,
        request.contract_field,
        request.target_stats,
        request.fit_patient_codes,
    )
    logger.info("target_normalize=zscore -> fit-split μ=%.4g σ=%.4g", mean, std)
    return {"target_mean": mean, "target_std": std}


def _survival_extras(request: _FitRequest) -> dict:
    """The discrete-time bin edges from the fit split. ``n_bins`` itself is the task's own option."""
    _warn_loss_weight_ignored(request)
    _warn_target_normalize_ignored(request, "regression only")
    n_bins = request.cfg.task.get("n_bins")
    if n_bins is None:
        raise ValueError("a survival task config sets n_bins, the number of discrete-time intervals.")
    n_bins = int(n_bins)
    time_edges = _fit_time_bins(
        request.database_uri,
        request.manifest_name,
        request.split_version,
        request.contract_field.source.column,
        request.contract_field.source.time_column,
        n_bins,
        request.fit_patient_codes,
    )
    return {"time_edges": time_edges}


def _expression_extras(request: _FitRequest) -> dict:
    """The active gene panel and the per-gene fit-split μ/σ of log1p expression."""
    _warn_loss_weight_ignored(request)
    _warn_target_normalize_ignored(request, "log1p + per-gene z-score is intrinsic")
    gene_ids, gene_means, gene_stds = _fit_gene_stats(
        request.matrix_path,
        request.database_uri,
        request.manifest_name,
        request.split_version,
        request.gene_panel,
        request.matrix_path.parent / "panels",
        request.fit_patient_codes,
    )
    return dict(
        matrix_path=request.matrix_path,
        gene_means=gene_means,
        gene_stds=gene_stds,
        gene_ids=gene_ids,
        gene_panel=request.gene_panel,
    )


# Each objective's fit-time setup. Every one reads the ``fit`` patients only.
_FIT_EXTRAS: dict[Objective, Callable[[_FitRequest], dict]] = {
    Objective.binary: _classification_extras,
    Objective.multiclass: _classification_extras,
    Objective.regression: _regression_extras,
    Objective.survival: _survival_extras,
    Objective.regression_vector: _expression_extras,
}


def resolve_fit_extras(
    objective,
    *,
    cfg,
    manifest_name,
    contract_field,
    target_normalize,
    loss_weight,
    database_uri,
    split_version,
    matrix_path,
    gene_panel,
    target_stats=None,  # build-db-resolved, fold-invariant (a binarize cut); None when the endpoint needs none
    fit_patient_codes=None,  # names the fit set directly when the split was drawn in memory
):
    """The endpoint's fold-specific, leakage-clean fit-time kwargs, exactly the extras its ``__init__``
    needs beyond ``common`` (see ``_FIT_EXTRAS``). Every stat reads the ``fit`` patients only, named either
    by ``split_version`` or by ``fit_patient_codes`` (see ``_fit_records``). A knob the objective does not
    use is logged as ignored.

    These are target-side statistics only. An input stream resolves its own (see ``Modality.from_spec``)."""
    request = _FitRequest(
        cfg=cfg,
        manifest_name=manifest_name,
        contract_field=contract_field,
        target_normalize=target_normalize,
        loss_weight=loss_weight,
        database_uri=database_uri,
        split_version=split_version,
        matrix_path=matrix_path,
        gene_panel=gene_panel,
        target_stats=target_stats,
        fit_patient_codes=fit_patient_codes,
    )
    return _FIT_EXTRAS[objective](request)


def resolve_recorded_extras(objective, recorded, *, matrix_path, gene_panel):
    """The mirror of ``resolve_fit_extras`` for a cohort with no ``fit`` split: the same target-side
    kwargs, restored from what the run recorded rather than fitted from the DB.

    Only endpoints whose target carries fit-derived state that scoring needs produce anything here.
    Survival's time edges are training-only, and a z-scored scalar target's μ/σ come from the run's
    ``metadata.json``, so both yield ``{}``, expression is the one endpoint that needs this.
    """
    if objective != Objective.regression_vector:
        return {}

    if gene_panel is None:
        raise ValueError("an expression run records its gene panel at task.target.gene_panel, and this one has none.")
    key = "expression"  # the stream key WSIBulkRNATask gives its RNA target
    state = recorded.get(key)
    if state is None:
        raise ValueError(
            f"scoring an expression target needs the per-gene statistics the run fitted, but it recorded "
            f"no '{key}' stream state. An external cohort has no fit split to refit them from, and "
            f"refitting here would change the input distribution anyway. Re-run that sweep so it records "
            f"its stream state."
        )
    gene_ids, gene_means, gene_stds = BulkRNA.recorded_stats(
        state, key=key, matrix_path=matrix_path, gene_panel=gene_panel
    )
    return dict(
        matrix_path=matrix_path,
        gene_means=gene_means,
        gene_stds=gene_stds,
        gene_ids=gene_ids,
        gene_panel=gene_panel,
    )


def _target_field(cfg) -> str:
    """The contract field this run predicts. ``target`` is a block so it can carry read parameters (e.g.
    a gene panel); it names a field rather than a modality because a target's storage is a property of
    the cohort, recorded in the contract, not a per-run choice."""
    target = cfg.task.get("target")
    if not isinstance(target, (dict, DictConfig)) or target.get("field") in (None, ""):
        raise ValueError("task.target must be a block naming a contract field, e.g. `target: {field: os}`.")
    return str(target["field"])


def build_inputs(cfg, ctx: ModalityContext) -> dict:
    """Construct this run's declared input streams from ``task.inputs``.

    Each modality builds itself from its own spec, so this stays generic, a new modality is a registry
    entry plus a ``from_spec``, never a branch here."""
    specs = cfg.task.get("inputs")
    if not specs:
        raise ValueError("task config must declare `inputs:` — the named data streams the model reads.")
    declared = OmegaConf.to_container(specs, resolve=True)
    if not isinstance(declared, dict):
        raise ValueError("task `inputs:` must be a mapping of stream name to spec.")
    inputs = {}
    for raw_name, spec in declared.items():
        name = str(raw_name)
        if not isinstance(spec, dict) or not spec.get("modality"):
            raise ValueError(f"input '{name}': must declare `modality:` (one of {sorted(MODALITIES)}).")
        modality = MODALITIES.get(str(spec["modality"]))
        if modality is None:
            raise ValueError(f"input '{name}': unknown modality '{spec['modality']}' (have: {sorted(MODALITIES)}).")
        inputs[name] = modality.from_spec(key=name, spec=spec, ctx=ctx)
    return inputs


def _warn_ungated_inputs(cfg, inputs: dict) -> None:
    """Warn when a run reads a modality whose patients the study does not gate on.

    Such an input silently shrinks (or distorts) the usable set for this arm only, so arms of the same
    study stop being comparable while every report still looks complete."""
    required = set(cfg.study.get("require_modalities") or [])
    ungated = sorted({m.name for m in inputs.values() if hasattr(type(m), "coverage")} - required)
    if ungated:
        logger.warning(
            "inputs %s are not in study.require_modalities %s: patients lacking them are still in the "
            "splits, so this arm may not be comparable with the study's other arms.",
            ungated,
            sorted(required),
        )


def build_task(cfg, cohort, contract_field, tiling, database_uri, split_version, fit_patient_codes=None):
    """Construct the endpoint's task (with its fold-specific fit-time setup + head width) from config.

    The task class comes from the task config's ``_target_``, like every other constructed component, and
    ``construct_task`` builds it. Each class then validates the contract objective it was handed.
    ``resolve_fit_extras`` supplies the class's fold-specific kwargs (class weighting / target z-score /
    gene panel / survival bins), all read from the ``fit`` patients only. Raises ValueError on a bad config
    value.

    Callers name the fit set one of two ways, and exactly one (see ``_fit_records``): ``split_version``
    for a split stored in the DB, ``fit_patient_codes`` for one drawn in memory."""
    objective = contract_field.objective
    task_class_and_options(cfg.task)  # refuse a bad task config before the slow steps below

    target_normalize = str(cfg.task.get("target_normalize", "none"))
    if target_normalize not in ("none", "zscore"):
        raise ValueError(f"task.target_normalize must be 'none' or 'zscore', got '{target_normalize}'.")

    # Validated through the schema: `weight` admits two literals or a per-class map.
    raw_loss = cfg.task.get("loss")
    loss_weight = LossSpec(**(OmegaConf.to_container(raw_loss, resolve=True) if raw_loss else {})).weight

    target = _target_field(cfg)
    data_description = compile_data_description(cohort, tiling, database_uri, split_version=split_version)

    def _fit_patient_codes(_cache={}):  # lazy + memoised: only a fold-dependent modality pays for it
        if "codes" not in _cache:
            _cache["codes"] = [
                str(p.patient_code) for p in _fit_records(database_uri, cohort.name, split_version, fit_patient_codes)
            ]
        return _cache["codes"]

    # Fold-invariant target statistics, resolved once at build-db over the whole cohort (a binarize
    # cut). Distinct from the fit extras below, which are per-fold and leakage-sensitive by design.
    target_stats = load_target_stats(
        layout.contract_stats_path(cfg.paths.studies_dir, str(cfg.study.name), cohort.name), target
    )

    inputs = build_inputs(
        cfg,
        ModalityContext(
            cohort_name=cohort.name,
            cohorts_dir=Path(cfg.paths.cohorts_dir),
            data_description=data_description,
            feature_extractor=cfg.feature_extractor,
            fit_patient_codes=_fit_patient_codes,
        ),
    )
    _warn_ungated_inputs(cfg, inputs)
    extras = resolve_fit_extras(
        objective,
        cfg=cfg,
        manifest_name=cohort.name,
        contract_field=contract_field,
        target_normalize=target_normalize,
        loss_weight=loss_weight,
        target_stats=target_stats,
        database_uri=database_uri,
        split_version=split_version,
        fit_patient_codes=fit_patient_codes,
        matrix_path=bulk_rna_matrix.matrix_path(cfg.paths.cohorts_dir, cohort.name),
        gene_panel=str(cfg.task.target.get("gene_panel", "full")),  # a read parameter of the target
    )
    return construct_task(
        cfg.task,
        target=target,
        contract_field=contract_field,
        data_description=data_description,
        inputs=inputs,
        target_stats=target_stats,
        fit_extras=extras,
    )
