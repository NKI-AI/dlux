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

import statistics
from pathlib import Path

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
    database_uri, manifest_name, split_version, target, contract_field, target_stats=None, fit_patient_codes=None
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
    target,
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
        _fit_class_indices(
            database_uri, manifest_name, split_version, target, contract_field, target_stats, fit_patient_codes
        )
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
    database_uri, manifest_name, split_version, target, contract_field, target_stats=None, fit_patient_codes=None
):
    """Mean/std of the continuous target over this fold's ``fit`` split only (leakage-clean, like
    ``_fit_pos_weight``), for optional z-score standardisation. Missing/NaN excluded; population std;
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


def _target_zscore(
    objective,
    target_normalize,
    database_uri,
    manifest_name,
    split_version,
    target,
    contract_field,
    target_stats=None,
    fit_patient_codes=None,
):
    """Scalar-target (μ, σ): fit-split stats when regression + ``zscore`` requested, else identity (0, 1).
    Shared by the regression and fusion paths (the fusion head trains its scalar target in z-space too)."""
    if objective == Objective.regression and target_normalize == "zscore":
        return _fit_target_stats(
            database_uri, manifest_name, split_version, target, contract_field, target_stats, fit_patient_codes
        )
    return 0.0, 1.0


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


def resolve_fit_extras(
    objective,
    *,
    cfg,
    manifest_name,
    contract_field,
    target,
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
    needs beyond ``common``. Every stat reads the ``fit`` patients only, named either by
    ``split_version`` or by ``fit_patient_codes`` (see ``_fit_records``).

    These are target-side statistics only. An input stream resolves its own (see ``Modality.from_spec``)."""
    extras = {}

    # Class weighting is meaningful only where there are classes. Warned rather than raised because a
    # study can share one experiment recipe across endpoints of different objectives.
    if loss_weight != "none" and objective not in (Objective.binary, Objective.multiclass):
        logger.warning(
            "task.loss.weight=%r ignored for %s endpoints (classification only).", loss_weight, objective.value
        )

    # Expression target: active gene panel + per-gene fit-split μ/σ.
    if objective == Objective.regression_vector:
        gene_ids, gene_means, gene_stds = _fit_gene_stats(
            matrix_path,
            database_uri,
            manifest_name,
            split_version,
            gene_panel,
            matrix_path.parent / "panels",
            fit_patient_codes,
        )
        extras.update(
            matrix_path=matrix_path,
            gene_means=gene_means,
            gene_stds=gene_stds,
            gene_ids=gene_ids,
            gene_panel=gene_panel,
        )

    # Survival: discrete-time bin edges (+ n_bins). Identical for the fused and unfused survival tasks.
    if objective == Objective.survival:
        n_bins = int(cfg.task.get("n_bins", 4))
        extras["time_edges"] = _fit_time_bins(
            database_uri,
            manifest_name,
            split_version,
            contract_field.source.column,
            contract_field.source.time_column,
            n_bins,
            fit_patient_codes,
        )
        extras["n_bins"] = n_bins
        return extras

    if objective == Objective.regression:
        mean, std = _target_zscore(
            objective,
            target_normalize,
            database_uri,
            manifest_name,
            split_version,
            target,
            contract_field,
            target_stats,
            fit_patient_codes,
        )
        if target_normalize == "zscore":
            logger.info("target_normalize=zscore -> fit-split μ=%.4g σ=%.4g", mean, std)
        return {"target_mean": mean, "target_std": std}

    if objective == Objective.regression_vector:
        if target_normalize != "none":
            logger.warning(
                "task.target_normalize=%r ignored for expression (log1p + per-gene z-score is intrinsic).",
                target_normalize,
            )
        return extras  # the gene extras assembled above

    # binary / multiclass -> classification (pos_weight for binary, class_weights for multiclass).
    if target_normalize != "none":
        logger.warning(
            "task.target_normalize=%r ignored for %s endpoints (regression only).", target_normalize, objective.value
        )
    return _fit_loss_weights(
        database_uri, manifest_name, split_version, target, contract_field, loss_weight, target_stats, fit_patient_codes
    )


def resolve_replay_extras(objective, recorded, *, matrix_path, gene_panel):
    """The mirror of ``resolve_fit_extras`` for a cohort with no ``fit`` split: the same target-side
    kwargs, replayed from what the run recorded rather than fitted from the DB.

    Only endpoints whose target carries fit-derived state that scoring needs produce anything here.
    Survival's time edges are training-only, and a z-scored scalar target's μ/σ come from the run's
    ``metadata.json``, so both yield ``{}``, expression is the one endpoint that needs this.
    """
    if objective != Objective.regression_vector:
        return {}

    key = "expression"  # the stream key WSIBulkRNATask gives its RNA target
    state = recorded.get(key)
    if state is None:
        raise ValueError(
            f"scoring an expression target needs the per-gene statistics the run fitted, but it recorded "
            f"no '{key}' stream state. An external cohort has no fit split to refit them from, and "
            f"refitting here would change the input distribution anyway. Re-run that sweep so it records "
            f"its stream state."
        )
    gene_ids, gene_means, gene_stds = BulkRNA.replay_stats(
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
    inputs = {}
    for name, spec in OmegaConf.to_container(specs, resolve=True).items():
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

    The task class comes from the task config's ``_target_``, like every other constructed component.
    Each class then validates the contract objective it was handed. ``resolve_fit_extras`` supplies the
    class's fold-specific kwargs (class weighting / target z-score / gene panel / survival bins), all
    read from the ``fit`` patients only. Raises ValueError on a bad config value.

    Callers name the fit set one of two ways, and exactly one (see ``_fit_records``): ``split_version``
    for a split stored in the DB, ``fit_patient_codes`` for one drawn in memory."""
    objective = contract_field.objective

    if "fuse_rna" in cfg.task:
        raise ValueError("task.fuse_rna is gone. Declare the model's data streams under `inputs:` instead.")

    target_normalize = str(cfg.task.get("target_normalize", "none"))
    if target_normalize not in ("none", "zscore"):
        raise ValueError(f"task.target_normalize must be 'none' or 'zscore', got '{target_normalize}'.")

    # Validated through the schema rather than by hand: `weight` admits two literals or a per-class
    # map, which is more shape than an inline check states clearly.
    raw_loss = cfg.task.get("loss")
    loss_weight = LossSpec(**(OmegaConf.to_container(raw_loss, resolve=True) if raw_loss else {})).weight

    task_path = cfg.task.get("_target_")
    if not task_path:
        raise ValueError("task config must declare `_target_` (the task class), e.g. config/task/wsi_survival.yaml.")
    task_cls = get_class(str(task_path))
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
    common = dict(
        target=target,
        contract_field=contract_field,
        data_description=data_description,
        inputs=inputs,
        target_stats=target_stats,
    )
    extras = resolve_fit_extras(
        objective,
        cfg=cfg,
        manifest_name=cohort.name,
        contract_field=contract_field,
        target=target,
        target_normalize=target_normalize,
        loss_weight=loss_weight,
        target_stats=target_stats,
        database_uri=database_uri,
        split_version=split_version,
        fit_patient_codes=fit_patient_codes,
        matrix_path=bulk_rna_matrix.matrix_path(cfg.paths.cohorts_dir, cohort.name),
        gene_panel=str(cfg.task.target.get("gene_panel", "full")),  # a read parameter of the target
    )
    return task_cls(**common, **extras)
