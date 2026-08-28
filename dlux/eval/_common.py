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
"""Framework leaf for the eval subsystem: the pieces shared by every ``eval/<objective>.py`` module and
by both drivers (``aggregate.py`` / ``external.py``). Result models, the NPZ loader, slide→patient
rollups, the scalar aggregation core (binary and regression), the shared matplotlib init, the confusion
figure, and the two handler types.

Import direction (must stay acyclic): objective modules and drivers import down from here. ``_common``
imports nothing from ``eval`` except the pure-math leaves. The aggregation cores take the endpoint
``handler`` as a parameter, so this leaf never imports the driver's registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import numpy as np

from dlux.config.cohort import Objective

if TYPE_CHECKING:
    from dlux.eval.external import ExternalResult


def _agg_pyplot():
    """Return a headless (Agg) ``pyplot``, the one place the ``matplotlib.use("Agg")`` init lives, so
    every figure helper is import-light and can't race on the backend. Import is lazy (figures are
    optional) and the module cache makes repeat calls fast."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# -- result models -----------------------------------------------------------
@dataclass
class PatientPrediction:
    field: str
    outer_fold: int
    patient_code: str
    ensemble_prob: float  # positive-class prob (binary) / predicted value (regression) / risk (survival)
    label: float  # class index (classification) / continuous target (regression) / follow-up time (survival)
    n_replicates: int
    ensemble_probs: Optional[np.ndarray] = None  # (K,) per-class softmax, multiclass only
    event: Optional[float] = None  # observed-event indicator (0/1), survival only


@dataclass
class OuterFoldMetric:
    field: str
    outer_fold: int
    n_patients: int
    n_positive: int
    metrics: Dict[str, float]


@dataclass
class AggregateResult:
    field: str
    endpoint_type: Objective
    patient_predictions: List[PatientPrediction]
    per_outer: List[OuterFoldMetric]
    mean: Dict[str, float]  # mean across outer folds, per metric
    std: Dict[str, float]  # std across outer folds, per metric
    pooled: Dict[str, float]  # metrics on the pooled OOF set
    pooled_n: int
    pooled_positive: int
    n_outer_folds: int
    inner_coverage: Dict[int, int]  # outer_fold -> #inner replicates aggregated
    pooled_gene_pearson: Optional[np.ndarray] = None  # (G,) per-gene OOF Pearson, expression endpoint only
    pooled_significant: Optional[dict] = None  # {n_significant, n_scored, alpha}, expression endpoint only
    pooled_conservative: Optional[dict] = None  # SEQUOIA count vs a random-baseline sweep, expression only
    # Pooled-OOF per-patient gene matrices, expression endpoint only. Columns share the order of
    # ``pooled_gene_pearson`` (and the per_gene CSV), so they join to the cohort's genes.csv the same way.
    patient_codes: Optional[List[str]] = None
    pooled_outer_folds: Optional[List[int]] = None  # (N,) outer fold each patient's OOF prediction came from
    pooled_preds: Optional[np.ndarray] = None  # (N, G) predicted expression, one ensembled vector per patient
    pooled_labels: Optional[np.ndarray] = None  # (N, G) measured expression, aligned to pooled_preds
    num_classes: Optional[int] = None  # K, multiclass endpoint only
    pooled_per_class: Optional[List[dict]] = None  # per-class pooled AUROC/precision/recall/f1/support, multiclass
    coverage: Optional[dict] = None  # {declared, scored, lost} TEST patients; None = coverage was not recorded


# -- IO + rollups ------------------------------------------------------------
def load_fold_predictions(npz_path: str | Path) -> Dict[str, np.ndarray]:
    """Load one fold's persisted prediction NPZ (written by dlux.eval.predictions)."""
    with np.load(npz_path, allow_pickle=True) as d:
        missing = {"patient_codes", "probs", "labels", "endpoint_type"} - set(d.keys())
        if missing:
            raise KeyError(f"{npz_path} missing keys: {sorted(missing)}")
        return {
            "patient_codes": np.asarray(d["patient_codes"]),
            "probs": np.asarray(d["probs"], dtype=np.float64),
            # float64, not int64: regression targets are continuous (int64 would truncate a [0,1)
            # fraction to 0). Binary paths re-cast labels to int where metrics need class indices.
            "labels": np.asarray(d["labels"], dtype=np.float64),
            # Normalize the wire string back to the enum at this single read boundary; everything
            # downstream carries the Objective (str-enum, so it still compares/keys as the string).
            "endpoint_type": Objective(str(d["endpoint_type"])),
        }


def _roll_slides_to_patients(
    patient_codes: np.ndarray, probs: np.ndarray, labels: np.ndarray
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Mean slide-prediction per patient (+ the patient's label as float, class idx for
    classification, continuous target for regression; binary metrics re-cast to int internally)."""
    slide_probs: Dict[str, List[float]] = {}
    patient_label: Dict[str, float] = {}
    for pc, prob, lbl in zip(patient_codes, probs, labels):
        pc = str(pc)
        slide_probs.setdefault(pc, []).append(float(prob))
        patient_label[pc] = float(lbl)
    return {pc: float(np.mean(v)) for pc, v in slide_probs.items()}, patient_label


def _roll_slides_to_patients_vec(
    patient_codes: np.ndarray, preds: np.ndarray, labels: np.ndarray
) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Mean slide-prediction vector per patient (+ the patient's label vector). The vector analog of
    ``_roll_slides_to_patients`` for the expression endpoint."""
    stacks: Dict[str, List[np.ndarray]] = {}
    patient_label: Dict[str, np.ndarray] = {}
    for pc, pred, lbl in zip(patient_codes, preds, labels):
        pc = str(pc)
        stacks.setdefault(pc, []).append(np.asarray(pred, dtype=np.float64))
        patient_label[pc] = np.asarray(lbl, dtype=np.float64)
    return {pc: np.mean(v, axis=0) for pc, v in stacks.items()}, patient_label


def _roll_preds_to_patients(patient_codes: np.ndarray, preds: np.ndarray, *, vector: bool):
    """Mean slide-prediction per patient, without labels, for unlabeled inference (``bin/predict``).
    ``vector`` selects the (N, K|G) mean-over-slides path (multiclass / expression) vs the scalar mean.
    The labelled analogues (``_roll_slides_to_patients`` / ``_vec``) also return the patient's label."""
    stacks: Dict[str, List] = {}
    for pc, pred in zip(patient_codes, preds):
        pc = str(pc)
        stacks.setdefault(pc, []).append(np.asarray(pred, dtype=np.float64) if vector else float(pred))
    if vector:
        return {pc: np.mean(v, axis=0) for pc, v in stacks.items()}
    return {pc: float(np.mean(v)) for pc, v in stacks.items()}


# -- scalar aggregation core (binary and regression) -------------------------
# These take the endpoint `handler` as a parameter so this leaf never imports the driver's registry.
def _per_outer_metrics(
    field: str, patient_preds: List[PatientPrediction], endpoint_type: Objective, handler: "_EndpointHandler"
) -> List[OuterFoldMetric]:
    by_outer: Dict[int, tuple[List[float], List[float]]] = {}
    for p in patient_preds:
        probs, labels = by_outer.setdefault(p.outer_fold, ([], []))
        probs.append(p.ensemble_prob)
        labels.append(p.label)
    dtype = np.float64 if endpoint_type == Objective.regression else np.int64
    out: List[OuterFoldMetric] = []
    for o in sorted(by_outer):
        probs, labels = by_outer[o]
        labels_arr = np.asarray(labels, dtype=dtype)
        n_pos = 0 if endpoint_type == Objective.regression else int(labels_arr.sum())  # not meaningful for regression
        out.append(OuterFoldMetric(field, o, len(labels), n_pos, handler.metrics(labels_arr, np.asarray(probs), 1)))
    return out


def _mean_std_across_folds(
    per_outer: List[OuterFoldMetric], handler: "_EndpointHandler"
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Mean and (population) std of each metric across outer folds, ignoring NaNs."""
    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}
    for key in handler.metric_labels:
        vals = np.asarray([m.metrics[key] for m in per_outer], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        mean[key] = float(finite.mean()) if finite.size else float("nan")
        std[key] = float(finite.std(ddof=0)) if finite.size else float("nan")
    return mean, std


def _aggregate_field(
    field: str, outer_to_reps: Dict[int, List[tuple[int, Path]]], handler: "_EndpointHandler"
) -> AggregateResult:
    """Shared scalar core for binary and regression: ensemble inner replicates -> patient preds -> per-
    outer + pooled metrics via ``handler.metrics``. Multiclass/survival/expression have their own cores
    (vector / coupled-label / per-gene shapes)."""
    endpoint_type = ""
    patient_preds: List[PatientPrediction] = []
    inner_coverage: Dict[int, int] = {}

    for outer in sorted(outer_to_reps):
        reps = sorted(outer_to_reps[outer])
        per_patient_probs: Dict[str, List[float]] = {}
        per_patient_label: Dict[str, int] = {}
        replicate_patient_sets: List[set] = []

        for _inner, npz in reps:
            data = load_fold_predictions(npz)
            endpoint_type = endpoint_type or data["endpoint_type"]
            patient_probs, patient_label = _roll_slides_to_patients(
                data["patient_codes"], data["probs"], data["labels"]
            )
            replicate_patient_sets.append(set(patient_probs))
            for pc, prob in patient_probs.items():
                per_patient_probs.setdefault(pc, []).append(prob)
                per_patient_label[pc] = patient_label[pc]

        # shared-TEST invariant: every inner replicate of an outer fold covers the same patients.
        first = replicate_patient_sets[0]
        for other in replicate_patient_sets[1:]:
            if other != first:
                raise ValueError(
                    f"shared-TEST invariant violated for field '{field}' outer fold {outer}: "
                    f"inner replicates cover different patient sets (symmetric diff: {sorted(first ^ other)})"
                )

        inner_coverage[outer] = len(reps)
        for pc, probs in per_patient_probs.items():
            patient_preds.append(
                PatientPrediction(field, outer, pc, float(np.mean(probs)), per_patient_label[pc], len(probs))
            )

    per_outer = _per_outer_metrics(field, patient_preds, endpoint_type, handler)
    mean, std = _mean_std_across_folds(per_outer, handler)
    pooled_dtype = np.float64 if endpoint_type == Objective.regression else np.int64
    pooled_labels = np.asarray([p.label for p in patient_preds], dtype=pooled_dtype)
    pooled_probs = np.asarray([p.ensemble_prob for p in patient_preds], dtype=np.float64)
    return AggregateResult(
        field=field,
        endpoint_type=endpoint_type,
        patient_predictions=patient_preds,
        per_outer=per_outer,
        mean=mean,
        std=std,
        pooled=handler.metrics(pooled_labels, pooled_probs, 1),
        pooled_n=int(pooled_labels.size),
        pooled_positive=0 if endpoint_type == Objective.regression else int(pooled_labels.sum()),
        n_outer_folds=len(inner_coverage),
        inner_coverage=inner_coverage,
    )


# -- shared figures + formatting ---------------------------------------------
def confusion_figure(labels: np.ndarray, preds: np.ndarray, num_classes: int, title: str, path: Path) -> None:
    """Row-normalized confusion matrix (recall per true class), cells annotated with the raw count.
    Rows = true class, columns = predicted. Shared by the internal (aggregate) + external scorers."""
    plt = _agg_pyplot()

    k = num_classes or 0
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    counts = np.zeros((k, k), dtype=np.int64)
    for t, pr in zip(labels, preds):
        counts[t, pr] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    norm = np.divide(counts, row_sums, out=np.zeros((k, k), dtype=np.float64), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(0.9 * k + 2.0, 0.9 * k + 2.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    for t in range(k):
        for pr in range(k):
            ax.text(
                pr,
                t,
                f"{counts[t, pr]}\n{norm[t, pr]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if norm[t, pr] > 0.5 else "black",
            )
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized (recall)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _fmt(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


# -- shared handler defaults -------------------------------------------------
def _no_summary_extra(res: AggregateResult) -> List[str]:
    return []


def _no_section_extra(result: "ExternalResult") -> List[str]:
    return []


def _no_scalar_metrics(labels: np.ndarray, values: np.ndarray, num_classes: int) -> Dict[str, float]:
    raise NotImplementedError("expression (regression_vector) computes per-gene metrics in its own core.")


def _no_patient_row(p: PatientPrediction) -> tuple:
    raise NotImplementedError("expression stores no per-patient scalar prediction.")


# -- external label model ----------------------------------------------------
# The label is not always a class index. Each external handler says how to turn the per-patient labels
# into the array its metric fn wants, what "missing" means for it, and how to count positives -- because
# "negative means missing" and "positives are the sum" are classification rules, not universal ones.
def _class_index_labels(values: List[Any]) -> np.ndarray:
    """(N,) int64 class indices, binary and multiclass."""
    return np.asarray(values, dtype=np.int64)


def _time_event_labels(values: List[Any]) -> np.ndarray:
    """(N, 2) float64 ``[time, event]``, survival's coupled label."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"survival labels must be (N, 2) [time, event], got shape {arr.shape}.")
    return arr


# Each rule below mirrors that objective's declared ``OBJECTIVE_TABLE[...].missing_sentinel``, -1 for
# the categorical objectives, NaN for the continuous ones. Do not invent extra conditions here: this
# guard exists to catch "the all_test split failed to filter missing labels", not to police data quality,
# and an undeclared rule would reject cohorts the contract considers valid.
def _negative_is_missing(label: Any) -> bool:
    """Categorical sentinel (-1). Any negative class index is invalid, so the test is `< 0` rather than
    an exact match. Not reusable for regression, whose targets may legitimately be negative."""
    return float(label) < 0


def _nan_time_is_missing(label: Any) -> bool:
    """Survival's sentinel is NaN, carried on the follow-up time (component 0 of the coupled label)."""
    return not np.isfinite(float(np.asarray(label, dtype=np.float64).ravel()[0]))


def _nan_is_missing(label: Any) -> bool:
    """Continuous sentinel (NaN). A negative value is data, bounds are a per-field property, and this
    handler serves the whole regression objective."""
    return not np.isfinite(float(label))


def _all_nan_is_missing(label: Any) -> bool:
    """Continuous sentinel for a vector label (regression_vector): missing means the patient has no
    measurement at all. Individual NaN genes are legitimate (the loss masks them per gene), so the test
    is all-NaN, not any-NaN."""
    return bool(np.all(~np.isfinite(np.asarray(label, dtype=np.float64))))


def _float_labels(values: List[Any]) -> np.ndarray:
    """(N,) float64 raw targets, regression. Also (N, G) for a vector label; np.asarray handles both."""
    return np.asarray(values, dtype=np.float64)


def _sum_positive(labels: np.ndarray) -> int:
    return int(labels.sum())


def _count_events(labels: np.ndarray) -> int:
    """Survival's 'positive' is an observed event, column 1 of the coupled label."""
    return int(labels[:, 1].sum())


def _roll_external(
    patient_codes: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    *,
    vector_preds: bool,
    vector_labels: bool,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Mean model prediction per patient + that patient's label, with the two shapes chosen independently.

    Prediction shape and label shape are separate axes: multiclass has vector predictions and a scalar
    label, survival has a scalar risk and a ``(time, event)`` pair. Collapsing them onto one flag (as
    ``vector_probs`` alone would) makes survival unrepresentable."""
    stacks: Dict[str, List[Any]] = {}
    patient_label: Dict[str, Any] = {}
    for pc, pred, lbl in zip(patient_codes, preds, labels):
        pc = str(pc)
        stacks.setdefault(pc, []).append(np.asarray(pred, dtype=np.float64) if vector_preds else float(pred))
        patient_label[pc] = np.asarray(lbl, dtype=np.float64) if vector_labels else float(lbl)
    if vector_preds:
        return {pc: np.mean(v, axis=0) for pc, v in stacks.items()}, patient_label
    return {pc: float(np.mean(v)) for pc, v in stacks.items()}, patient_label


# -- handler types (one registry entry per subsystem; deliberately not merged) ------------------------
@dataclass(frozen=True)
class _EndpointHandler:
    """Aggregate-reporting behaviour for one endpoint kind: metric labels + fn, summary footer/extra,
    per-field artifacts, headline, per-patient CSV row."""

    metric_labels: Dict[str, str]
    footer: str
    has_positive_count: bool  # classification reports a per-fold positive count; scalar/vector do not
    write_artifacts: Callable[[AggregateResult, Path, Path], None]  # (res, out_dir, figures_dir)
    headline: Callable[[AggregateResult], str]  # one-line stdout summary
    metrics: Callable[[np.ndarray, np.ndarray, int], Dict[str, float]]  # (labels, values, num_classes) -> metric dict
    patient_row: Callable[[PatientPrediction], tuple]  # (prediction_cell, label_cell) for the per-patient CSV
    strip_group: str  # which combined cross-field per-fold strip figure this endpoint joins
    summary_extra: Callable[[AggregateResult], List[str]] = _no_summary_extra


@dataclass(frozen=True)
class _ExternalHandler:
    """External-scoring behaviour for one endpoint kind: metric fn + prediction/label shapes, per-patient
    CSV, figure, section extra. Distinct from ``_EndpointHandler`` (a separate subsystem, not merged)."""

    metric_labels: Dict[str, str]
    metrics: Callable[[np.ndarray, np.ndarray, int], Dict[str, float]]  # (labels, values, num_classes)
    vector_probs: bool  # per-patient prediction is a (K,) vector (multiclass) vs a scalar
    has_positive_count: bool  # whether the report shows a "(n positive)" count at all
    write_patient_csv: Callable[[List["ExternalResult"], Path], None]  # per-endpoint-type CSV strategy
    write_artifact: Callable[["ExternalResult", Path], None]  # per-field figure (ROC / confusion / KM)
    section_extra: Callable[["ExternalResult"], List[str]] = _no_section_extra
    # Which combined cross-field per-outer strip this endpoint joins, and the metric it contributes.
    # Mirrors ``_EndpointHandler.strip_group``: AUROC and C-index are both [0,1] with chance at 0.5, but
    # they are not the same quantity and must not share an axis.
    strip_group: str = "classification"
    strip_metric: str = "auroc"
    # -- label model (see "external label model" above). Defaults are the classification rules, which
    # binary and multiclass share; anything with a non-index label overrides all three.
    vector_labels: bool = False  # per-patient label is a vector ((time, event), per-gene) vs a scalar
    label_array: Callable[[List[Any]], np.ndarray] = _class_index_labels
    is_missing: Callable[[Any], bool] = _negative_is_missing
    count_positive: Callable[[np.ndarray], int] = _sum_positive
    # Scalar labels arrive off the wire as floats. This says what one IS, which is what the per-patient
    # CSVs write: a class index for classification, the raw value for regression. Unused when
    # ``vector_labels`` (the vector keeps its float dtype).
    scalar_label_cast: Callable[[float], Any] = int
