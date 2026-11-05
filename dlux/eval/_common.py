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
by both drivers (``aggregate.py`` / ``external.py``). Result models, the NPZ loader, the slide→patient
roll, the out-of-fold patient table, the one aggregation core, the shared matplotlib init, the confusion
figure, and the two handler types.

Import direction (must stay acyclic): objective modules and drivers import down from here. ``_common``
imports nothing from ``eval`` except the pure-math leaves. The aggregation cores take the endpoint
``handler`` as a parameter, so this leaf never imports the driver's registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypedDict

import numpy as np

from dlux.config.cohort import Objective

if TYPE_CHECKING:
    from dlux.eval.external import ExternalResult


def _agg_pyplot():
    """Return a headless (Agg) ``pyplot``. Import is lazy (figures are optional) and the module cache
    makes repeat calls fast."""
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
    # The pooled out-of-fold patient table (see ``PatientTable``). For expression the columns of the (N, G)
    # matrices share the order of ``pooled_gene_pearson``, so they join to the cohort's genes.csv.
    patient_codes: Optional[List[str]] = None
    pooled_outer_folds: Optional[List[int]] = None  # (N,) outer fold each patient's OOF prediction came from
    pooled_preds: Optional[np.ndarray] = None  # (N,), or (N, K|G) for a vector prediction
    pooled_labels: Optional[np.ndarray] = None  # (N,), or (N, 2|G) for a vector label, aligned to pooled_preds
    num_classes: Optional[int] = None  # K, multiclass endpoint only
    pooled_per_class: Optional[List[dict]] = None  # per-class pooled AUROC/precision/recall/f1/support, multiclass
    coverage: Optional[dict] = None  # {declared, scored, lost} TEST patients; None = coverage was not recorded


# -- IO + rollups ------------------------------------------------------------
class FoldPredictions(TypedDict):
    """One fold's persisted predictions, as dlux.eval.predictions writes them."""

    patient_codes: np.ndarray
    probs: np.ndarray
    labels: np.ndarray
    endpoint_type: Objective


def load_fold_predictions(npz_path: str | Path) -> FoldPredictions:
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


def roll_slides_to_patients(
    patient_codes: np.ndarray,
    preds: np.ndarray,
    labels: Optional[np.ndarray] = None,
    *,
    vector_preds: bool,
    vector_labels: bool = False,
) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """A patient's prediction is the mean of its slides' predictions. Returns the prediction per patient and
    the label per patient, or None for the labels when ``labels`` is None (unlabelled inference).

    Prediction shape and label shape are separate axes: multiclass has vector predictions and a scalar
    label, survival has a scalar risk and a ``(time, event)`` pair, expression has vectors for both."""
    stacks: Dict[str, List[Any]] = {}
    patient_label: Optional[Dict[str, Any]] = None if labels is None else {}
    for i, (pc, pred) in enumerate(zip(patient_codes, preds)):
        pc = str(pc)
        stacks.setdefault(pc, []).append(np.asarray(pred, dtype=np.float64) if vector_preds else float(pred))
        if patient_label is not None:
            label = labels[i]  # type: ignore[index]
            patient_label[pc] = np.asarray(label, dtype=np.float64) if vector_labels else float(label)
    if vector_preds:
        return {pc: np.mean(v, axis=0) for pc, v in stacks.items()}, patient_label
    return {pc: float(np.mean(v)) for pc, v in stacks.items()}, patient_label


def ensemble_models(
    per_model: Dict[Any, Dict[str, Any]], models: List[Any], patients: List[str], *, vector: bool
) -> Dict[str, Any]:
    """The mean of ``models``' per-patient predictions, for each of ``patients``. Every model must have
    predicted every patient."""
    if vector:
        return {
            p: np.mean(np.stack([np.asarray(per_model[m][p], dtype=np.float64) for m in models]), axis=0)
            for p in patients
        }
    return {p: float(np.mean([per_model[m][p] for m in models])) for p in patients}


def require_same_patients(name: str, per_model: Dict[Any, Dict[str, Any]]) -> List[str]:
    """The patients every model predicted, sorted. Raises when two models predicted different patients: every
    model of a sweep runs over the same split, so a difference means a model saw other data."""
    if not per_model:
        raise ValueError(f"[{name}] no models to ensemble.")
    sets = {key: set(predictions) for key, predictions in per_model.items()}
    first_key, first = next(iter(sets.items()))
    differing = sorted(key for key, patients in sets.items() if patients != first)
    if differing:
        raise ValueError(
            f"[{name}] models {differing} predict other patients than model {first_key} ({len(first)} patients)."
        )
    return sorted(first)


@dataclass
class PatientTable:
    """A sweep's out-of-fold predictions, one row per patient: the outer fold that tested it, its prediction
    ensembled over that fold's inner replicates, and its label. Rows are grouped by outer fold, in fold
    order, and sorted by patient code within a fold."""

    codes: List[str]
    outer: np.ndarray  # (N,) int
    preds: np.ndarray  # (N,), or (N, K|G) for a vector prediction
    labels: np.ndarray  # (N,), or (N, 2|G) for a vector label
    n_replicates: np.ndarray  # (N,) int, the inner replicates ensembled into each row


def pool_out_of_fold(
    field: str, outer_to_reps: Dict[int, List[tuple[int, Path]]], *, vector_preds: bool, vector_labels: bool
) -> tuple[PatientTable, Dict[int, int]]:
    """Roll each fold's slides to patients, ensemble every outer fold's inner replicates (the mean), and
    stack the outer folds. Returns the table and the number of inner replicates per outer fold.

    The inner replicates of an outer fold share its TEST split, so they must score the same patients.
    A difference raises."""
    codes: List[str] = []
    outer: List[int] = []
    preds: List[Any] = []
    labels: List[Any] = []
    n_replicates: List[int] = []
    inner_coverage: Dict[int, int] = {}
    for o in sorted(outer_to_reps):
        reps = sorted(outer_to_reps[o])
        per_patient: Dict[str, List[Any]] = {}
        label_of: Dict[str, Any] = {}
        replicate_sets: List[set] = []
        for _inner, npz in reps:
            data = load_fold_predictions(npz)
            if vector_preds and data["probs"].ndim != 2:
                raise ValueError(f"{npz} has {data['probs'].ndim}-D predictions; expected 2-D (N, K).")
            if vector_labels and data["labels"].ndim != 2:
                raise ValueError(f"{npz} has labels of shape {data['labels'].shape}; expected 2-D.")
            rolled, rolled_label = roll_slides_to_patients(
                data["patient_codes"],
                data["probs"],
                data["labels"],
                vector_preds=vector_preds,
                vector_labels=vector_labels,
            )
            assert rolled_label is not None
            replicate_sets.append(set(rolled))
            for pc, pred in rolled.items():
                per_patient.setdefault(pc, []).append(pred)
                label_of[pc] = rolled_label[pc]
        first = replicate_sets[0]
        for other in replicate_sets[1:]:
            if other != first:
                raise ValueError(
                    f"shared-TEST invariant violated for field '{field}' outer fold {o}: "
                    f"inner replicates cover different patient sets (symmetric diff: {sorted(first ^ other)})"
                )
        inner_coverage[o] = len(reps)
        for pc in sorted(per_patient):
            codes.append(pc)
            outer.append(o)
            preds.append(np.mean(per_patient[pc], axis=0) if vector_preds else float(np.mean(per_patient[pc])))
            labels.append(label_of[pc])
            n_replicates.append(len(per_patient[pc]))
    if not codes:
        raise ValueError(f"[{field}] has no folds to aggregate.")
    table = PatientTable(
        codes=codes,
        outer=np.asarray(outer, dtype=np.int64),
        preds=np.asarray(preds, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.float64),
        n_replicates=np.asarray(n_replicates, dtype=np.int64),
    )
    return table, inner_coverage


def write_per_patient_npz(result: AggregateResult, path: str | Path, *, dtypes: tuple[type, type]) -> None:
    """The pooled out-of-fold patient table: ``patient_codes``, ``outer_folds``, ``preds``, ``labels`` and
    ``endpoint_type``. This file is the record every reader of aggregate results reads. The CSVs beside it
    are for people only. ``dtypes`` are the (prediction, label) dtypes the objective stores."""
    if result.patient_codes is None or result.pooled_preds is None or result.pooled_labels is None:
        raise ValueError(f"[{result.field}] has no pooled patient table to write.")
    pred_dtype, label_dtype = dtypes
    np.savez_compressed(
        path,
        patient_codes=np.asarray(result.patient_codes),
        outer_folds=np.asarray(result.pooled_outer_folds, dtype=np.int64),
        preds=result.pooled_preds.astype(pred_dtype),
        labels=result.pooled_labels.astype(label_dtype),
        endpoint_type=np.asarray(result.endpoint_type.value),
    )


def read_per_patient_npz(path: str | Path) -> PatientTable:
    """The table ``write_per_patient_npz`` wrote, widened to float64. ``n_replicates`` is not stored."""
    with np.load(path, allow_pickle=False) as data:
        codes = [str(c) for c in data["patient_codes"]]
        return PatientTable(
            codes=codes,
            outer=np.asarray(data["outer_folds"], dtype=np.int64),
            preds=np.asarray(data["preds"], dtype=np.float64),
            labels=np.asarray(data["labels"], dtype=np.float64),
            n_replicates=np.zeros(len(codes), dtype=np.int64),
        )


def _mean_std_across_folds(
    per_outer: List[OuterFoldMetric], handler: "_EndpointHandler"
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Mean and (population) std of each metric across outer folds, ignoring NaNs."""
    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}
    for key in handler.model.metric_labels:
        vals = np.asarray([m.metrics[key] for m in per_outer], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        mean[key] = float(finite.mean()) if finite.size else float("nan")
        std[key] = float(finite.std(ddof=0)) if finite.size else float("nan")
    return mean, std


def aggregate_field(
    field: str,
    endpoint_type: Objective,
    outer_to_reps: Dict[int, List[tuple[int, Path]]],
    handler: "_EndpointHandler",
    random_reps: Optional[Dict[int, List[tuple[int, Path]]]] = None,
) -> AggregateResult:
    """One endpoint's out-of-fold report: the patient table (``pool_out_of_fold``), scored per outer fold and
    over the pooled set with ``handler.model.metrics``. The handler's label model says what the predictions and
    labels look like, and its ``finish`` hook adds what only its objective reports. ``random_reps`` is a
    random-baseline sweep's folds, for a handler that uses one."""
    table, inner_coverage = pool_out_of_fold(
        field, outer_to_reps, vector_preds=handler.model.vector_preds, vector_labels=handler.model.vector_labels
    )

    def _score(rows: np.ndarray) -> tuple[Dict[str, float], int, int]:
        labels = handler.model.label_array(table.labels[rows])
        preds = table.preds[rows]
        num_classes = preds.shape[1] if preds.ndim == 2 and not handler.model.vector_labels else 1
        n_positive = handler.model.count_positive(labels) if handler.model.has_positive_count else 0
        return handler.model.metrics(labels, preds, num_classes), int(labels.shape[0]), n_positive

    per_outer: List[OuterFoldMetric] = []
    for o in sorted(inner_coverage):
        metrics, n, n_positive = _score(table.outer == o)
        per_outer.append(OuterFoldMetric(field, o, n, n_positive, metrics))
    pooled, pooled_n, pooled_positive = _score(np.ones(len(table.codes), dtype=bool))
    mean, std = _mean_std_across_folds(per_outer, handler)
    patient_predictions = [
        prediction
        for i, code in enumerate(table.codes)
        if (
            prediction := handler.patient_prediction(
                field, int(table.outer[i]), code, table.preds[i], table.labels[i], int(table.n_replicates[i])
            )
        )
        is not None
    ]
    result = AggregateResult(
        field=field,
        endpoint_type=endpoint_type,
        patient_predictions=patient_predictions,
        per_outer=per_outer,
        mean=mean,
        std=std,
        pooled=pooled,
        pooled_n=pooled_n,
        pooled_positive=pooled_positive,
        n_outer_folds=len(inner_coverage),
        inner_coverage=inner_coverage,
        patient_codes=list(table.codes),
        pooled_outer_folds=[int(o) for o in table.outer],
        pooled_preds=table.preds,
        pooled_labels=table.labels,
    )
    random_table = None
    if random_reps is not None:
        random_table, _ = pool_out_of_fold(
            field, random_reps, vector_preds=handler.model.vector_preds, vector_labels=handler.model.vector_labels
        )
    handler.finish(result, table, random_table)
    return result


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


def write_scalar_patient_csv(res: AggregateResult, path: Path, *, label_cast: Callable[[float], Any]) -> None:
    """The per-patient CSV, for people, of an objective with a scalar prediction and a scalar label: one row
    per patient with its outer fold, prediction, label and the number of inner replicates ensembled."""
    import csv

    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_code", "outer_fold", "prediction", "label", "n_replicates"])
        for p in res.patient_predictions:
            w.writerow([p.patient_code, p.outer_fold, f"{p.ensemble_prob:.6f}", label_cast(p.label), p.n_replicates])


def _no_finish(result: AggregateResult, table: PatientTable, random_table: Optional[PatientTable]) -> None:
    return None


def _scalar_patient_prediction(
    field: str, outer: int, code: str, pred: Any, label: Any, n_replicates: int
) -> Optional[PatientPrediction]:
    """A scalar prediction and a scalar label (binary, regression)."""
    return PatientPrediction(field, outer, code, float(pred), float(label), n_replicates)


# -- external label model ----------------------------------------------------
# The label is not always a class index. Each external handler says how to turn the per-patient labels
# into the array its metric fn wants, what "missing" means for it, and how to count positives -- because
# "negative means missing" and "positives are the sum" are classification rules, not universal ones.
def _class_index_labels(values: List[Any] | np.ndarray) -> np.ndarray:
    """(N,) int64 class indices, binary and multiclass."""
    return np.asarray(values, dtype=np.int64)


def _time_event_labels(values: List[Any]) -> np.ndarray:
    """(N, 2) float64 ``[time, event]``, survival's coupled label."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"survival labels must be (N, 2) [time, event], got shape {arr.shape}.")
    return arr


# Each rule below is that objective's missing-label sentinel: -1 for the categorical objectives, NaN for
# the continuous ones. Do not invent extra conditions here: this
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


# -- one objective's label model, and the two reports built on it ------------------------------------------
@dataclass(frozen=True)
class LabelModel:
    """What one objective's per-patient predictions and labels look like, and how they are scored.

    Aggregate and external scoring report differently, but they score the same objective, so both handlers
    take this one model and restate none of it."""

    metric_labels: Dict[str, str]
    metrics: Callable[[np.ndarray, np.ndarray, int], Dict[str, float]]  # (labels, values, num_classes) -> metrics
    has_positive_count: bool  # whether reports show a positive count (classes, events) at all
    vector_preds: bool = False  # per-patient prediction is a vector (multiclass softmax, expression genes)
    vector_labels: bool = False  # per-patient label is a vector ((time, event), expression genes)
    # The defaults below are the classification rules, which binary and multiclass share.
    label_array: Callable[[Any], np.ndarray] = _class_index_labels  # the labels as ``metrics`` takes them
    is_missing: Callable[[Any], bool] = _negative_is_missing  # the objective's missing-label sentinel
    count_positive: Callable[[np.ndarray], int] = _sum_positive  # used when ``has_positive_count``
    # What one scalar label is, for the per-patient CSVs: a class index for classification, the raw value
    # for regression. Unused when ``vector_labels``.
    scalar_label_cast: Callable[[float], Any] = int


@dataclass(frozen=True)
class _EndpointHandler:
    """Aggregate reporting for one objective: summary footer, per-field artifacts (figures and the per-patient
    CSV), headline, and what ``aggregate_field`` adds for this objective only."""

    model: LabelModel
    footer: str
    write_artifacts: Callable[[AggregateResult, Path, Path], None]  # (res, out_dir, figures_dir)
    headline: Callable[[AggregateResult], str]  # one-line stdout summary
    strip_group: str  # which combined cross-field per-fold strip figure this endpoint joins
    summary_extra: Callable[[AggregateResult], List[str]] = _no_summary_extra
    # One row of the per-patient report, or None for an objective that keeps its patients in the table only.
    patient_prediction: Callable[..., Optional[PatientPrediction]] = _scalar_patient_prediction
    # Adds what only this objective reports: (result, table, random-baseline table or None).
    finish: Callable[[AggregateResult, PatientTable, Optional[PatientTable]], None] = _no_finish
    uses_random_baseline: bool = False  # takes a random-baseline sweep (expression's SEQUOIA count)
    npz_dtypes: tuple[type, type] = (np.float64, np.float64)  # (prediction, label) in per_patient_<field>.npz


@dataclass(frozen=True)
class _ExternalHandler:
    """External-scoring reporting for one objective: per-patient CSV, figure, section extra, and the
    combined per-outer strip it joins."""

    model: LabelModel
    write_patient_csv: Callable[[List["ExternalResult"], Path], None]  # per-endpoint-type CSV strategy
    write_artifact: Callable[["ExternalResult", Path], None]  # per-field figure (ROC / confusion / KM)
    section_extra: Callable[["ExternalResult"], List[str]] = _no_section_extra
    # Which combined cross-field per-outer strip this endpoint joins, and the metric it contributes. AUROC
    # and C-index are both [0,1] with chance at 0.5, but they are not the same quantity and must not share
    # an axis.
    strip_group: str = "classification"
    strip_metric: str = "auroc"
