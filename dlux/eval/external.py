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
"""External-cohort scoring: ensemble a development sweep's models over a held-out cohort.

Pure numpy (no torch/ahcore), so trivially testable. The inference driver
(``bin/evaluate_external``) runs each of the development sweep's K×J models over the
external cohort and rolls each model's slide-probs to patient level. This module then
ensembles those per-model patient-probs into the two numbers we report:

- **grand ensemble**, mean-of-probs over all K×J models → a single point estimate. This
  is the *headline*: it's the deployable model, and it has no leakage (no model saw the
  external cohort).
- **per-outer stability**, for each outer fold, mean over its J inner models → one score,
  reported as mean ± std across the K outer ensembles. This is a *stability diagnostic*
  ("does the result survive which outer split trained it"), not an uncertainty band, the
  K scores are the same external patients, so the spread is model-variability, not sampling
  variance. It must not be compared apples-to-apples with the internal per-outer band.

Endpoint-varying behaviour (metric fn, scalar-vs-vector prob shape, per-patient CSV, figure,
per-class table) is dispatched through ``_HANDLERS`` keyed on ``endpoint_type``, the same
per-subsystem registry pattern ``aggregate`` uses, so no function re-branches on the type.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from dlux.config.cohort import Objective
from dlux.data import layout
from dlux.eval._common import (
    LabelModel,
    _ExternalHandler,
    _fmt,
    ensemble_models,
    roll_slides_to_patients,
)
from dlux.eval.binary import EXTERNAL_HANDLER as _BINARY_EXTERNAL_HANDLER
from dlux.eval.multiclass import EXTERNAL_HANDLER as _MULTICLASS_EXTERNAL_HANDLER
from dlux.eval.regression import EXTERNAL_HANDLER as _REGRESSION_EXTERNAL_HANDLER
from dlux.eval.regression_vector import EXTERNAL_HANDLER as _EXPRESSION_EXTERNAL_HANDLER
from dlux.eval.survival import EXTERNAL_HANDLER as _SURVIVAL_EXTERNAL_HANDLER

logger = logging.getLogger(__name__)

# per-model patient predictions: {(outer, inner): {patient_code: prediction}}. The prediction is a float
# (binary, regression, survival risk) or an array (the multiclass softmax, the expression gene vector), and
# the endpoint handler interprets it.
PerModelProbs = Dict[Tuple[int, int], Dict[str, object]]


@dataclass
class ExternalResult:
    field: str
    cohort: str
    n_patients: int
    n_positive: int
    n_models: int
    grand: Dict[str, float]  # grand K×J ensemble metrics (headline)
    endpoint_type: Objective = Objective.binary
    num_classes: int = 1
    outer_folds: List[int] = dc_field(default_factory=list)  # sorted outer-fold ids (row labels)
    per_outer: List[Dict[str, float]] = dc_field(default_factory=list)  # per-outer-ensemble metrics
    per_outer_mean: Dict[str, float] = dc_field(default_factory=dict)  # stability band (not an uncertainty CI)
    per_outer_std: Dict[str, float] = dc_field(default_factory=dict)
    # Per-patient ensembled predictions: a scalar for binary, regression and survival (the risk), a (K,) vector
    # for multiclass, or a (G,) gene vector for expression. The handler interprets them.
    grand_patient_probs: Dict[str, Any] = dc_field(default_factory=dict)
    per_outer_patient_probs: Dict[int, Dict[str, Any]] = dc_field(default_factory=dict)
    # Per-patient label, shaped by the endpoint: a class index (binary/multiclass), a raw scalar
    # (regression), or a vector such as survival's (time, event). The handler interprets it.
    labels: Dict[str, Any] = dc_field(default_factory=dict)
    # Expression only: SEQUOIA's conservative significant-gene count vs a random-init model, when a
    # random-baseline arm was supplied. {n_significant, n_scored, alpha, fdr}; None otherwise.
    conservative: Any = None


def _mean_std(per_outer: List[Dict[str, float]], metric_labels: Dict[str, str]) -> Tuple[Dict, Dict]:
    """Mean and population std of each metric across outer-fold ensembles, ignoring NaNs."""
    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}
    for key in metric_labels:
        vals = np.asarray([m[key] for m in per_outer], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        mean[key] = float(finite.mean()) if finite.size else float("nan")
        std[key] = float(finite.std(ddof=0)) if finite.size else float("nan")
    return mean, std


def external_ensemble(
    field: str,
    cohort: str,
    per_model: PerModelProbs,
    labels: Dict[str, Any],
    *,
    endpoint_type: Objective = Objective.binary,
    num_classes: int = 1,
) -> ExternalResult:
    """Ensemble per-model patient-probs on the external cohort into grand + per-outer metrics.

    Every model must cover exactly the same external patients (the whole cohort). The endpoint
    handler supplies the metric fn and the prediction shape.
    """
    if not per_model:
        raise ValueError(f"[{field}/{cohort}] no development models supplied to ensemble")
    endpoint_type = Objective(endpoint_type)  # normalize str|Objective -> enum (stored on the result)

    # The all_test_<target> split filters to valid-label patients up front, so every scored patient
    # must carry a valid label here. A missing label is not something to quietly drop. It means the
    # cohort DB is stale or the wrong split was scored, so fail loudly rather than paper over it. What
    # counts as missing is the endpoint's business (a negative regression target is data, not a
    # sentinel), so the rule comes from the handler.
    handler = _handler(endpoint_type)
    patients = sorted(labels)
    invalid = [p for p in patients if handler.model.is_missing(labels[p])]
    if invalid:
        raise ValueError(
            f"[{field}/{cohort}] {len(invalid)} scored patient(s) carry a missing label "
            f"(e.g. {invalid[:8]}) — the all_test_{field} split must exclude these before inference. "
            f"Rebuild the cohort DB."
        )
    if not patients:
        raise ValueError(f"[{field}/{cohort}] no patients to score.")
    patient_set = set(patients)
    for (o, i), probs in per_model.items():
        if not patient_set <= set(probs):
            raise ValueError(
                f"[{field}/{cohort}] model o{o}_i{i} is missing scored patients "
                f"(absent: {sorted(patient_set - set(probs))[:8]})"
            )

    label_arr = handler.model.label_array([labels[p] for p in patients])
    outer_folds = sorted({outer for (outer, _inner) in per_model})

    def _ensemble(models: list) -> Dict[str, object]:
        return ensemble_models(per_model, models, patients, vector=handler.model.vector_preds)

    def _metrics(prob_map: Dict[str, Any]) -> Dict[str, float]:
        if handler.model.vector_preds:
            return handler.model.metrics(label_arr, np.stack([prob_map[p] for p in patients]), num_classes)
        return handler.model.metrics(
            label_arr, np.asarray([prob_map[p] for p in patients], dtype=np.float64), num_classes
        )

    grand_probs = _ensemble(list(per_model))
    grand = _metrics(grand_probs)

    per_outer: List[Dict[str, float]] = []
    per_outer_probs: Dict[int, Dict[str, object]] = {}
    for o in outer_folds:
        inner_models = [m for m in per_model if m[0] == o]
        o_prob_map = _ensemble(inner_models)
        per_outer_probs[o] = o_prob_map
        per_outer.append(_metrics(o_prob_map))
    mean, std = _mean_std(per_outer, handler.model.metric_labels)

    return ExternalResult(
        field=field,
        cohort=cohort,
        n_patients=len(patients),
        n_positive=handler.model.count_positive(label_arr) if handler.model.has_positive_count else 0,
        n_models=len(per_model),
        grand=grand,
        endpoint_type=endpoint_type,
        num_classes=num_classes,
        outer_folds=outer_folds,
        per_outer=per_outer,
        per_outer_mean=mean,
        per_outer_std=std,
        grand_patient_probs=grand_probs,
        per_outer_patient_probs=per_outer_probs,
        labels=labels,
    )


# -- reporting ---------------------------------------------------------------
def _field_section(result: ExternalResult) -> list[str]:
    """One markdown section for a field: per-outer rows (o0…oK) + stability + grand, plus any
    endpoint-specific extra (multiclass per-class table). Columns come from the endpoint handler."""
    handler = _handler(result.endpoint_type)
    labels = handler.model.metric_labels
    cols = list(labels.values())
    rows = ["| fold | " + " | ".join(cols) + " |", "|:--|" + "|".join(["--:"] * len(cols)) + "|"]
    for outer, m in zip(result.outer_folds, result.per_outer):
        rows.append(f"| o{outer} | " + " | ".join(_fmt(m[k]) for k in labels) + " |")
    rows.append(
        "| **stability (mean±std)** | "
        + " | ".join(f"{_fmt(result.per_outer_mean[k])}±{_fmt(result.per_outer_std[k])}" for k in labels)
        + " |"
    )
    rows.append("| **GRAND ensemble** | " + " | ".join(f"**{_fmt(result.grand[k])}**" for k in labels) + " |")
    positive = f" ({result.n_positive} positive)" if handler.model.has_positive_count else ""
    return [
        f"## `{result.field}`",
        "",
        f"{result.n_patients} patients{positive} | {result.n_models} development models",
        "",
        *rows,
        "",
        *handler.section_extra(result),
    ]


def write_external_npz(result: ExternalResult, path: str | Path) -> None:
    """One endpoint's external predictions, the record readers of external results read: ``patient_codes``,
    the grand-ensemble ``preds``, the ``labels`` and the ``endpoint_type``, plus each outer fold's ensemble
    (``ensemble_outer_folds`` and ``outer_preds``, one row per outer fold)."""
    patients = sorted(result.labels)

    def _stack(prob_map: Dict[str, Any]) -> np.ndarray:
        return np.asarray([np.asarray(prob_map[p], dtype=np.float64) for p in patients])

    np.savez_compressed(
        path,
        patient_codes=np.asarray(patients),
        preds=_stack(result.grand_patient_probs),
        labels=np.asarray([np.asarray(result.labels[p], dtype=np.float64) for p in patients]),
        endpoint_type=np.asarray(result.endpoint_type.value),
        ensemble_outer_folds=np.asarray(result.outer_folds, dtype=np.int64),
        outer_preds=np.stack([_stack(result.per_outer_patient_probs[o]) for o in result.outer_folds]),
    )


def write_external_report(results: list[ExternalResult], experiment_name: str, out_dir: str | Path) -> Path:
    """Write each endpoint's ``per_patient_<field>.npz`` (the record), its per-patient CSV and the per-fold
    CSV for people, summary.md and figures for all scored endpoints. The folder is written in full before it
    replaces the previous one (``layout.replaced_dir``).

    Mirrors ``dlux.eval.aggregate.write_reports``: a section per endpoint, a per-endpoint figure, and
    one combined per-outer strip per metric family. Per-patient CSV + figure are dispatched via ``_HANDLERS``."""
    out_dir = Path(out_dir)
    if not results:
        raise ValueError(f"No external results to write under {out_dir}.")
    with layout.replaced_dir(out_dir) as tmp:
        _write_external_report_into(results, experiment_name, tmp)
    logger.info("Wrote external report (%d endpoint(s)) -> %s", len(results), out_dir)
    return out_dir


def _write_external_per_fold_csv(results: list[ExternalResult], path: Path) -> None:
    """One row per (endpoint, outer-fold ensemble), every metric any endpoint reports as a column. A metric
    an endpoint does not report is left empty."""
    labels = list(dict.fromkeys(k for res in results for k in _handler(res.endpoint_type).model.metric_labels))
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "outer_fold", "n_patients", "n_positive", *labels])
        for res in results:
            for outer, metrics in zip(res.outer_folds, res.per_outer):
                values = (f"{metrics[k]:.6f}" if k in metrics else "" for k in labels)
                w.writerow([res.field, outer, res.n_patients, res.n_positive, *values])


def _write_external_report_into(results: list[ExternalResult], experiment_name: str, out_dir: Path) -> None:
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cohort = results[0].cohort

    # Per-patient CSV: each endpoint handler owns its columns. Group by type, hand each group to its handler.
    by_type: Dict[Objective, List[ExternalResult]] = {}
    for res in results:
        by_type.setdefault(res.endpoint_type, []).append(res)
    for endpoint_type, group in by_type.items():
        _handler(endpoint_type).write_patient_csv(group, out_dir)
    for res in results:
        write_external_npz(res, layout.per_patient_npz(out_dir, res.field))
    _write_external_per_fold_csv(results, layout.per_fold_metrics_csv(out_dir))

    lines = [f"# External validation — {experiment_name} on `{cohort}`", ""]
    for res in results:
        lines += _field_section(res)
    lines += [
        "_**GRAND ensemble** (mean-of-probs over all K×J models) is the HEADLINE — the deployable, "
        "leakage-free point estimate. The per-outer rows + **stability (mean±std)** are a "
        "*stability diagnostic* (does the result survive which outer split trained it), NOT an "
        "uncertainty interval: the K scores are the same external patients, so the spread is "
        "model-variability, and must not be compared apples-to-apples with the internal per-outer band._",
        "",
    ]
    layout.summary_md(out_dir).write_text("\n".join(lines))

    for res in results:
        _handler(res.endpoint_type).write_artifact(res, figures_dir)
    # One combined strip per metric family: AUROC and C-index are both [0,1] with chance at 0.5, but
    # they are not the same quantity and must not share an axis.
    by_strip: Dict[tuple, List[ExternalResult]] = {}
    for res in results:
        h = _handler(res.endpoint_type)
        if not h.strip_group:  # opted out. Its scores do not share an axis with anything (regression)
            continue
        by_strip.setdefault((h.strip_group, h.strip_metric), []).append(res)
    for (_group, metric), group_results in by_strip.items():
        _plot_external_strip(group_results, metric, figures_dir / f"{metric}_by_outer.png")


def _plot_external_strip(results: list[ExternalResult], metric: str, path: Path) -> None:
    """Combined strip: one column per endpoint, per-outer scores (stability) as points, the
    grand-ensemble score as a bar, chance dashed. One call per metric family (see write_external_report):
    ``auroc`` for binary/multiclass, ``c_index`` for survival."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(max(4.5, 1.7 * len(results)), 4.5))
    all_vals: list[float] = []
    for idx, res in enumerate(results):
        scores = np.asarray([m[metric] for m in res.per_outer if np.isfinite(m[metric])])
        if scores.size:
            jitter = np.linspace(-0.08, 0.08, scores.size)
            ax.scatter(np.full(scores.size, idx) + jitter, scores, s=40, alpha=0.85, color="C0", zorder=3)
            all_vals.extend(scores.tolist())
        grand = res.grand[metric]
        if np.isfinite(grand):
            ax.hlines(grand, idx - 0.24, idx + 0.24, colors="C3", lw=2.2, zorder=4)  # grand ensemble
            ax.text(idx + 0.28, grand, f"{grand:.3f}", va="center", fontsize=8, color="C3")
            all_vals.append(grand)
    ax.axhline(0.5, color="grey", lw=0.8, linestyle="--", alpha=0.7)  # chance

    handles = [
        Line2D([0], [0], marker="o", color="C0", linestyle="", label="per-outer ensemble"),
        Line2D([0], [0], color="C3", lw=2.2, label="GRAND ensemble"),
        Line2D([0], [0], color="grey", lw=0.8, linestyle="--", label="chance"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    y_lo = min(0.4, min(all_vals) - 0.03) if all_vals else 0.4
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([res.field for res in results])
    label = _handler(results[0].endpoint_type).model.metric_labels.get(metric, metric)
    ax.set(xlim=(-0.5, len(results) - 0.5 + 0.6), ylim=(max(0.0, y_lo), 1.01), ylabel=label)
    ax.set_title(f"External per-outer {label} — {results[0].cohort}")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -- endpoint handlers (dispatched on endpoint_type via _HANDLERS) ------------
# One handler per endpoint kind owns external eval's endpoint-varying behaviour: the metric fn, the
# per-patient prob shape, whether a positive count applies, the per-patient CSV strategy, the figure,
# and the summary's extra lines. external_ensemble + write_external_report look the handler up once.
# Each handler carries its own label model (shape, missing rule, positive count), so
# adding an objective here needs no change to the scoring core.
_HANDLERS: Dict[str, _ExternalHandler] = {
    "binary": _BINARY_EXTERNAL_HANDLER,
    "multiclass": _MULTICLASS_EXTERNAL_HANDLER,
    "regression": _REGRESSION_EXTERNAL_HANDLER,
    "regression_vector": _EXPRESSION_EXTERNAL_HANDLER,
    "survival": _SURVIVAL_EXTERNAL_HANDLER,
}


def supported_objectives() -> list[str]:
    """The objectives external eval can actually report on, read from the handler registry, so a caller
    cannot advertise a capability that is not wired."""
    return sorted(_HANDLERS)


def _handler(endpoint_type: Objective) -> _ExternalHandler:
    try:
        return _HANDLERS[endpoint_type]  # a str enum, so it keys the string-keyed registry
    except KeyError:
        raise NotImplementedError(f"external eval supports {sorted(_HANDLERS)} (got '{endpoint_type}').")


def label_model(endpoint_type: Objective) -> LabelModel:
    """What this objective's per-patient predictions and labels look like (see ``LabelModel``)."""
    return _handler(endpoint_type).model


def patient_arrays(
    endpoint_type: Objective, patient_codes: np.ndarray, preds: np.ndarray, labels: np.ndarray
) -> tuple[list[str], np.ndarray, np.ndarray, int]:
    """One model's per-slide predictions rolled to patients by the mean: ``(patients, labels, predictions,
    n_positive)``, sorted by patient. The handler's label model gives the label array and the positive
    count (0 where the endpoint has none)."""
    h = _handler(endpoint_type)
    rolled, patient_label = roll_slides_to_patients(
        patient_codes, preds, labels, vector_preds=h.model.vector_preds, vector_labels=h.model.vector_labels
    )
    assert patient_label is not None
    patients = sorted(rolled)
    y = h.model.label_array([patient_label[p] for p in patients])
    x = np.stack([rolled[p] for p in patients]) if h.model.vector_preds else np.asarray([rolled[p] for p in patients])
    return patients, y, x, h.model.count_positive(y) if h.model.has_positive_count else 0


def headline(result: ExternalResult) -> str:
    """One-line stdout summary on the endpoint's primary metric (``strip_metric``), AUROC for
    classification, C-index for survival, so the driver never names a metric an endpoint lacks."""
    h = _handler(result.endpoint_type)
    key = h.strip_metric
    label = h.model.metric_labels.get(key, key)
    return (
        f"External {result.field} on {result.cohort}: GRAND {label}={_fmt(result.grand[key])} "
        f"(stability {_fmt(result.per_outer_mean[key])}±{_fmt(result.per_outer_std[key])}, "
        f"{result.n_models} models)."
    )
