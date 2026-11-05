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
"""Survival endpoint (coupled ``(time, event)`` label, risk score): the label model (scalar risk, coupled label,
observed events as the positive count) scored by Harrell's C-index, Kaplan-Meier-by-risk figure, per-patient
CSV, headline, and both handlers, ``AGGREGATE_HANDLER`` (internal reporting) and ``EXTERNAL_HANDLER`` (scoring a
held-out cohort). The C-index and Kaplan-Meier math lives in the pure-math leaf ``survival_metrics``. Imports
down from ``_common`` + ``survival_metrics``. ``ExternalResult`` is referenced only in annotations
(TYPE_CHECKING), so external.py can import this."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

from dlux.data import layout
from dlux.eval._common import (
    AggregateResult,
    LabelModel,
    PatientPrediction,
    _agg_pyplot,
    _count_events,
    _EndpointHandler,
    _ExternalHandler,
    _fmt,
    _nan_time_is_missing,
    _time_event_labels,
)
from dlux.eval.survival_metrics import SURVIVAL_METRIC_LABELS, concordance_index, kaplan_meier

if TYPE_CHECKING:
    from dlux.eval.external import ExternalResult

_SURVIVAL_FOOTER = (
    "_Harrell's C-index over comparable pairs (0.5 = chance); risk = -Σ survival mass, so higher risk "
    "should mean shorter survival. 'pos' = observed events (censored patients still count as the "
    "longer-survival side of a pair). mean±std is across outer folds; the pooled OOF C is the number to "
    "trust — per-fold C is noisy at these fold sizes._"
)


def _survival_patient_prediction(
    field: str, outer: int, code: str, pred: float, label: np.ndarray, n_replicates: int
) -> PatientPrediction:
    """The risk, the follow-up time and the event indicator."""
    return PatientPrediction(field, outer, code, float(pred), float(label[0]), n_replicates, event=float(label[1]))


def _write_survival_predictions_csv(res: AggregateResult, path: Path) -> None:
    """Per-patient pooled-OOF survival predictions: patient_code, outer_fold, risk, time, event."""
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_code", "outer_fold", "risk", "time", "event"])
        for p in res.patient_predictions:
            w.writerow([p.patient_code, p.outer_fold, f"{p.ensemble_prob:.6f}", f"{p.label:.6f}", int(p.event or 0)])


def km_by_risk_figure(
    risk: np.ndarray, time: np.ndarray, event: np.ndarray, title: str, stats: str, path: Path
) -> None:
    """Kaplan-Meier curves for the low- vs high-risk halves of a scored set (split at the median risk),
    the standard visual for 'does the risk score separate survival'. Descriptive, not a test. Shared by
    the internal (pooled OOF) and external (grand ensemble) scorers so both read identically."""
    plt = _agg_pyplot()
    risk = np.asarray(risk, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    if risk.size:
        median = float(np.median(risk))
        high = risk >= median
        for mask, label, color in ((~high, "low risk", "#1f77b4"), (high, "high risk", "#c44e52")):
            if not mask.any():
                continue
            steps, surv = kaplan_meier(time[mask], event[mask])
            ax.step(steps, surv, where="post", color=color, lw=2, label=f"{label} (n={int(mask.sum())})")
    ax.set(xlabel="time", ylabel="survival probability", ylim=(0.0, 1.02), title=title)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.grid(True, lw=0.3, alpha=0.4)
    ax.text(
        0.97,
        0.97,
        stats,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="grey"),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _artifacts_survival(res: AggregateResult, out_dir: Path, figures_dir: Path) -> None:
    _write_survival_predictions_csv(res, layout.per_patient_csv(out_dir, res.field))
    km_by_risk_figure(
        np.asarray([p.ensemble_prob for p in res.patient_predictions], dtype=np.float64),
        np.asarray([p.label for p in res.patient_predictions], dtype=np.float64),
        np.asarray([(p.event or 0.0) for p in res.patient_predictions], dtype=np.float64),
        f"{res.field}: KM by OOF risk (median split)",
        f"pooled C = {_fmt(res.pooled['c_index'])}\nn = {res.pooled_n}, events = {res.pooled_positive}",
        figures_dir / f"km_{res.field}.png",
    )


def _headline_survival(res: AggregateResult) -> str:
    return (
        f"[{res.field}] mean±std C-index = {res.mean['c_index']:.3f} ± {res.std['c_index']:.3f} "
        f"(pooled OOF {res.pooled['c_index']:.3f}; {res.pooled_positive} events / {res.pooled_n} patients; "
        f"{res.n_outer_folds} outer folds)"
    )


def _survival_metrics(labels: np.ndarray, values: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Harrell's C. ``labels`` is the (N, 2) [time, event] array the handler's ``label_array`` built;
    ``values`` are the ensembled risks."""
    return {"c_index": concordance_index(labels[:, 0], values, labels[:, 1])}


MODEL = LabelModel(
    SURVIVAL_METRIC_LABELS,
    _survival_metrics,
    has_positive_count=True,  # 'positive' is the observed-event count (fold power)
    vector_labels=True,  # the prediction is a scalar risk, but the label is the coupled (time, event) pair
    label_array=_time_event_labels,
    is_missing=_nan_time_is_missing,
    count_positive=_count_events,
)

AGGREGATE_HANDLER = _EndpointHandler(
    MODEL,
    _SURVIVAL_FOOTER,
    _artifacts_survival,
    _headline_survival,
    strip_group="survival",
    patient_prediction=_survival_patient_prediction,
)


# -- external scoring --------------------------------------------------------
def _external_arrays(result: "ExternalResult", prob_map: Dict[str, Any]) -> tuple:
    """(risk, time, event) over the external cohort's patients, in sorted-code order."""
    patients = sorted(result.labels)
    risk = np.asarray([float(prob_map[p]) for p in patients], dtype=np.float64)
    te = np.asarray([np.asarray(result.labels[p], dtype=np.float64) for p in patients], dtype=np.float64)
    return risk, te[:, 0], te[:, 1]


def _write_external_survival_patient_csv(results: List["ExternalResult"], out_dir: Path) -> None:
    """One CSV per field, for people: one row per (patient x model), model = each outer-fold ensemble
    (o0..oK) plus grand. time/event repeat per row (they are a property of the patient, not the model) so
    each row is self-contained."""
    for res in results:
        with layout.per_patient_csv(out_dir, res.field).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_code", "model", "risk", "time", "event"])
            for p in sorted(res.labels):
                time, event = (float(v) for v in np.asarray(res.labels[p], dtype=np.float64))
                for model, prob_map in [
                    *((f"o{o}", res.per_outer_patient_probs[o]) for o in res.outer_folds),
                    ("grand", res.grand_patient_probs),
                ]:
                    w.writerow([p, model, f"{float(prob_map[p]):.6f}", f"{time:.6f}", int(event)])


def _plot_external_km(result: "ExternalResult", figures_dir: Path) -> None:
    """KM by grand-ensemble risk on the external cohort, the external counterpart of the pooled-OOF KM.
    The per-outer ensembles are a stability diagnostic and would need K curves to show; the headline is
    the one that gets plotted."""
    risk, time, event = _external_arrays(result, result.grand_patient_probs)
    km_by_risk_figure(
        risk,
        time,
        event,
        f"{result.field}: KM by GRAND risk on {result.cohort} (median split)",
        f"C = {_fmt(result.grand['c_index'])}\nn = {result.n_patients}, events = {int(event.sum())}",
        figures_dir / f"km_{result.field}.png",
    )


EXTERNAL_HANDLER = _ExternalHandler(
    MODEL,
    write_patient_csv=_write_external_survival_patient_csv,
    write_artifact=_plot_external_km,
    strip_group="survival",
    strip_metric="c_index",
)
