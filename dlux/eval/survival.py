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
"""Survival endpoint (coupled ``(time, event)`` label, risk score): its own aggregation core scoring
Harrell's C-index, Kaplan-Meier-by-risk figure, per-patient CSV, headline, and both handlers,
``AGGREGATE_HANDLER`` (internal reporting) and ``EXTERNAL_HANDLER`` (scoring a held-out cohort).
The C-index math lives in the pure-math leaf ``survival_metrics``. This module supplies only the
reporting + the coupled-label core. Imports down from ``_common`` + ``survival_metrics``; the core
references the module-level ``AGGREGATE_HANDLER`` (resolved at call time) for metric labels.
``ExternalResult`` is referenced solely in annotations (TYPE_CHECKING), so external.py can import this."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import numpy as np

from dlux.config.cohort import Objective
from dlux.eval._common import (
    AggregateResult,
    OuterFoldMetric,
    PatientPrediction,
    _agg_pyplot,
    _count_events,
    _EndpointHandler,
    _ExternalHandler,
    _fmt,
    _mean_std_across_folds,
    _nan_time_is_missing,
    _time_event_labels,
    load_fold_predictions,
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


def _aggregate_field_survival(field: str, outer_to_reps: Dict[int, List[tuple[int, Path]]]) -> AggregateResult:
    """Survival aggregation: roll each slide's risk to the patient (mean), ensemble inner replicates
    (mean risk), and score Harrell's C per outer fold + over the pooled OOF. The persisted label is the
    ``(time, event)`` pair (probs = risk), so this core carries three per-patient arrays instead of the
    scalar core's (label, value), hence its own core, like multiclass/expression."""
    patient_preds: List[PatientPrediction] = []
    per_outer: List[OuterFoldMetric] = []
    inner_coverage: Dict[int, int] = {}
    pooled_risk: List[np.ndarray] = []
    pooled_time: List[np.ndarray] = []
    pooled_event: List[np.ndarray] = []

    for outer in sorted(outer_to_reps):
        reps = sorted(outer_to_reps[outer])
        per_patient_risk: Dict[str, List[float]] = {}
        per_patient_te: Dict[str, tuple[float, float]] = {}  # patient -> (time, event), constant per patient
        replicate_patient_sets: List[set] = []

        for _inner, npz in reps:
            data = load_fold_predictions(npz)
            labels = data["labels"]  # (N, 2) [time, event]
            if labels.ndim != 2 or labels.shape[1] != 2:
                raise ValueError(f"survival fold {npz} has labels shape {labels.shape}; expected (N, 2) [time, event].")
            slide_risk: Dict[str, List[float]] = {}
            te: Dict[str, tuple[float, float]] = {}
            for pc, risk, lbl in zip(data["patient_codes"], data["probs"], labels):
                pc = str(pc)
                slide_risk.setdefault(pc, []).append(float(risk))
                te[pc] = (float(lbl[0]), float(lbl[1]))
            rolled = {pc: float(np.mean(v)) for pc, v in slide_risk.items()}  # slide -> patient mean risk
            replicate_patient_sets.append(set(rolled))
            for pc, risk in rolled.items():
                per_patient_risk.setdefault(pc, []).append(risk)
                per_patient_te[pc] = te[pc]

        # shared-TEST invariant: every inner replicate of an outer fold covers the same patients.
        first = replicate_patient_sets[0]
        for other in replicate_patient_sets[1:]:
            if other != first:
                raise ValueError(
                    f"shared-TEST invariant violated for field '{field}' outer fold {outer}: "
                    f"inner replicates cover different patient sets (symmetric diff: {sorted(first ^ other)})"
                )

        inner_coverage[outer] = len(reps)
        pcs = sorted(per_patient_risk)
        risk_o = np.asarray([float(np.mean(per_patient_risk[pc])) for pc in pcs])
        time_o = np.asarray([per_patient_te[pc][0] for pc in pcs])
        event_o = np.asarray([per_patient_te[pc][1] for pc in pcs])
        per_outer.append(
            OuterFoldMetric(
                field, outer, len(pcs), int(event_o.sum()), {"c_index": concordance_index(time_o, risk_o, event_o)}
            )
        )
        for i, pc in enumerate(pcs):
            patient_preds.append(
                PatientPrediction(
                    field,
                    outer,
                    pc,
                    float(risk_o[i]),
                    float(time_o[i]),
                    len(per_patient_risk[pc]),
                    event=float(event_o[i]),
                )
            )
        pooled_risk.append(risk_o)
        pooled_time.append(time_o)
        pooled_event.append(event_o)

    all_risk = np.concatenate(pooled_risk)
    all_time = np.concatenate(pooled_time)
    all_event = np.concatenate(pooled_event)
    mean, std = _mean_std_across_folds(per_outer, AGGREGATE_HANDLER)
    return AggregateResult(
        field=field,
        endpoint_type=Objective.survival,
        patient_predictions=patient_preds,
        per_outer=per_outer,
        mean=mean,
        std=std,
        pooled={"c_index": concordance_index(all_time, all_risk, all_event)},
        pooled_n=int(all_time.size),
        pooled_positive=int(all_event.sum()),  # observed events over the pooled OOF
        n_outer_folds=len(inner_coverage),
        inner_coverage=inner_coverage,
    )


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
    _write_survival_predictions_csv(res, out_dir / f"predictions_{res.field}.csv")
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


def _survival_own_core_metrics(labels: np.ndarray, values: np.ndarray, num_classes: int) -> Dict[str, float]:
    raise NotImplementedError("survival computes C-index in its own core (needs time + event + risk, not a scalar).")


AGGREGATE_HANDLER = _EndpointHandler(
    SURVIVAL_METRIC_LABELS,
    _SURVIVAL_FOOTER,
    True,  # 'pos' column = observed events per fold (fold power)
    _artifacts_survival,
    _headline_survival,
    metrics=_survival_own_core_metrics,  # unused, the survival core scores C-index directly
    patient_row=lambda p: (f"{p.ensemble_prob:.6f}", f"{p.label:.6f}"),  # (risk, time); event in predictions CSV
    strip_group="survival",
)


# -- external scoring --------------------------------------------------------
def _external_arrays(result: "ExternalResult", prob_map: Dict[str, object]) -> tuple:
    """(risk, time, event) over the external cohort's patients, in sorted-code order."""
    patients = sorted(result.labels)
    risk = np.asarray([float(prob_map[p]) for p in patients], dtype=np.float64)
    te = np.asarray([np.asarray(result.labels[p], dtype=np.float64) for p in patients], dtype=np.float64)
    return risk, te[:, 0], te[:, 1]


def _survival_external_metrics(labels: np.ndarray, values: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Harrell's C over the external cohort. ``labels`` is the (N, 2) [time, event] array the handler's
    ``label_array`` built; ``values`` are the ensembled risks."""
    return {"c_index": concordance_index(labels[:, 0], values, labels[:, 1])}


def _write_external_survival_patient_csv(results: List["ExternalResult"], out_dir: Path) -> None:
    """Long-format per-patient risks across the survival endpoints: one row per (field x patient x
    model), model = each outer-fold ensemble (o0..oK) plus grand. time/event repeat per row (they are a
    property of the patient, not the model) so each row is self-contained."""
    with (out_dir / "per_patient_external_survival.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "patient_code", "model", "risk", "time", "event"])
        for res in results:
            for p in sorted(res.labels):
                time, event = (float(v) for v in np.asarray(res.labels[p], dtype=np.float64))
                for model, prob_map in [
                    *((f"o{o}", res.per_outer_patient_probs[o]) for o in res.outer_folds),
                    ("grand", res.grand_patient_probs),
                ]:
                    w.writerow([res.field, p, model, f"{float(prob_map[p]):.6f}", f"{time:.6f}", int(event)])


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
    SURVIVAL_METRIC_LABELS,
    _survival_external_metrics,
    vector_probs=False,  # the prediction is a scalar risk score...
    has_positive_count=True,  # ...and 'positive' is the observed-event count
    write_patient_csv=_write_external_survival_patient_csv,
    write_artifact=_plot_external_km,
    vector_labels=True,  # ...but the label is the coupled (time, event) pair
    label_array=_time_event_labels,
    is_missing=_nan_time_is_missing,
    count_positive=_count_events,
    strip_group="survival",
    strip_metric="c_index",
)
