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
"""Binary endpoint: metric labels + fn, ROC figures, headline, and the two handlers the drivers
assemble into their registries (``AGGREGATE_HANDLER`` for internal reporting, ``EXTERNAL_HANDLER`` for
external scoring, binary is the only externally-scorable objective wired today).

Imports down from ``_common`` only; ``ExternalResult`` is referenced solely in annotations (TYPE_CHECKING)
so this module never takes a runtime edge to the ``external`` driver."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from dlux.eval._common import AggregateResult, _agg_pyplot, _EndpointHandler, _ExternalHandler
from dlux.eval.score_dist import plot_binary_score_distribution

if TYPE_CHECKING:
    from dlux.eval.external import ExternalResult

_THRESHOLD = 0.5

# Ordered metric keys → column labels for the summary table (also reused by external's binary handler).
BINARY_METRIC_LABELS: Dict[str, str] = {
    "auroc": "AUROC",
    "ap": "AP",
    "accuracy": "acc",
    "balanced_accuracy": "bal_acc",
    "sensitivity": "sens",
    "specificity": "spec",
    "precision": "prec",
    "npv": "npv",
    "f1": "F1",
    "mcc": "MCC",
    "kappa": "κ",
}

_BINARY_FOOTER = (
    "_Thresholded metrics (acc, bal\\_acc, sens, spec, prec, F1) at a FIXED threshold 0.5 — not tuned "
    "(tuning on the held-out folds would leak). AUROC/AP are threshold-free. mean±std is across outer "
    "folds; pooled OOF is computed on all out-of-fold patients._"
)


def binary_metrics(labels: np.ndarray, probs: np.ndarray, *, threshold: float = _THRESHOLD) -> Dict[str, float]:
    """Binary metrics from labels + positive-class probs. Threshold-free (AUROC/AP) + @threshold.

    Degenerate (single-class) sets yield NaN for the metrics that are undefined there.
    """
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    n = int(labels.size)
    n_pos = int(labels.sum())
    nan = float("nan")
    both_classes = 0 < n_pos < n

    preds = (probs >= threshold).astype(np.int64)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    sens = tp / (tp + fn) if (tp + fn) else nan
    spec = tn / (tn + fp) if (tn + fp) else nan
    # MCC (Matthews), guard a zero factor in the denominator.
    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom else nan
    # Cohen's kappa vs chance agreement (computed directly to avoid sklearn's degenerate warnings).
    p_o = (tp + tn) / n if n else nan
    p_e = (((tp + fp) * (tp + fn)) + ((tn + fn) * (tn + fp))) / (n * n) if n else nan
    kappa = (p_o - p_e) / (1 - p_e) if (n and (1 - p_e)) else nan
    return {
        "auroc": float(roc_auc_score(labels, probs)) if both_classes else nan,
        "ap": float(average_precision_score(labels, probs)) if both_classes else nan,
        "accuracy": (tp + tn) / n if n else nan,
        "balanced_accuracy": (sens + spec) / 2 if not (np.isnan(sens) or np.isnan(spec)) else nan,
        "sensitivity": sens,
        "specificity": spec,
        "precision": tp / (tp + fp) if (tp + fp) else nan,
        "npv": tn / (tn + fn) if (tn + fn) else nan,
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else nan,
        "mcc": mcc,
        "kappa": kappa,
    }


# -- internal (aggregate) reporting ------------------------------------------
def _plot_roc(result: AggregateResult, path: Path) -> None:
    """Per-outer-fold ROC curves + a mean±std band (TPR interpolated on a common FPR grid)."""
    from sklearn.metrics import roc_curve

    plt = _agg_pyplot()
    grid = np.linspace(0.0, 1.0, 201)
    by_outer: Dict[int, tuple[List[float], List[int]]] = {}
    for p in result.patient_predictions:
        probs, labels = by_outer.setdefault(p.outer_fold, ([], []))
        probs.append(p.ensemble_prob)
        labels.append(p.label)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    interp_tprs: List[np.ndarray] = []
    auroc_by_outer = {m.outer_fold: m.metrics["auroc"] for m in result.per_outer}
    for o in sorted(by_outer):
        probs, labels = by_outer[o]
        labels_arr = np.asarray(labels)
        if not (0 < labels_arr.sum() < len(labels_arr)):
            continue
        fpr, tpr, _ = roc_curve(labels_arr, np.asarray(probs))
        interp = np.interp(grid, fpr, tpr)
        interp[0] = 0.0
        interp_tprs.append(interp)
        ax.plot(fpr, tpr, lw=1.0, alpha=0.4, label=f"o{o} (AUROC={auroc_by_outer[o]:.3f})")

    if interp_tprs:
        stack = np.vstack(interp_tprs)
        mean_tpr = stack.mean(axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = stack.std(axis=0, ddof=0)
        label = f"mean ({result.mean['auroc']:.3f} ± {result.std['auroc']:.3f})"
        ax.plot(grid, mean_tpr, lw=2.2, color="C3", label=label)
        ax.fill_between(
            grid, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1), color="C3", alpha=0.15
        )

    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, linestyle="--", label="chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.005), xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title(f"ROC — {result.field}")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _artifacts_binary(res: AggregateResult, out_dir: Path, figures_dir: Path) -> None:
    _plot_roc(res, figures_dir / f"roc_{res.field}.png")
    plot_binary_score_distribution(
        [p.ensemble_prob for p in res.patient_predictions],
        [int(p.label) for p in res.patient_predictions],
        field=res.field,
        path=figures_dir / f"score_dist_{res.field}.png",
    )


def _headline_binary(res: AggregateResult) -> str:
    return (
        f"[{res.field}] mean±std AUROC = {res.mean['auroc']:.3f} ± {res.std['auroc']:.3f} "
        f"(pooled OOF {res.pooled['auroc']:.3f}; acc {res.mean['accuracy']:.3f}, "
        f"F1 {res.mean['f1']:.3f}; {res.n_outer_folds} outer folds)"
    )


# -- external scoring --------------------------------------------------------
def _write_external_binary_patient_csv(results: List[ExternalResult], out_dir: Path) -> None:
    """Long-format per-patient probs across the binary endpoints: one row per (field × patient × model),
    model = each outer-fold ensemble (o0…oK) plus grand, figures reconstructable from disk."""
    with (out_dir / "per_patient_external.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "patient_code", "model", "prob", "label"])
        for res in results:
            for p in sorted(res.labels):
                for o in res.outer_folds:
                    w.writerow(
                        [res.field, p, f"o{o}", f"{float(res.per_outer_patient_probs[o][p]):.6f}", res.labels[p]]
                    )
                w.writerow([res.field, p, "grand", f"{float(res.grand_patient_probs[p]):.6f}", res.labels[p]])


def _plot_external_roc(result: ExternalResult, figures_dir: Path) -> None:
    """Grand-ensemble ROC (the headline, bold) + per-outer ROC curves (thin, the stability spread)."""
    from sklearn.metrics import roc_curve

    plt = _agg_pyplot()
    patients = sorted(result.labels)
    labels_arr = np.asarray([result.labels[p] for p in patients], dtype=np.int64)
    if not (0 < labels_arr.sum() < len(labels_arr)):  # single-class external set -> ROC undefined
        import logging

        logging.getLogger(__name__).warning(
            "[%s/%s] external set is single-class; skipping ROC plot.", result.field, result.cohort
        )
        return

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    auroc_by_outer = {o: m["auroc"] for o, m in zip(result.outer_folds, result.per_outer)}
    for o in result.outer_folds:
        probs = np.asarray([result.per_outer_patient_probs[o][p] for p in patients], dtype=np.float64)
        fpr, tpr, _ = roc_curve(labels_arr, probs)
        ax.plot(fpr, tpr, lw=1.0, alpha=0.4, label=f"o{o} (AUROC={auroc_by_outer[o]:.3f})")

    grand_probs = np.asarray([result.grand_patient_probs[p] for p in patients], dtype=np.float64)
    fpr, tpr, _ = roc_curve(labels_arr, grand_probs)
    ax.plot(fpr, tpr, lw=2.4, color="C3", label=f"GRAND ensemble ({result.grand['auroc']:.3f})")

    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, linestyle="--", label="chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.005), xlabel="False positive rate", ylabel="True positive rate")
    ax.set_title(f"External ROC — {result.field} on {result.cohort}")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(figures_dir / f"roc_{result.field}.png", dpi=150)
    plt.close(fig)


def _artifacts_external_binary(result: ExternalResult, figures_dir: Path) -> None:
    """External ROC (skipped for a single-class cohort) + the grand-ensemble score-distribution beeswarm.
    The external read is threshold transfer: whether the score distributions shift on the held-out cohort."""
    _plot_external_roc(result, figures_dir)
    patients = sorted(result.labels)
    plot_binary_score_distribution(
        [float(result.grand_patient_probs[p]) for p in patients],
        [int(result.labels[p]) for p in patients],
        field=result.field,
        path=figures_dir / f"score_dist_{result.field}.png",
        title_suffix=f" (external: {result.cohort})",
    )


# -- registry entries assembled by the drivers -------------------------------
AGGREGATE_HANDLER = _EndpointHandler(
    BINARY_METRIC_LABELS,
    _BINARY_FOOTER,
    True,
    _artifacts_binary,
    _headline_binary,
    metrics=lambda labels, values, k: binary_metrics(labels, values),
    patient_row=lambda p: (f"{p.ensemble_prob:.6f}", int(p.label)),
    strip_group="classification",
)

EXTERNAL_HANDLER = _ExternalHandler(
    BINARY_METRIC_LABELS,
    lambda labels, values, k: binary_metrics(labels, values),
    vector_probs=False,
    has_positive_count=True,
    write_patient_csv=_write_external_binary_patient_csv,
    write_artifact=_artifacts_external_binary,
)
