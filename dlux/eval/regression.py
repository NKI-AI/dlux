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
"""Regression endpoint (scalar continuous target): metric labels + fn, predicted-vs-actual scatter,
headline, and both handlers, ``AGGREGATE_HANDLER`` (internal reporting) and ``EXTERNAL_HANDLER``
(scoring a held-out cohort). Shares the scalar aggregation core in ``_common``, this module only
supplies the handlers. Imports down from ``_common`` only; ``ExternalResult`` is referenced solely in
annotations (TYPE_CHECKING), so external.py can import this module."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import numpy as np

from dlux.eval._common import (
    AggregateResult,
    _agg_pyplot,
    _EndpointHandler,
    _ExternalHandler,
    _float_labels,
    _fmt,
    _nan_is_missing,
)

if TYPE_CHECKING:
    from dlux.eval.external import ExternalResult

REGRESSION_METRIC_LABELS: Dict[str, str] = {
    "mae": "MAE",
    "rmse": "RMSE",
    "r2": "R²",
    "pearson": "Pearson r",
    "spearman": "Spearman ρ",
}

_REGRESSION_FOOTER = (
    "_MAE/RMSE are in target units; R²/Pearson/Spearman are scale-free (Spearman = rank correlation, the "
    "honest 'does it track the target' number). mean±std is across outer folds; pooled OOF is over all "
    "out-of-fold patients._"
)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation via Pearson of the ranks (argsort ranks; ties broken arbitrarily,
    fine for a diagnostic)."""
    ar = np.argsort(np.argsort(a))
    br = np.argsort(np.argsort(b))
    return float(np.corrcoef(ar, br)[0, 1])


def regression_metrics(targets: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    """MAE, RMSE, R², Pearson r, Spearman ρ from continuous targets + predictions. Degenerate
    (constant target/pred, n<2) yields NaN where the metric is undefined."""
    targets = np.asarray(targets, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    n = int(targets.size)
    nan = float("nan")
    if n == 0:
        return {k: nan for k in REGRESSION_METRIC_LABELS}
    err = preds - targets
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    both_vary = n > 1 and preds.std() > 0 and targets.std() > 0
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else nan,
        "pearson": float(np.corrcoef(preds, targets)[0, 1]) if both_vary else nan,
        "spearman": _spearman(preds, targets) if both_vary else nan,
    }


def scatter_figure(actual: np.ndarray, preds: np.ndarray, title: str, stats: str, path: Path) -> None:
    """Predicted-vs-actual scatter (semi-transparent points so overlap shows clustering), with y=x
    (ideal) and the OLS fit whose slope exposes regression-to-the-mean shrinkage. Shared by the internal
    (pooled OOF) and external (grand ensemble) scorers so both read identically."""
    plt = _agg_pyplot()
    preds = np.asarray(preds, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if preds.size == 0:
        return

    lo = float(min(actual.min(), preds.min()))
    hi = float(max(actual.max(), preds.max()))
    pad = 0.03 * ((hi - lo) or 1.0)
    lo, hi = lo - pad, hi + pad

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    # Semi-transparent points: overlap darkens, so clustering reads directly off the real points.
    ax.scatter(actual, preds, s=18, alpha=0.4, color="#1f77b4", edgecolors="none")
    ax.plot([lo, hi], [lo, hi], color="grey", lw=1.0, linestyle="--", label="y = x (ideal)")
    slope, intercept = np.polyfit(actual, preds, 1)
    ax.plot(
        [lo, hi],
        [slope * lo + intercept, slope * hi + intercept],
        color="crimson",
        lw=1.5,
        label=f"OLS fit (slope {slope:.2f})",
    )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set(xlabel="actual", ylabel="predicted (ensemble)")
    ax.set_title(title)
    ax.text(
        0.03,
        0.97,
        stats,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="grey"),
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _artifacts_regression(res: AggregateResult, out_dir: Path, figures_dir: Path) -> None:
    scatter_figure(
        np.array([p.label for p in res.patient_predictions], dtype=np.float64),
        np.array([p.ensemble_prob for p in res.patient_predictions], dtype=np.float64),
        f"{res.field} — predicted vs actual",
        f"n = {len(res.patient_predictions)}\nR² = {_fmt(res.pooled['r2'])}\n"
        f"Spearman = {_fmt(res.pooled['spearman'])}\nMAE = {_fmt(res.pooled['mae'])}",
        figures_dir / f"scatter_{res.field}.png",
    )


def _headline_regression(res: AggregateResult) -> str:
    return (
        f"[{res.field}] mean±std R² = {res.mean['r2']:.3f} ± {res.std['r2']:.3f} "
        f"(pooled OOF {res.pooled['r2']:.3f}; Spearman {res.mean['spearman']:.3f}, "
        f"MAE {res.mean['mae']:.3f}; {res.n_outer_folds} outer folds)"
    )


AGGREGATE_HANDLER = _EndpointHandler(
    REGRESSION_METRIC_LABELS,
    _REGRESSION_FOOTER,
    False,
    _artifacts_regression,
    _headline_regression,
    metrics=lambda labels, values, k: regression_metrics(labels, values),
    patient_row=lambda p: (f"{p.ensemble_prob:.6f}", f"{p.label:.6f}"),
    strip_group="regression",
)


# -- external scoring --------------------------------------------------------
def _write_external_regression_patient_csv(results: List["ExternalResult"], out_dir: Path) -> None:
    """Long-format per-patient predictions across the regression endpoints: one row per (field x patient
    x model), model = each outer-fold ensemble (o0..oK) plus grand. Same shape as binary's, with the raw
    predicted value in place of a probability."""
    with (out_dir / "per_patient_external_regression.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "patient_code", "model", "predicted", "actual"])
        for res in results:
            for p in sorted(res.labels):
                for model, prob_map in [
                    *((f"o{o}", res.per_outer_patient_probs[o]) for o in res.outer_folds),
                    ("grand", res.grand_patient_probs),
                ]:
                    w.writerow([res.field, p, model, f"{float(prob_map[p]):.6f}", f"{float(res.labels[p]):.6f}"])


def _plot_external_scatter(result: "ExternalResult", figures_dir: Path) -> None:
    """Predicted-vs-actual for the grand ensemble on the external cohort, the external counterpart of
    the pooled-OOF scatter."""
    patients = sorted(result.labels)
    actual = np.asarray([float(result.labels[p]) for p in patients], dtype=np.float64)
    preds = np.asarray([float(result.grand_patient_probs[p]) for p in patients], dtype=np.float64)
    scatter_figure(
        actual,
        preds,
        f"{result.field} — predicted vs actual (GRAND on {result.cohort})",
        f"n = {result.n_patients}\nR² = {_fmt(result.grand['r2'])}\n"
        f"Spearman = {_fmt(result.grand['spearman'])}\nMAE = {_fmt(result.grand['mae'])}",
        figures_dir / f"scatter_{result.field}.png",
    )


EXTERNAL_HANDLER = _ExternalHandler(
    REGRESSION_METRIC_LABELS,
    lambda labels, values, k: regression_metrics(labels, values),
    vector_probs=False,
    has_positive_count=False,  # "positive" is a classification notion; a continuous target has none
    write_patient_csv=_write_external_regression_patient_csv,
    write_artifact=_plot_external_scatter,
    label_array=_float_labels,  # raw values, not class indices
    is_missing=_nan_is_missing,  # the objective's declared sentinel; a negative value is data
    scalar_label_cast=float,
    # Regression joins no cross-field strip: that figure frames scores in [0,1] against a 0.5 chance
    # line, which is meaningless for R²/MAE/Spearman. Its per-field scatter carries the same information.
    strip_group="",
    strip_metric="r2",
)
