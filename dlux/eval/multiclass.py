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
"""Multiclass endpoint: macro/per-class metrics, its own vector-prob aggregation core, confusion +
one-vs-rest ROC figures, per-patient CSV, headline, and both handlers, ``AGGREGATE_HANDLER`` (internal reporting) and
``EXTERNAL_HANDLER`` (scoring a held-out cohort). Per-patient predictions are (K,) softmax vectors in both
lanes; class = argmax, never a tuned threshold.
Imports down from ``_common`` only; the core references the module-level ``AGGREGATE_HANDLER`` (resolved at
call time) for its metric fn, so it never reaches up into the driver's registry. ``ExternalResult`` is
referenced solely in annotations (TYPE_CHECKING), so external.py can import this module."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List

import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score

from dlux.config.cohort import Objective
from dlux.eval._common import (
    AggregateResult,
    OuterFoldMetric,
    PatientPrediction,
    _agg_pyplot,
    _EndpointHandler,
    _ExternalHandler,
    _fmt,
    _mean_std_across_folds,
    confusion_figure,
    load_fold_predictions,
)
from dlux.eval.score_dist import plot_multiclass_score_distribution

if TYPE_CHECKING:
    from dlux.eval.external import ExternalResult

# Multiclass per-fold table: macro summaries only (comparable across folds, fixed columns). The
# per-class breakdown is a pooled-OOF table in the summary, not per-fold (per-class-per-fold = noisy).
MULTICLASS_METRIC_LABELS: Dict[str, str] = {
    "auroc": "macro AUROC",
    "accuracy": "acc",
    "balanced_accuracy": "bal_acc",
    "macro_f1": "macro F1",
    "qwk": "QWK",
}

_MULTICLASS_FOOTER = (
    "_Macro AUROC = mean one-vs-rest AUROC over classes; balanced acc = mean per-class recall (over classes "
    "with support); macro F1 averages per-class F1. Class = argmax (no tuned threshold), while the one-vs-rest "
    "ROC is threshold-free. The per-class table, confusion matrix and ROC are on the pooled OOF; class indices "
    "follow the contract map order. mean±std is across outer folds._"
)


def _ovr_auroc(is_c: np.ndarray, prob_c: np.ndarray) -> float:
    """One-vs-rest AUROC for one class: class-c indicator vs its softmax prob. NaN when the class is
    absent or universal in this set (AUROC undefined without both a positive and a negative)."""
    n_pos = int(is_c.sum())
    if not (0 < n_pos < is_c.size):
        return float("nan")
    return float(roc_auc_score(is_c.astype(np.int64), prob_c))


def multiclass_per_class(labels: np.ndarray, probs: np.ndarray, num_classes: int) -> List[dict]:
    """Per-class one-vs-rest AUROC + precision/recall/F1 (argmax) + support, one row per class."""
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    preds = np.argmax(probs, axis=1)
    nan = float("nan")
    rows: List[dict] = []
    for c in range(num_classes):
        is_c = labels == c
        support = int(is_c.sum())
        tp = int(np.sum((preds == c) & is_c))
        fp = int(np.sum((preds == c) & ~is_c))
        fn = int(np.sum((preds != c) & is_c))
        precision = tp / (tp + fp) if (tp + fp) else nan
        recall = tp / (tp + fn) if (tp + fn) else nan
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else nan
        rows.append(
            {
                "class": c,
                "support": support,
                "auroc": _ovr_auroc(is_c, probs[:, c]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def multiclass_metrics(labels: np.ndarray, probs: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Macro one-vs-rest AUROC, accuracy, balanced accuracy (mean per-class recall over present
    classes) and macro-F1 from class-index labels + (N, K) softmax probs. Class = argmax (no tuned
    threshold). NaNs where a component is undefined are ignored in the macro means."""
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    nan = float("nan")
    if labels.size == 0:
        return {k: nan for k in MULTICLASS_METRIC_LABELS}
    preds = np.argmax(probs, axis=1)
    per_class = multiclass_per_class(labels, probs, num_classes)

    def _macro(key: str, present_only: bool = False) -> float:
        vals = [r[key] for r in per_class if not (present_only and r["support"] == 0)]
        finite = [v for v in vals if not np.isnan(v)]
        return float(np.mean(finite)) if finite else nan

    # QWK rewards near-misses over far ones, the ordinal-aware metric. NaN when there is no rating
    # variance (kappa is 0/0), so an all-one-class fold does not force the mean.
    labs = list(range(num_classes))
    qwk = cohen_kappa_score(labels, preds, weights="quadratic", labels=labs) if labels.size else nan
    return {
        "auroc": _macro("auroc"),
        "accuracy": float(np.mean(preds == labels)),
        "balanced_accuracy": _macro("recall", present_only=True),  # mean recall over classes with support
        "macro_f1": _macro("f1"),
        "qwk": float(qwk),
    }


def _aggregate_field_multiclass(field: str, outer_to_reps: Dict[int, List[tuple[int, Path]]]) -> AggregateResult:
    """Multiclass aggregation: roll (N, K) softmax probs slide->patient (mean), ensemble inner
    replicates (mean-of-probs), per-outer + pooled macro/per-class metrics. Parallels _aggregate_field
    but carries per-class prob vectors (like the expression core) instead of a scalar positive prob."""
    num_classes = 0
    patient_preds: List[PatientPrediction] = []
    per_outer: List[OuterFoldMetric] = []
    inner_coverage: Dict[int, int] = {}
    pooled_probs_list: List[np.ndarray] = []
    pooled_labels_list: List[np.ndarray] = []

    for outer in sorted(outer_to_reps):
        reps = sorted(outer_to_reps[outer])
        per_patient_probs: Dict[str, List[np.ndarray]] = {}
        per_patient_label: Dict[str, int] = {}
        replicate_patient_sets: List[set] = []

        for _inner, npz in reps:
            data = load_fold_predictions(npz)
            probs = data["probs"]  # (N, K)
            if probs.ndim != 2:
                raise ValueError(f"multiclass fold {npz} has {probs.ndim}-D probs; expected 2-D (N, K).")
            num_classes = num_classes or probs.shape[1]
            slide_probs: Dict[str, List[np.ndarray]] = {}
            label_by_pc: Dict[str, int] = {}
            for pc, prob_vec, lbl in zip(data["patient_codes"], probs, data["labels"]):
                pc = str(pc)
                slide_probs.setdefault(pc, []).append(np.asarray(prob_vec, dtype=np.float64))
                label_by_pc[pc] = int(round(float(lbl)))  # loaded as float64; class index
            rolled = {pc: np.mean(v, axis=0) for pc, v in slide_probs.items()}  # slide -> patient mean
            replicate_patient_sets.append(set(rolled))
            for pc, vec in rolled.items():
                per_patient_probs.setdefault(pc, []).append(vec)
                per_patient_label[pc] = label_by_pc[pc]

        # shared-TEST invariant: every inner replicate of an outer fold covers the same patients.
        first = replicate_patient_sets[0]
        for other in replicate_patient_sets[1:]:
            if other != first:
                raise ValueError(
                    f"shared-TEST invariant violated for field '{field}' outer fold {outer}: "
                    f"inner replicates cover different patient sets (symmetric diff: {sorted(first ^ other)})"
                )

        inner_coverage[outer] = len(reps)
        pcs = sorted(per_patient_probs)
        probs_o = np.stack([np.mean(per_patient_probs[pc], axis=0) for pc in pcs])  # (N_o, K) ensembled
        labels_o = np.asarray([per_patient_label[pc] for pc in pcs], dtype=np.int64)
        per_outer.append(
            OuterFoldMetric(field, outer, len(pcs), 0, AGGREGATE_HANDLER.metrics(labels_o, probs_o, num_classes))
        )
        for i, pc in enumerate(pcs):
            patient_preds.append(
                PatientPrediction(
                    field,
                    outer,
                    pc,
                    float("nan"),
                    int(labels_o[i]),
                    len(per_patient_probs[pc]),
                    ensemble_probs=probs_o[i],
                )
            )
        pooled_probs_list.append(probs_o)
        pooled_labels_list.append(labels_o)

    pooled_probs = np.concatenate(pooled_probs_list)
    pooled_labels = np.concatenate(pooled_labels_list)
    mean, std = _mean_std_across_folds(per_outer, AGGREGATE_HANDLER)
    return AggregateResult(
        field=field,
        endpoint_type=Objective.multiclass,
        patient_predictions=patient_preds,
        per_outer=per_outer,
        mean=mean,
        std=std,
        pooled=AGGREGATE_HANDLER.metrics(pooled_labels, pooled_probs, num_classes),
        pooled_n=int(pooled_labels.size),
        pooled_positive=0,  # no single positive class in multiclass
        n_outer_folds=len(inner_coverage),
        inner_coverage=inner_coverage,
        num_classes=num_classes,
        pooled_per_class=multiclass_per_class(pooled_labels, pooled_probs, num_classes),
    )


def _per_class_table(rows: List[dict], caption: str) -> List[str]:
    """Markdown per-class one-vs-rest table (AUROC/precision/recall/F1/support). Shared by the internal
    summary and the external report so both read identically."""
    if not rows:
        return []
    out = [
        caption,
        "",
        "| class | support | AUROC | precision | recall | F1 |",
        "|:--|--:|--:|--:|--:|--:|",
    ]
    for r in rows:
        out.append(
            f"| {r['class']} | {r['support']} | {_fmt(r['auroc'])} | {_fmt(r['precision'])} | "
            f"{_fmt(r['recall'])} | {_fmt(r['f1'])} |"
        )
    out.append("")
    return out


def _multiclass_summary_extra(res: AggregateResult) -> List[str]:
    """Per-class pooled-OOF table under a multiclass endpoint's macro table: AUROC/precision/recall/F1/support."""
    return _per_class_table(
        res.pooled_per_class, "Per-class (pooled OOF, one-vs-rest; class index = contract map order):"
    )


def _write_multiclass_predictions_csv(res: AggregateResult, path: Path) -> None:
    """Per-patient pooled-OOF probs for the whole panel: patient_code, outer_fold, prob_0..K-1, pred, label."""
    k = res.num_classes or 0
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_code", "outer_fold", *[f"prob_{c}" for c in range(k)], "pred", "label"])
        for p in res.patient_predictions:
            probs = np.asarray(p.ensemble_probs, dtype=np.float64)
            w.writerow(
                [p.patient_code, p.outer_fold, *(f"{v:.6f}" for v in probs), int(np.argmax(probs)), int(p.label)]
            )


def _ovr_roc_figure(labels: np.ndarray, probs: np.ndarray, num_classes: int, title: str, path: Path) -> None:
    """One ROC curve per class (class-c indicator vs its softmax prob), with the per-class AUROC in the
    legend and the macro mean in the title.

    Threshold-free, so it answers a different question from the confusion matrix beside it, which is
    argmax at a fixed operating point. A class absent or universal in this set has no ROC and is
    labelled as such rather than skipped silently."""
    from sklearn.metrics import roc_curve

    plt = _agg_pyplot()
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    aurocs: List[float] = []
    for c in range(num_classes):
        is_c = labels == c
        auroc = _ovr_auroc(is_c, probs[:, c])
        if np.isnan(auroc):
            ax.plot([], [], lw=1.6, label=f"class {c} (undefined, {int(is_c.sum())}/{labels.size})")
            continue
        aurocs.append(auroc)
        fpr, tpr, _ = roc_curve(is_c.astype(np.int64), probs[:, c])
        ax.plot(fpr, tpr, lw=1.6, label=f"class {c} (AUROC={auroc:.3f}, n={int(is_c.sum())})")

    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, linestyle="--", label="chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.005), xlabel="False positive rate", ylabel="True positive rate")
    macro = f"macro {np.mean(aurocs):.3f}" if aurocs else "macro undefined"
    ax.set_title(f"{title}\none-vs-rest ROC, {macro}", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_ovr_roc(res: AggregateResult, path: Path) -> None:
    """Pooled-OOF one-vs-rest ROC. Pooled rather than per-fold because a per-class curve on one outer
    fold rests on that fold's share of the class, which is why the per-class table is pooled too."""
    labels = np.asarray([p.label for p in res.patient_predictions], dtype=np.int64)
    probs = np.stack([np.asarray(p.ensemble_probs, dtype=np.float64) for p in res.patient_predictions])
    _ovr_roc_figure(labels, probs, res.num_classes or 0, f"{res.field} (pooled OOF, {res.pooled_n} patients)", path)


def _plot_confusion(res: AggregateResult, path: Path) -> None:
    """Pooled-OOF confusion matrix for a multiclass aggregate result (argmax over the ensembled probs)."""
    labels = [int(p.label) for p in res.patient_predictions]
    preds = [int(np.argmax(p.ensemble_probs)) for p in res.patient_predictions]
    confusion_figure(
        labels, preds, res.num_classes or 0, f"{res.field}: confusion (pooled OOF, {res.pooled_n} patients)", path
    )


def _artifacts_multiclass(res: AggregateResult, out_dir: Path, figures_dir: Path) -> None:
    _plot_confusion(res, figures_dir / f"confusion_{res.field}.png")
    _plot_ovr_roc(res, figures_dir / f"roc_{res.field}.png")
    plot_multiclass_score_distribution(
        np.stack([np.asarray(p.ensemble_probs, dtype=np.float64) for p in res.patient_predictions]),
        [int(p.label) for p in res.patient_predictions],
        field=res.field,
        path=figures_dir / f"score_dist_{res.field}.png",
        num_classes=res.num_classes or 0,
    )
    _write_multiclass_predictions_csv(res, out_dir / f"predictions_{res.field}.csv")


def _headline_multiclass(res: AggregateResult) -> str:
    return (
        f"[{res.field}] mean±std macro-AUROC = {res.mean['auroc']:.3f} ± {res.std['auroc']:.3f} "
        f"(pooled OOF {res.pooled['auroc']:.3f}; acc {res.mean['accuracy']:.3f}, "
        f"macro-F1 {res.mean['macro_f1']:.3f}; {res.n_outer_folds} outer folds)"
    )


AGGREGATE_HANDLER = _EndpointHandler(
    MULTICLASS_METRIC_LABELS,
    _MULTICLASS_FOOTER,
    False,
    _artifacts_multiclass,
    _headline_multiclass,
    metrics=lambda labels, values, k: multiclass_metrics(labels, values, k),
    patient_row=lambda p: (int(np.argmax(p.ensemble_probs)), int(p.label)),
    strip_group="classification",
    summary_extra=_multiclass_summary_extra,
)


# -- external scoring --------------------------------------------------------
def _external_grand_arrays(result: "ExternalResult") -> tuple[np.ndarray, np.ndarray]:
    """(labels, grand-ensemble (N, K) probs) over the external cohort's patients, in sorted-code order."""
    patients = sorted(result.labels)
    labels = np.asarray([result.labels[p] for p in patients], dtype=np.int64)
    probs = np.stack([np.asarray(result.grand_patient_probs[p], dtype=np.float64) for p in patients])
    return labels, probs


def _write_external_multiclass_patient_csv(results: List["ExternalResult"], out_dir: Path) -> None:
    """One CSV per field (class count varies by endpoint, so columns cannot be shared across fields):
    patient_code, model, prob_0..K-1, pred, label. ``model`` is each outer-fold ensemble (o0…oK) plus
    ``grand``, the same long shape as binary's, so the figures stay reconstructable from disk."""
    for res in results:
        k = res.num_classes or 0
        with (out_dir / f"per_patient_external_{res.field}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_code", "model", *[f"prob_{c}" for c in range(k)], "pred", "label"])
            for p in sorted(res.labels):
                for model, prob_map in [
                    *((f"o{o}", res.per_outer_patient_probs[o]) for o in res.outer_folds),
                    ("grand", res.grand_patient_probs),
                ]:
                    vec = np.asarray(prob_map[p], dtype=np.float64)
                    w.writerow([p, model, *(f"{v:.6f}" for v in vec), int(np.argmax(vec)), res.labels[p]])


def _plot_external_figures(result: "ExternalResult", figures_dir: Path) -> None:
    """Grand-ensemble confusion matrix + one-vs-rest ROC on the external cohort. The per-outer
    ensembles are a stability diagnostic and would need K figures each to show; the headline is what
    gets plotted."""
    labels, probs = _external_grand_arrays(result)
    num_classes = result.num_classes or 0
    confusion_figure(
        labels,
        np.argmax(probs, axis=1),
        num_classes,
        f"{result.field}: confusion (GRAND ensemble on {result.cohort}, {labels.size} patients)",
        figures_dir / f"confusion_{result.field}.png",
    )
    _ovr_roc_figure(
        labels,
        probs,
        num_classes,
        f"{result.field} (GRAND ensemble on {result.cohort}, {labels.size} patients)",
        figures_dir / f"roc_{result.field}.png",
    )
    plot_multiclass_score_distribution(
        probs,
        labels,
        field=result.field,
        path=figures_dir / f"score_dist_{result.field}.png",
        num_classes=num_classes,
        title_suffix=f" (external: {result.cohort})",
    )


def _multiclass_section_extra(result: "ExternalResult") -> List[str]:
    """Per-class one-vs-rest table on the grand ensemble, the external counterpart of the pooled-OOF
    table in the internal summary."""
    labels, probs = _external_grand_arrays(result)
    rows = multiclass_per_class(labels, probs, result.num_classes or 0)
    return _per_class_table(rows, "Per-class (GRAND ensemble, one-vs-rest; class index = contract map order):")


EXTERNAL_HANDLER = _ExternalHandler(
    MULTICLASS_METRIC_LABELS,
    lambda labels, values, k: multiclass_metrics(labels, values, k),
    vector_probs=True,  # per-patient prediction is a (K,) softmax vector, not a scalar
    has_positive_count=False,  # "positive" is binary-only; K classes have no single positive count
    write_patient_csv=_write_external_multiclass_patient_csv,
    write_artifact=_plot_external_figures,
    section_extra=_multiclass_section_extra,
)
