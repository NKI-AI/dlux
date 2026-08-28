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
"""Expression endpoint (regression_vector: a per-gene RSEM vector target): the per-gene aggregation
core (with the optional SEQUOIA random-baseline count), per-gene Pearson histogram + CSV, gene-count
summary callouts, headline, and the ``AGGREGATE_HANDLER``. Not externally scorable. The per-gene metric
math lives in the ``gene_metrics`` leaf; this module supplies the vector-shape core + reporting. Imports
down from ``_common`` + ``gene_metrics``; the core references the module-level ``AGGREGATE_HANDLER``
(resolved at call time) for its metric labels."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from dlux.config.cohort import Objective
from dlux.eval._common import (
    AggregateResult,
    OuterFoldMetric,
    _agg_pyplot,
    _all_nan_is_missing,
    _EndpointHandler,
    _ExternalHandler,
    _float_labels,
    _mean_std_across_folds,
    _no_patient_row,
    _no_scalar_metrics,
    _roll_slides_to_patients_vec,
    load_fold_predictions,
)
from dlux.eval.gene_metrics import (
    conservative_significant_gene_count,
    gene_metric_summary,
    per_gene_pearson,
    significant_gene_count,
)

logger = logging.getLogger(__name__)

_TOP_K = 100  # "top-k best-predicted genes" for the gene_pearson_topk headline

if TYPE_CHECKING:
    from dlux.eval.external import ExternalResult


EXPRESSION_METRIC_LABELS: Dict[str, str] = {
    "gene_pearson_mean": "mean gene Pearson",
    "gene_pearson_topk": f"top-{_TOP_K} Pearson",
    "gene_spearman_median": "median gene Spearman",
}

_FOOTER = (
    f"_Per-gene correlations across patients (preds + labels in log1p space): mean gene Pearson is the "
    f"headline; top-{_TOP_K} = mean Pearson of the best-predicted genes; median gene Spearman "
    f"is rank-based. Genes constant in preds/labels are excluded. mean±std is across outer folds; pooled "
    f"OOF is over all out-of-fold patients (the tighter per-gene estimate)._"
)


def _pool_expression_oof(field: str, outer_to_reps: Dict[int, List[tuple[int, Path]]]):
    """Ensemble inner replicates + roll slides->patients per outer fold, concatenate across outer folds
    -> the pooled OOF. Returns ``(patient_codes, preds (N,G), labels (N,G), per_outer, inner_coverage)``
    with patient_codes tracking the row order of preds/labels (needed to align a second sweep)."""
    per_outer: List[OuterFoldMetric] = []
    inner_coverage: Dict[int, int] = {}
    patient_codes: List[str] = []
    outer_folds: List[int] = []  # the outer fold each pooled-OOF patient came from, aligned with the rest
    oof_preds: List[np.ndarray] = []
    oof_labels: List[np.ndarray] = []

    for outer in sorted(outer_to_reps):
        reps = sorted(outer_to_reps[outer])
        per_patient_pred: Dict[str, List[np.ndarray]] = {}
        per_patient_label: Dict[str, np.ndarray] = {}
        replicate_patient_sets: List[set] = []
        for _inner, npz in reps:
            data = load_fold_predictions(npz)
            rolled_pred, rolled_label = _roll_slides_to_patients_vec(
                data["patient_codes"], data["probs"], data["labels"]
            )
            replicate_patient_sets.append(set(rolled_pred))
            for pc, vec in rolled_pred.items():
                per_patient_pred.setdefault(pc, []).append(vec)
                per_patient_label[pc] = rolled_label[pc]

        first = replicate_patient_sets[0]  # shared-TEST invariant across inner replicates
        for other in replicate_patient_sets[1:]:
            if other != first:
                raise ValueError(
                    f"shared-TEST invariant violated for field '{field}' outer fold {outer}: "
                    f"inner replicates cover different patient sets (symmetric diff: {sorted(first ^ other)})"
                )

        inner_coverage[outer] = len(reps)
        pcs = sorted(per_patient_pred)
        preds_o = np.stack([np.mean(per_patient_pred[pc], axis=0) for pc in pcs])  # (N_o, G) ensembled
        labels_o = np.stack([per_patient_label[pc] for pc in pcs])  # (N_o, G)
        summ = gene_metric_summary(preds_o, labels_o, top_k=_TOP_K)
        per_outer.append(OuterFoldMetric(field, outer, len(pcs), 0, {k: summ[k] for k in EXPRESSION_METRIC_LABELS}))
        patient_codes.extend(pcs)
        outer_folds.extend([outer] * len(pcs))
        oof_preds.append(preds_o)
        oof_labels.append(labels_o)

    return patient_codes, outer_folds, np.concatenate(oof_preds), np.concatenate(oof_labels), per_outer, inner_coverage


def _aggregate_field_expression(
    field: str,
    outer_to_reps: Dict[int, List[tuple[int, Path]]],
    random_reps: Optional[Dict[int, List[tuple[int, Path]]]] = None,
) -> AggregateResult:
    """Expression (regression_vector) aggregation: per-gene metrics per outer fold + over the pooled
    OOF. When ``random_reps`` (a random-baseline sweep's folds) is given, also compute SEQUOIA's
    conservative significant-gene count, trained vs random per gene, aligned by patient."""
    pcs, folds, pooled_preds, pooled_labels, per_outer, inner_coverage = _pool_expression_oof(field, outer_to_reps)
    pooled_summ = gene_metric_summary(pooled_preds, pooled_labels, top_k=_TOP_K)
    mean, std = _mean_std_across_folds(per_outer, AGGREGATE_HANDLER)

    conservative = None
    if random_reps is not None:
        r_pcs, _r_folds, r_preds, _r_labels, _r_per_outer, _r_cov = _pool_expression_oof(field, random_reps)
        r_by_pc = {pc: r_preds[i] for i, pc in enumerate(r_pcs)}
        idx = [i for i, pc in enumerate(pcs) if pc in r_by_pc]  # patients in both sweeps, trained OOF order
        if idx:
            random_aligned = np.stack([r_by_pc[pcs[i]] for i in idx])
            conservative = conservative_significant_gene_count(pooled_preds[idx], random_aligned, pooled_labels[idx])
        else:
            logger.warning("[%s] random baseline shares no patients with the trained OOF; skipping.", field)

    return AggregateResult(
        field=field,
        endpoint_type=Objective.regression_vector,
        patient_predictions=[],  # per-patient gene vectors ride pooled_preds/labels below, not this list
        patient_codes=pcs,
        pooled_outer_folds=folds,
        pooled_preds=pooled_preds,
        pooled_labels=pooled_labels,
        per_outer=per_outer,
        mean=mean,
        std=std,
        pooled={k: pooled_summ[k] for k in EXPRESSION_METRIC_LABELS},
        pooled_n=int(pooled_labels.shape[0]),
        pooled_positive=0,
        n_outer_folds=len(inner_coverage),
        inner_coverage=inner_coverage,
        pooled_gene_pearson=per_gene_pearson(pooled_preds, pooled_labels),
        pooled_significant=significant_gene_count(pooled_preds, pooled_labels),
        pooled_conservative=conservative,
    )


def _write_per_gene_csv(result: AggregateResult, path: Path) -> None:
    """Pooled-OOF per-gene Pearson for the whole panel. ``gene_index`` maps to the rnaseq matrix /
    genes.csv column order (the npz does not carry gene IDs); join to genes.csv for symbols."""
    r = result.pooled_gene_pearson
    if r is None:
        return
    order = np.argsort(-np.nan_to_num(r, nan=-np.inf))  # best-predicted genes first
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["gene_index", "pooled_pearson"])
        for gi in order:
            w.writerow([int(gi), f"{float(r[gi]):.6f}" if not np.isnan(r[gi]) else ""])


def _plot_gene_pearson_hist(result: AggregateResult, path: Path) -> None:
    """Histogram of per-gene OOF Pearson across the panel, how well each gene is predicted, with the
    mean marked and r=0 for reference."""
    plt = _agg_pyplot()
    r = result.pooled_gene_pearson
    r = r[~np.isnan(r)] if r is not None else np.zeros(0)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if r.size:
        ax.hist(r, bins=60, color="#8172B3")
        ax.axvline(float(r.mean()), color="#c44e52", lw=2, label=f"mean {float(r.mean()):.3f}")
        ax.axvline(0.0, color="#555", lw=1, ls="--")
        ax.legend()
    ax.set_xlabel("per-gene Pearson r (pooled OOF)")
    ax.set_ylabel("genes")
    ax.set_title(f"{result.field}: per-gene prediction\n{result.pooled_n} patients, {r.size} scored genes", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _expression_summary_extra(res: AggregateResult) -> List[str]:
    """Gene-count callouts under an expression endpoint's table: significant + well-predicted-vs-random."""
    out: List[str] = []
    sig = res.pooled_significant
    if sig is not None and sig["n_scored"]:
        pct = 100 * sig["n_significant"] / sig["n_scored"]
        out += [
            f"**Significantly predicted genes** (pooled OOF, BH-FDR q<{sig['alpha']:g}): "
            f"{sig['n_significant']} / {sig['n_scored']} ({pct:.0f}%).",
            "",
        ]
    con = res.pooled_conservative
    if con is not None and con["n_scored"]:
        cpct = 100 * con["n_significant"] / con["n_scored"]
        out += [
            f"**Well-predicted vs random baseline** (SEQUOIA: r1>r2 by Williams p<{con['alpha']:g} + "
            f"BH<{con['fdr']:g}, & rmse1<rmse2): {con['n_significant']} / {con['n_scored']} ({cpct:.0f}%).",
            "",
        ]
    return out


def _write_per_patient_npz(result: AggregateResult, path: Path) -> None:
    """Pooled-OOF per-patient predicted + measured expression, one ensembled vector per patient.

    Predictions are float16 (the model's precision does not warrant more); labels stay float32. Gene
    columns share the order of ``pooled_gene_pearson`` / the per_gene CSV, so they join to the cohort's
    genes.csv the same way. This is the artifact stage-1 ``compare`` recomputes its metric from, and it
    keeps the predicted transcriptome available for downstream use without re-running the model."""
    if result.pooled_preds is None or result.pooled_labels is None:
        return
    np.savez_compressed(
        path,
        patient_codes=np.asarray(result.patient_codes),
        outer_folds=np.asarray(result.pooled_outer_folds),
        preds=result.pooled_preds.astype(np.float16),
        labels=result.pooled_labels.astype(np.float32),
        gene_index=np.arange(result.pooled_preds.shape[1]),
    )


def _artifacts_expression(res: AggregateResult, out_dir: Path, figures_dir: Path) -> None:
    _plot_gene_pearson_hist(res, figures_dir / f"gene_pearson_{res.field}.png")
    _write_per_gene_csv(res, out_dir / f"per_gene_pearson_{res.field}.csv")
    _write_per_patient_npz(res, out_dir / f"per_patient_{res.field}.npz")


def _headline_expression(res: AggregateResult) -> str:
    vs_random = (
        f"; vs-random {res.pooled_conservative['n_significant']}/{res.pooled_conservative['n_scored']}"
        if res.pooled_conservative
        else ""
    )
    return (
        f"[{res.field}] mean±std gene-Pearson = {res.mean['gene_pearson_mean']:.3f} ± "
        f"{res.std['gene_pearson_mean']:.3f} (pooled OOF {res.pooled['gene_pearson_mean']:.3f}; "
        f"top-{_TOP_K} {res.pooled['gene_pearson_topk']:.3f}, Spearman-median "
        f"{res.pooled['gene_spearman_median']:.3f}; sig genes "
        f"{res.pooled_significant['n_significant']}/{res.pooled_significant['n_scored']}"
        f"{vs_random}; {res.n_outer_folds} outer folds)"
    )


AGGREGATE_HANDLER = _EndpointHandler(
    EXPRESSION_METRIC_LABELS,
    _FOOTER,
    False,
    _artifacts_expression,
    _headline_expression,
    metrics=_no_scalar_metrics,
    patient_row=_no_patient_row,
    strip_group="expression",
    summary_extra=_expression_summary_extra,
)


# -- external scoring --------------------------------------------------------
def _external_arrays(result: "ExternalResult", prob_map: Dict[str, object]):
    """(preds, labels) as (N, G) over the external cohort's patients, in sorted-code order."""
    patients = sorted(result.labels)
    preds = np.stack([np.asarray(prob_map[p], dtype=np.float64) for p in patients])
    labels = np.stack([np.asarray(result.labels[p], dtype=np.float64) for p in patients])
    return preds, labels


def _expression_external_metrics(labels: np.ndarray, values: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Per-gene correlation summary on the external cohort. ``labels`` and ``values`` are both (N, G):
    this is the only endpoint whose predictions and labels are vectors."""
    summ = gene_metric_summary(values, labels, top_k=_TOP_K)
    return {k: summ[k] for k in EXPRESSION_METRIC_LABELS}


def _write_external_expression_per_gene_csv(results: List["ExternalResult"], out_dir: Path) -> None:
    """Per-gene Pearson on the grand ensemble, one file per field. Not per-patient like the other
    endpoints: a per-patient row here would be a 20k-gene vector, and the per-gene view is the one that
    answers 'which genes transfer'. ``gene_index`` maps to the panel order the run recorded."""
    for res in results:
        preds, labels = _external_arrays(res, res.grand_patient_probs)
        r = per_gene_pearson(preds, labels)
        order = np.argsort(-np.nan_to_num(r, nan=-np.inf))  # best-predicted genes first
        with (out_dir / f"per_gene_external_{res.field}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gene_index", "grand_pearson"])
            for gi in order:
                w.writerow([int(gi), f"{float(r[gi]):.6f}" if not np.isnan(r[gi]) else ""])


def _plot_external_gene_hist(result: "ExternalResult", figures_dir: Path) -> None:
    """Per-gene Pearson histogram for the grand ensemble, the external counterpart of the pooled-OOF one."""
    plt = _agg_pyplot()
    preds, labels = _external_arrays(result, result.grand_patient_probs)
    r = per_gene_pearson(preds, labels)
    r = r[~np.isnan(r)]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if r.size:
        ax.hist(r, bins=60, color="#8172B3")
        ax.axvline(float(r.mean()), color="#c44e52", lw=2, label=f"mean {float(r.mean()):.3f}")
        ax.axvline(0.0, color="#555", lw=1, ls="--")
        ax.legend()
    ax.set_xlabel("per-gene Pearson r (GRAND ensemble)")
    ax.set_ylabel("genes")
    # Two lines: a cohort name is arbitrarily long, and a single-line title overruns the axes width.
    ax.set_title(f"{result.field} on {result.cohort}\nper-gene prediction ({result.n_patients} patients)", fontsize=11)
    fig.tight_layout()
    fig.savefig(figures_dir / f"gene_pearson_{result.field}.png", dpi=150)
    plt.close(fig)


def _expression_external_section_extra(result) -> List[str]:
    """The trained-vs-random significant-gene count, printed only when a random-baseline arm was scored
    (``result.conservative`` set). Mirrors what aggregate reports internally."""
    c = getattr(result, "conservative", None)
    if not c:
        return []
    return [
        f"**Well-predicted genes (trained beats a random-init model):** {c['n_significant']} / "
        f"{c['n_scored']} scored (α={c['alpha']}, FDR<{c['fdr']}). Williams has df = n−3, so on a small "
        'external cohort a low count means "too few patients", not "does not transfer".',
        "",
    ]


EXTERNAL_HANDLER = _ExternalHandler(
    EXPRESSION_METRIC_LABELS,
    _expression_external_metrics,
    vector_probs=True,  # the prediction is a (G,) gene vector...
    has_positive_count=False,
    write_patient_csv=_write_external_expression_per_gene_csv,
    write_artifact=_plot_external_gene_hist,
    vector_labels=True,  # ...and so is the label. The only endpoint where both are vectors.
    label_array=_float_labels,
    is_missing=_all_nan_is_missing,  # a single NaN gene is data; an all-NaN row is a missing patient
    section_extra=_expression_external_section_extra,  # the trained-vs-random gene count, when scored
    # No cross-field strip: a mean gene correlation does not share an axis with AUROC or C-index.
    strip_group="",
    strip_metric="gene_pearson_mean",
)
