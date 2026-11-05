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
"""Expression endpoint (regression_vector: a per-gene RSEM vector target): the ``finish`` hook that adds the
pooled per-gene Pearson and significant-gene count (and, with a random-baseline sweep, SEQUOIA's conservative
count), per-gene Pearson histogram + CSV, gene-count summary callouts, headline, and the ``AGGREGATE_HANDLER``
and ``EXTERNAL_HANDLER``. The per-gene metric math lives in the ``gene_metrics`` leaf. Imports down from
``_common`` + ``gene_metrics``."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from dlux.data import layout
from dlux.eval._common import (
    AggregateResult,
    LabelModel,
    PatientTable,
    _agg_pyplot,
    _all_nan_is_missing,
    _EndpointHandler,
    _ExternalHandler,
    _float_labels,
)
from dlux.eval.gene_metrics import (
    conservative_significant_gene_count,
    gene_metric_summary,
    per_gene_pearson,
    significant_gene_count,
)

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


def _expression_metrics(labels: np.ndarray, values: np.ndarray, num_classes: int) -> Dict[str, float]:
    """Per-gene correlation summary across patients. ``labels`` and ``values`` are both (N, G): this is the
    only endpoint whose predictions and labels are vectors."""
    summ = gene_metric_summary(values, labels, top_k=_TOP_K)
    return {k: summ[k] for k in EXPRESSION_METRIC_LABELS}


def _no_patient_prediction(*_args) -> None:
    """Expression keeps its per-patient gene vectors in the result's pooled table only."""
    return None


def _finish_expression(result: AggregateResult, table: PatientTable, random_table: Optional[PatientTable]) -> None:
    """The pooled per-gene Pearson and the significant-gene count, and with a random-baseline sweep SEQUOIA's
    conservative significant-gene count, trained vs random per gene, aligned by patient."""
    result.pooled_gene_pearson = per_gene_pearson(table.preds, table.labels)
    result.pooled_significant = significant_gene_count(table.preds, table.labels)
    if random_table is None:
        return
    # Both sweeps run the same study and splits, so they score the same patients.
    if set(random_table.codes) != set(table.codes):
        raise ValueError(
            f"[{result.field}] the random baseline scores other patients than the trained sweep "
            f"({len(set(random_table.codes) - set(table.codes))} only in the random, "
            f"{len(set(table.codes) - set(random_table.codes))} only in the trained)."
        )
    random_row = {code: i for i, code in enumerate(random_table.codes)}
    random_aligned = random_table.preds[[random_row[code] for code in table.codes]]  # trained OOF order
    result.pooled_conservative = conservative_significant_gene_count(table.preds, random_aligned, table.labels)


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


def _artifacts_expression(res: AggregateResult, out_dir: Path, figures_dir: Path) -> None:
    _plot_gene_pearson_hist(res, figures_dir / f"gene_pearson_{res.field}.png")
    _write_per_gene_csv(res, layout.per_gene_csv(out_dir, res.field))


def _headline_expression(res: AggregateResult) -> str:
    # The expression aggregate fills both, and it is the only endpoint this headline serves.
    assert res.pooled_significant is not None
    pooled_conservative = res.pooled_conservative
    vs_random = (
        f"; vs-random {pooled_conservative['n_significant']}/{pooled_conservative['n_scored']}"
        if pooled_conservative
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


MODEL = LabelModel(
    EXPRESSION_METRIC_LABELS,
    _expression_metrics,
    has_positive_count=False,
    vector_preds=True,  # the prediction is a (G,) gene vector...
    vector_labels=True,  # ...and so is the label. The only endpoint where both are vectors.
    label_array=_float_labels,
    is_missing=_all_nan_is_missing,  # a single NaN gene is data; an all-NaN row is a missing patient
)

AGGREGATE_HANDLER = _EndpointHandler(
    MODEL,
    _FOOTER,
    _artifacts_expression,
    _headline_expression,
    strip_group="expression",
    summary_extra=_expression_summary_extra,
    patient_prediction=_no_patient_prediction,
    finish=_finish_expression,
    uses_random_baseline=True,
    # The model's precision does not warrant more than float16 predictions for a (N, G) gene matrix.
    npz_dtypes=(np.float16, np.float32),
)


# -- external scoring --------------------------------------------------------
def _external_arrays(result: "ExternalResult", prob_map: Dict[str, object]):
    """(preds, labels) as (N, G) over the external cohort's patients, in sorted-code order."""
    patients = sorted(result.labels)
    preds = np.stack([np.asarray(prob_map[p], dtype=np.float64) for p in patients])
    labels = np.stack([np.asarray(result.labels[p], dtype=np.float64) for p in patients])
    return preds, labels


def _write_external_expression_per_gene_csv(results: List["ExternalResult"], out_dir: Path) -> None:
    """Per-gene Pearson on the grand ensemble, one file per field. Not per-patient like the other
    endpoints: a per-patient row here would be a 20k-gene vector, and the per-gene view is the one that
    answers 'which genes transfer'. ``gene_index`` maps to the panel order the run recorded."""
    for res in results:
        preds, labels = _external_arrays(res, res.grand_patient_probs)
        r = per_gene_pearson(preds, labels)
        order = np.argsort(-np.nan_to_num(r, nan=-np.inf))  # best-predicted genes first
        with layout.per_gene_csv(out_dir, res.field).open("w", newline="") as f:
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
    MODEL,
    write_patient_csv=_write_external_expression_per_gene_csv,
    write_artifact=_plot_external_gene_hist,
    section_extra=_expression_external_section_extra,  # the trained-vs-random gene count, when scored
    # No cross-field strip: a mean gene correlation does not share an axis with AUROC or C-index.
    strip_group="",
    strip_metric="gene_pearson_mean",
)
