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
"""Per-gene evaluation for the expression (regression_vector) endpoint, pure numpy, testable.

Predictions and labels are ``(N_patients, G)`` matrices in log1p space (as persisted by
``collect_predictions``). The metric is a correlation *per gene* (across patients), then summarised
across the panel: the headline is the mean per-gene Pearson; also the top-k mean (best-predicted
genes) and the median per-gene Spearman. A gene that is constant in preds or labels has an undefined
correlation (NaN) and is excluded from the summary (reported as ``n_scored`` / ``n_genes``).
"""

from __future__ import annotations

import math

import numpy as np

_erfc = np.vectorize(math.erfc)  # vectorised complementary error function (no scipy dependency)


def per_gene_pearson(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """``(G,)`` Pearson r per gene, across patients. NaN where preds or labels are constant for a gene."""
    preds = np.asarray(preds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    pc = preds - preds.mean(axis=0, keepdims=True)  # centred
    lc = labels - labels.mean(axis=0, keepdims=True)
    num = (pc * lc).sum(axis=0)
    den = np.sqrt((pc**2).sum(axis=0) * (lc**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den  # NaN where den == 0 (a constant gene)


def _rank_columns(x: np.ndarray) -> np.ndarray:
    """Ordinal ranks down each column (ties broken arbitrarily, matches ``aggregate._spearman``)."""
    return np.argsort(np.argsort(x, axis=0), axis=0).astype(np.float64)


def per_gene_spearman(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """``(G,)`` Spearman ρ per gene = per-gene Pearson of the column ranks. NaN where preds or labels are
    constant for a gene, matching ``per_gene_pearson``.

    The constant case must be excluded explicitly: ``argsort`` breaks ties by position, so it invents a
    strictly increasing ranking for a constant column. An unmeasured gene (constant in both predictions
    and labels) would then rank identically on both sides and score a perfect ρ = +1, inflating the
    median over a panel where such genes are common, where Pearson correctly reports it as undefined.
    """
    preds = np.asarray(preds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    rho = per_gene_pearson(_rank_columns(preds), _rank_columns(labels))
    with np.errstate(invalid="ignore"):
        constant = (np.nanmax(preds, axis=0) == np.nanmin(preds, axis=0)) | (
            np.nanmax(labels, axis=0) == np.nanmin(labels, axis=0)
        )
    rho[constant] = np.nan
    return rho


def significant_gene_count(preds: np.ndarray, labels: np.ndarray, *, alpha: float = 0.05) -> dict:
    """Count genes whose per-gene Pearson is significant after Benjamini-Hochberg FDR at ``alpha``.

    Two-sided p-value from the Fisher z-transform (``z = arctanh(r)*sqrt(n-3)``; exact for large n,
    intended for the pooled OOF, not small per-fold sets). Constant genes (undefined correlation) are
    excluded. This is SEQUOIA's criterion (1), "positive & significant", without its random-model
    baseline (criteria 2-3), so it is a permissive significance count, not the conservative one.
    """
    r = per_gene_pearson(preds, labels)
    r = r[~np.isnan(r)]
    n = int(np.asarray(preds).shape[0])
    if n < 4 or r.size == 0:
        return {"n_significant": 0, "n_scored": int(r.size), "alpha": float(alpha)}
    z = np.arctanh(np.clip(r, -0.999999, 0.999999)) * math.sqrt(n - 3)
    p = _erfc(np.abs(z) / math.sqrt(2.0))  # two-sided normal-tail p-value
    order = np.argsort(p)  # Benjamini-Hochberg step-up
    thresh = alpha * np.arange(1, p.size + 1) / p.size
    passing = p[order] <= thresh
    n_sig = int(np.nonzero(passing)[0].max() + 1) if passing.any() else 0
    return {"n_significant": n_sig, "n_scored": int(r.size), "alpha": float(alpha)}


def _bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values) for a 1-D array of p-values."""
    p = np.asarray(p, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]  # enforce monotonicity from the largest p down
    q = np.empty_like(q_sorted)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q


def _williams_t(r1: np.ndarray, r2: np.ndarray, r12: np.ndarray, n: int) -> np.ndarray:
    """Williams (1959) t for comparing two dependent correlations that share a variable, ``r1`` and
    ``r2`` both correlate with the ground truth, ``r12`` is the correlation between the two predictors.
    Steiger (1980) endorses this over Hotelling's T; df = n-3. Positive when r1 > r2. Vectorised over
    genes (formula matches ``psych::r.test``)."""
    determin = 1 - r1**2 - r2**2 - r12**2 + 2 * r1 * r2 * r12
    av = (r1 + r2) / 2
    cube = (1 - r12) ** 3
    denom = 2 * ((n - 1) / (n - 3)) * determin + av**2 * cube
    with np.errstate(invalid="ignore", divide="ignore"):
        return (r1 - r2) * np.sqrt((n - 1) * (1 + r12) / denom)


def conservative_significant_gene_count(
    trained_preds: np.ndarray, random_preds: np.ndarray, labels: np.ndarray, *, alpha: float = 0.05, fdr: float = 0.2
) -> dict:
    """SEQUOIA's well-predicted-gene count: the trained model must beat a random-init model of the same
    architecture, not merely beat zero correlation. Three per-gene criteria (all required):
      (1) ``r1 > 0`` and its Fisher-z p < ``alpha``  (r1 = trained-vs-truth Pearson);
      (2) ``r1 > r2`` by the Williams test, one-sided p < ``alpha`` and BH-adjusted p < ``fdr``
          (r2 = random-vs-truth Pearson; r12 = trained-vs-random);
      (3) ``rmse1 < rmse2``.
    Genes with any undefined correlation are excluded. Predictions/labels are (N, G), same patients."""
    trained_preds = np.asarray(trained_preds, dtype=np.float64)
    random_preds = np.asarray(random_preds, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n = int(labels.shape[0])
    r1 = per_gene_pearson(trained_preds, labels)
    r2 = per_gene_pearson(random_preds, labels)
    r12 = per_gene_pearson(trained_preds, random_preds)
    valid = ~(np.isnan(r1) | np.isnan(r2) | np.isnan(r12))
    if n < 4 or not valid.any():
        return {"n_significant": 0, "n_scored": int(valid.sum()), "alpha": float(alpha), "fdr": float(fdr)}

    rmse1 = np.sqrt(np.nanmean((trained_preds - labels) ** 2, axis=0))
    rmse2 = np.sqrt(np.nanmean((random_preds - labels) ** 2, axis=0))
    p1 = _erfc(np.abs(np.arctanh(np.clip(r1, -0.999999, 0.999999)) * math.sqrt(n - 3)) / math.sqrt(2.0))
    t = _williams_t(r1, r2, r12, n)
    with np.errstate(invalid="ignore"):
        p2 = 0.5 * _erfc(t / math.sqrt(2.0))  # one-sided (r1 > r2)
    q2 = np.ones_like(p2)
    q2[valid] = _bh_adjust(p2[valid])  # BH over the tested genes only

    passing = valid & (r1 > 0) & (p1 < alpha) & (r1 > r2) & (p2 < alpha) & (q2 < fdr) & (rmse1 < rmse2)
    return {"n_significant": int(passing.sum()), "n_scored": int(valid.sum()), "alpha": float(alpha), "fdr": float(fdr)}


def gene_metric_summary(preds: np.ndarray, labels: np.ndarray, *, top_k: int = 100) -> dict:
    """Summarise per-gene correlations across the panel.

    Returns ``gene_pearson_mean`` (headline), ``gene_pearson_topk`` (mean Pearson of the top-k
    best-predicted genes), ``gene_spearman_median``, and ``n_scored`` / ``n_genes`` / ``top_k``.
    Genes with an undefined correlation (constant in preds or labels) are excluded.
    """
    r = per_gene_pearson(preds, labels)
    rho = per_gene_spearman(preds, labels)
    n_genes = int(r.size)
    nan = float("nan")
    r_valid = r[~np.isnan(r)]
    rho_valid = rho[~np.isnan(rho)]
    n_scored = int(r_valid.size)
    if n_scored == 0:
        return {
            "gene_pearson_mean": nan,
            "gene_pearson_topk": nan,
            "gene_spearman_median": nan,
            "n_scored": 0,
            "n_genes": n_genes,
            "top_k": int(top_k),
        }
    k = min(int(top_k), n_scored)
    return {
        "gene_pearson_mean": float(r_valid.mean()),
        "gene_pearson_topk": float(np.sort(r_valid)[-k:].mean()),
        "gene_spearman_median": float(np.median(rho_valid)) if rho_valid.size else nan,
        "n_scored": n_scored,
        "n_genes": n_genes,
        "top_k": k,
    }
