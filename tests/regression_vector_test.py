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
"""Expression endpoint: RNA matrix adapter + masked vector loss/metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from dlux.eval.gene_metrics import (
    conservative_significant_gene_count,
    gene_metric_summary,
    per_gene_pearson,
    per_gene_spearman,
    significant_gene_count,
)
from dlux.tasks.regression_vector import GenePearsonMean, regression_vector_loss

from ahcore.data.adapters.rna_expression import RNAExpressionAdapter


@dataclass
class _FakeSlideView:
    patient_code: str
    slide_id: str


def _write_matrix(tmp_path):
    # 3 patients x 4 genes (Entrez columns); values chosen distinct per cell.
    matrix = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
        index=["caseA", "caseB", "caseC"],
        columns=[100, 200, 300, 400],
    )
    path = tmp_path / "matrix.parquet"
    matrix.to_parquet(path)
    return path


def test_adapter_serves_patient_vector(tmp_path):
    adapter = RNAExpressionAdapter(_write_matrix(tmp_path))
    assert adapter.n_genes == 4 and adapter.genes == [100, 200, 300, 400]
    bundle = adapter(_FakeSlideView(patient_code="caseB", slide_id="s1"))
    vec = bundle.data["expression"]
    assert torch.allclose(vec, torch.tensor([5.0, 6.0, 7.0, 8.0])) and bundle.meta["slide_id"] == "s1"


def test_adapter_subselects_gene_panel(tmp_path):
    # gene_ids restricts + reorders the served vector to the active panel.
    adapter = RNAExpressionAdapter(_write_matrix(tmp_path), gene_ids=[300, 100])
    assert adapter.n_genes == 2 and adapter.genes == [300, 100]
    vec = adapter(_FakeSlideView(patient_code="caseB", slide_id="s1")).data["expression"]
    assert torch.allclose(vec, torch.tensor([7.0, 5.0]))  # caseB gene300=7, gene100=5, in panel order


def test_adapter_missing_patient_is_all_nan(tmp_path):
    adapter = RNAExpressionAdapter(_write_matrix(tmp_path))
    vec = adapter(_FakeSlideView(patient_code="not_in_matrix", slide_id="s9")).data["expression"]
    assert vec.shape == (4,) and bool(torch.isnan(vec).all())


def test_adapter_patient_code_coerced_to_str(tmp_path):
    # matrix indexed by numeric-looking ids -> lookup by the (str) patient_code must still hit.
    matrix = pd.DataFrame([[1.0, 2.0]], index=[42], columns=[10, 20])
    path = tmp_path / "m.parquet"
    matrix.to_parquet(path)
    vec = RNAExpressionAdapter(path)(_FakeSlideView(patient_code="42", slide_id="s")).data["expression"]
    assert torch.allclose(vec, torch.tensor([1.0, 2.0]))


def test_expression_loss_masks_nan():
    preds = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    target = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]])
    # valid residuals are all zero -> loss 0
    assert float(regression_vector_loss(preds, target)) == 0.0
    # a single nonzero residual on a valid element
    target2 = torch.tensor([[1.0, float("nan"), 3.0], [4.0, 5.0, float("nan")]])
    preds2 = preds.clone()
    preds2[0, 0] = 3.0  # residual 2 over 4 valid elements -> mean sq = 4/4 = 1.0
    assert math.isclose(float(regression_vector_loss(preds2, target2)), 1.0, rel_tol=1e-6)


def test_expression_loss_all_nan_is_zero_with_graph():
    preds = torch.zeros(2, 3, requires_grad=True)
    target = torch.full((2, 3), float("nan"))
    loss = regression_vector_loss(preds, target)
    assert float(loss) == 0.0 and loss.requires_grad  # graph intact for the backward pass


def test_gene_pearson_mean_metric_accumulates_across_batches():
    # gene0 perfectly correlated (r=+1), gene1 anti-correlated (r=-1), gene2 constant target (excluded)
    preds = torch.tensor([[0.0, 3.0, 9.0], [1.0, 2.0, 9.0], [2.0, 1.0, 9.0], [3.0, 0.0, 9.0]])
    target = torch.tensor([[0.0, 0.0, 5.0], [1.0, 1.0, 5.0], [2.0, 2.0, 5.0], [3.0, 3.0, 5.0]])
    m = GenePearsonMean(num_genes=3)
    m.update(preds[:2], target[:2])  # accumulation across two batches must equal one pass
    m.update(preds[2:], target[2:])
    assert abs(float(m.compute())) < 1e-6  # mean of {+1, -1}; constant gene2 excluded


def test_gene_pearson_mean_masks_nan_and_excludes_dead_genes():
    preds = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    target = torch.tensor([[0.0, float("nan")], [1.0, float("nan")], [2.0, float("nan")]])
    m = GenePearsonMean(num_genes=2)
    m.update(preds, target)
    assert abs(float(m.compute()) - 1.0) < 1e-6  # gene0 r=1; gene1 all-NaN target -> excluded


# -- per-gene metrics (aggregate-time) --------------------------------------
def test_per_gene_pearson_perfect_and_constant():
    labels = np.array([[1.0, 5.0, 2.0], [2.0, 5.0, 4.0], [3.0, 5.0, 6.0], [4.0, 5.0, 8.0]])
    preds = np.array([[2.0, 1.0, 8.0], [4.0, 9.0, 6.0], [6.0, 3.0, 4.0], [8.0, 7.0, 2.0]])
    r = per_gene_pearson(preds, labels)
    assert math.isclose(r[0], 1.0, abs_tol=1e-9)  # gene 0: preds = 2*labels -> r=1
    assert math.isnan(r[1])  # gene 1: labels constant -> undefined
    assert math.isclose(r[2], -1.0, abs_tol=1e-9)  # gene 2: preds decrease as labels increase -> r=-1


def test_per_gene_spearman_monotonic_nonlinear_is_one():
    labels = np.array([[1.0], [2.0], [3.0], [4.0]])
    preds = np.array([[1.0], [4.0], [9.0], [16.0]])  # monotone (squared) -> Pearson<1 but Spearman=1
    assert math.isclose(per_gene_spearman(preds, labels)[0], 1.0, abs_tol=1e-9)


def test_gene_metric_summary_excludes_constant_genes_and_topk():
    labels = np.array([[1.0, 5.0, 2.0], [2.0, 5.0, 4.0], [3.0, 5.0, 6.0], [4.0, 5.0, 8.0]])
    preds = np.array([[2.0, 1.0, 8.0], [4.0, 9.0, 6.0], [6.0, 3.0, 4.0], [8.0, 7.0, 2.0]])
    summ = gene_metric_summary(preds, labels, top_k=1)
    assert summ["n_genes"] == 3 and summ["n_scored"] == 2  # constant gene 1 excluded
    assert math.isclose(summ["gene_pearson_mean"], 0.0, abs_tol=1e-9)  # mean of {+1, -1}
    assert math.isclose(summ["gene_pearson_topk"], 1.0, abs_tol=1e-9)  # top-1 of {+1, -1}


def test_gene_metric_summary_all_constant_is_nan():
    preds = np.ones((4, 3))
    labels = np.arange(12.0).reshape(4, 3)
    summ = gene_metric_summary(preds, labels)
    assert summ["n_scored"] == 0 and math.isnan(summ["gene_pearson_mean"])


def test_conservative_count_requires_beating_random():
    n = 40
    lab = np.arange(n, dtype=float)
    alt = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    labels = np.stack([lab, lab, lab], axis=1)
    trained = np.stack([lab, alt, alt], axis=1)  # gene0 r1=1; gene1 r1~0; gene2 r1~0
    random_ = np.stack([alt, -alt, lab], axis=1)  # gene0 r2~0; gene1 r2~0; gene2 r2=1 (random beats trained)
    out = conservative_significant_gene_count(trained, random_, labels)
    assert out["n_scored"] == 3 and out["n_significant"] == 1  # only gene0 beats its random baseline


def test_significant_gene_count_flags_strong_excludes_constant_and_null():
    n = 40
    labels = np.stack([np.arange(n, dtype=float)] * 3, axis=1)
    preds = np.empty((n, 3))
    preds[:, 0] = np.arange(n)  # r=1 -> significant
    preds[:, 1] = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)  # |r|~0.04 -> not significant
    preds[:, 2] = 5.0  # constant -> undefined correlation, excluded
    out = significant_gene_count(preds, labels, alpha=0.05)
    assert out["n_scored"] == 2 and out["n_significant"] == 1
