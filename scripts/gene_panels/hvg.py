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
"""Build a highly-variable-gene (HVG) panel from a cohort's RNA matrix.

Computes each gene's variance of log1p expression across ALL cohort patients, takes the top-N, and
writes a frozen Entrez gene-list (``panels/hvg<N>.csv``) that the expression task can select via
``task.gene_panel=hvg<N>``. Panel selection is unsupervised (target variance only, no predictions),
computed once on the full cohort so the panel is fold-invariant.

    python scripts/gene_panels/hvg.py --matrix .../rnaseq/matrix.parquet --n 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser("gene_panel_hvg")
    ap.add_argument("--matrix", type=Path, required=True, help="cohort rnaseq/matrix.parquet")
    ap.add_argument("--n", type=int, default=2000, help="panel size (top-N by log1p variance)")
    ap.add_argument("--genes", type=Path, default=None, help="genes.csv for Hugo symbols (default: matrix sibling)")
    ap.add_argument("--out-dir", type=Path, default=None, help="panels dir (default: <matrix parent>/panels)")
    args = ap.parse_args()

    matrix = pd.read_parquet(args.matrix)
    variance = np.nanvar(np.log1p(matrix.to_numpy(dtype=float)), axis=0)
    n = min(args.n, matrix.shape[1])
    top = np.argsort(-variance)[:n]  # highest-variance genes first
    panel = pd.DataFrame({"Entrez_Gene_Id": [matrix.columns[i] for i in top], "variance": variance[top]})

    genes_csv = args.genes or (args.matrix.parent / "genes.csv")
    if genes_csv.exists():
        hugo = pd.read_csv(genes_csv).astype({"Entrez_Gene_Id": str}).set_index("Entrez_Gene_Id")["Hugo_Symbol"]
        panel.insert(1, "Hugo_Symbol", panel["Entrez_Gene_Id"].astype(str).map(hugo).values)

    out_dir = args.out_dir or (args.matrix.parent / "panels")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hvg{n}.csv"
    panel.to_csv(out_path, index=False)
    print(f"[hvg] {n} genes (var {variance[top][-1]:.3g}–{variance[top][0]:.3g}) -> {out_path}")


if __name__ == "__main__":
    main()
