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
"""Build a curated gene panel from an MSigDB GMT (e.g. Hallmark, KEGG, GO).

Reads a GMT (one gene set per line), unions the chosen set(s), maps to Entrez if the GMT holds
symbols, intersects with the cohort's RNA matrix, and writes a frozen panel (``panels/<name>.csv``)
usable via ``task.gene_panel=<name>`` — the same artifact as an HVG panel. The GMT is an external
input (a path arg, downloaded from MSigDB), never copied into the repo.

    python scripts/gene_panels/from_gmt.py \
        --gmt h.all.v2024.1.Hs.entrez.gmt --matrix .../rnaseq/matrix.parquet --name hallmark
    python scripts/gene_panels/from_gmt.py \
        --gmt c2.cp.kegg.v2024.1.Hs.entrez.gmt --sets KEGG_WNT_SIGNALING_PATHWAY --name wnt ...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_gmt(path: Path) -> dict:
    """Parse a GMT: each line is ``<set_name>\\t<description>\\t<gene>...`` -> {set_name: [genes]}."""
    sets = {}
    with Path(path).open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            sets[parts[0]] = [g.strip() for g in parts[2:] if g.strip()]
    return sets


def main() -> None:
    ap = argparse.ArgumentParser("gene_panel_from_gmt")
    ap.add_argument("--gmt", type=Path, required=True, help="MSigDB GMT file (Entrez IDs, or symbols with --symbols)")
    ap.add_argument("--matrix", type=Path, required=True, help="cohort rnaseq/matrix.parquet")
    ap.add_argument("--name", type=str, required=True, help="panel name -> panels/<name>.csv")
    ap.add_argument("--sets", type=str, default=None, help="comma-separated set names (default: union of ALL sets)")
    ap.add_argument("--symbols", action="store_true", help="GMT holds Hugo symbols -> map to Entrez via genes.csv")
    ap.add_argument(
        "--genes", type=Path, default=None, help="genes.csv for symbol->Entrez / Hugo (default: matrix sibling)"
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="panels dir (default: <matrix parent>/panels)")
    args = ap.parse_args()

    sets = read_gmt(args.gmt)
    chosen = [s.strip() for s in args.sets.split(",")] if args.sets else list(sets)
    unknown = [s for s in chosen if s not in sets]
    if unknown:
        raise SystemExit(f"[gmt] set(s) not in {args.gmt.name}: {unknown}\n  available: {sorted(sets)[:8]}...")

    seen, union = set(), []  # union across chosen sets, first-seen order
    for name in chosen:
        for gene in sets[name]:
            if gene not in seen:
                seen.add(gene)
                union.append(gene)

    genes = pd.read_csv(args.genes or (args.matrix.parent / "genes.csv"))
    genes["Entrez_Gene_Id"] = genes["Entrez_Gene_Id"].astype(str)
    if args.symbols:  # GMT symbols -> Entrez
        hugo_to_entrez = genes.dropna(subset=["Hugo_Symbol"]).drop_duplicates("Hugo_Symbol").set_index("Hugo_Symbol")
        union = [hugo_to_entrez.loc[g, "Entrez_Gene_Id"] for g in union if g in hugo_to_entrez.index]

    matrix = pd.read_parquet(args.matrix)
    by_str = {str(c): c for c in matrix.columns}  # str-match, keep matrix native ids for matrix[ids]
    present = [by_str[g] for g in union if g in by_str]
    if not present:
        raise SystemExit(f"[gmt] none of the {len(union)} panel genes are in the matrix")

    hugo = genes.set_index("Entrez_Gene_Id")["Hugo_Symbol"]
    panel = pd.DataFrame({"Entrez_Gene_Id": present})
    panel["Hugo_Symbol"] = panel["Entrez_Gene_Id"].astype(str).map(hugo).values

    out_dir = args.out_dir or (args.matrix.parent / "panels")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}.csv"
    panel.to_csv(out_path, index=False)
    print(f"[gmt] {args.name}: {len(chosen)} set(s), {len(union)} genes, {len(present)} in matrix -> {out_path}")


if __name__ == "__main__":
    main()
