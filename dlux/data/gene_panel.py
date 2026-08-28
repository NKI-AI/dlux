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
"""Gene panels for the expression endpoint: resolve a panel spec to the concrete gene subset.

A panel is a materialised gene-list (Entrez IDs), whether computed (e.g. highly-variable genes) or
declared (e.g. PAM50). Every panel funnels through the same resolution: intersect the panel's genes
with the cohort's RNA matrix and return the matrix gene-ids to model (the head width + the metric
axis follow). The panel is a fixed, frozen input. It must be selected unsupervised (never by
predictability), so resolution never touches predictions or labels.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional


def panel_path(spec: Optional[str], panels_dir: Path) -> Optional[Path]:
    """Resolve a ``task.gene_panel`` spec to a panel file, or None for the full panel.

    ``full`` / ``""`` / None -> None (all genes). An existing path -> that file. Otherwise a bare name
    -> ``<panels_dir>/<name>.csv`` (a computed panel living beside the cohort's RNA matrix)."""
    if spec in (None, "", "full"):
        return None
    p = Path(spec)
    return p if p.exists() else Path(panels_dir) / f"{spec}.csv"


def read_panel_entrez(path: Path) -> List[str]:
    """Entrez IDs (as strings, file order) from a panel CSV with an ``Entrez_Gene_Id`` column."""
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "Entrez_Gene_Id" not in reader.fieldnames:
            raise ValueError(f"panel file {path} must have an 'Entrez_Gene_Id' column (got {reader.fieldnames})")
        return [row["Entrez_Gene_Id"].strip() for row in reader if row["Entrez_Gene_Id"].strip()]


def resolve_panel(spec: Optional[str], panels_dir: Path, matrix_genes: List) -> List:
    """The cohort matrix gene-ids to model for ``spec``, in panel order, intersected with the matrix.

    ``full`` / None -> all ``matrix_genes`` (matrix order). Named/path panel -> its genes that are
    present in the matrix (dropping absent ones, reported). Matching is by string so an int-typed
    matrix column and a string panel id still line up; the returned ids keep the matrix's native type
    so ``matrix[ids]`` selects columns directly."""
    path = panel_path(spec, panels_dir)
    if path is None:
        return list(matrix_genes)
    if not path.exists():
        raise FileNotFoundError(f"gene panel '{spec}' not found at {path}")
    wanted = read_panel_entrez(path)
    by_str = {str(g): g for g in matrix_genes}
    resolved = [by_str[w] for w in wanted if w in by_str]
    if not resolved:
        raise ValueError(f"gene panel '{spec}': none of its {len(wanted)} genes are in the cohort matrix")
    missing = len(wanted) - len(resolved)
    if missing:
        print(f"[panel] '{spec}': {missing}/{len(wanted)} genes absent from the matrix, dropped ({len(resolved)} kept)")
    return resolved
