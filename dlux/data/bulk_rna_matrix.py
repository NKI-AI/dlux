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
"""Locating and reading the RNA-expression modality's data.

The built expression matrix is a derived artifact of the cohort's preprocessing (``scripts/cohort_builders/<cohort>/rna.py``),
not a declared cohort input, so its location is a convention (``<cohorts_dir>/<cohort>/rnaseq/matrix.parquet``)
resolved here rather than reconstructed at every call site. Keyed by patient in the matrix index.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


def matrix_path(cohorts_dir: Union[str, Path], cohort_name: str) -> Path:
    """The cohort's built RNA-seq expression matrix (``regression_vector`` target)."""
    return Path(cohorts_dir) / str(cohort_name) / "rnaseq" / "matrix.parquet"


def panels_dir(cohorts_dir: Union[str, Path], cohort_name: str) -> Path:
    """Where the cohort's frozen gene panels live (siblings of the matrix)."""
    return matrix_path(cohorts_dir, cohort_name).parent / "panels"


def matrix_summary(path: Optional[Union[str, Path]]) -> Optional[dict]:
    """Characterise the matrix for display: patient/gene counts + per-gene mean log1p expression
    (the dynamic-range view). None if the matrix is absent."""
    if path is None or not Path(path).exists():
        return None
    import numpy as np
    import pandas as pd

    mat = pd.read_parquet(path)
    gene_means = np.nanmean(np.log1p(mat.to_numpy(dtype=float)), axis=0)  # mean per gene across patients
    return {"n_patients": int(mat.shape[0]), "n_genes": int(mat.shape[1]), "gene_means": gene_means}
