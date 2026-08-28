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
"""Gene-panel resolution: spec -> concrete matrix gene subset."""

from __future__ import annotations

import pytest
from dlux.data.gene_panel import panel_path, resolve_panel

_MATRIX_GENES = [100, 200, 300, 400]  # native (int) matrix columns, like the real RSEM matrix


def _panel_file(tmp_path, name, entrez):
    p = tmp_path / f"{name}.csv"
    p.write_text("Entrez_Gene_Id,Hugo_Symbol\n" + "\n".join(f"{e},G{e}" for e in entrez) + "\n")
    return p


def test_full_returns_all_genes_in_matrix_order():
    assert resolve_panel("full", "/nonexistent", _MATRIX_GENES) == _MATRIX_GENES
    assert resolve_panel(None, "/nonexistent", _MATRIX_GENES) == _MATRIX_GENES


def test_named_panel_subselects_in_panel_order(tmp_path):
    _panel_file(tmp_path, "sub", [300, 100])  # panel order, subset of the matrix
    got = resolve_panel("sub", tmp_path, _MATRIX_GENES)
    assert got == [300, 100] and all(isinstance(g, int) for g in got)  # native int, ready for matrix[ids]


def test_string_panel_ids_match_int_matrix_columns(tmp_path):
    # panel file carries Entrez as strings; matrix columns are ints -> must still line up.
    _panel_file(tmp_path, "sub", ["200", "400"])
    assert resolve_panel("sub", tmp_path, _MATRIX_GENES) == [200, 400]


def test_absent_genes_dropped(tmp_path):
    _panel_file(tmp_path, "sub", [100, 999, 300])  # 999 not in the matrix
    assert resolve_panel("sub", tmp_path, _MATRIX_GENES) == [100, 300]


def test_no_overlap_raises(tmp_path):
    _panel_file(tmp_path, "sub", [999, 998])
    with pytest.raises(ValueError):
        resolve_panel("sub", tmp_path, _MATRIX_GENES)


def test_missing_panel_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_panel("does_not_exist", tmp_path, _MATRIX_GENES)


def test_panel_path_resolution(tmp_path):
    assert panel_path("full", tmp_path) is None
    assert panel_path(None, tmp_path) is None
    existing = _panel_file(tmp_path, "here", [100])
    assert panel_path(str(existing), tmp_path) == existing  # an existing path is used directly
    assert panel_path("hvg2000", tmp_path) == tmp_path / "hvg2000.csv"  # a bare name -> panels_dir/<name>.csv
