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
"""Tests for dlux.data.layout: the study directory's on-disk layout (the db/ subdir)."""

from __future__ import annotations

from pathlib import Path

from dlux.data import layout


def test_db_and_sidecars_live_under_db_subdir():
    root = Path("/data/studies")
    assert layout.db_dir(root, "s") == root / "s" / "db"
    assert layout.db_path(root, "s", "c") == root / "s" / "db" / "c.db"
    assert layout.db_uri(root, "s", "c") == f"sqlite:///{root / 's' / 'db' / 'c.db'}"
    assert layout.contract_stats_path(root, "s", "c") == root / "s" / "db" / "c_contract_stats.json"
    assert layout.splits_csv_path(root, "s", "c") == root / "s" / "db" / "c_splits.csv"
    assert layout.splits_source_path(root, "s", "c") == root / "s" / "db" / "c_splits_source.json"
    assert layout.analysis_dir(root, "s", "c") == root / "s" / "db" / "c_analysis"


def test_sidecars_sit_beside_the_db():
    root = Path("/data/studies")
    parent = layout.db_path(root, "s", "c").parent
    for p in (
        layout.contract_stats_path(root, "s", "c"),
        layout.splits_csv_path(root, "s", "c"),
        layout.splits_source_path(root, "s", "c"),
        layout.analysis_dir(root, "s", "c"),
    ):
        assert p.parent == parent  # build_db derives sidecars from db_path().parent — keep that invariant


def test_accepts_str_or_path_root():
    assert layout.db_path("/data/studies", "s", "c") == Path("/data/studies/s/db/c.db")
