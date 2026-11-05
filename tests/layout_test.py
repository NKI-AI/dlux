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

import pytest

from dlux.data import layout


def test_each_cohort_build_lives_in_its_own_db_folder():
    root = Path("/data/studies")
    db = root / "s" / "db" / "c"
    assert layout.db_dir(root, "s", "c") == db
    assert layout.db_path(root, "s", "c") == db / "manifest.db"
    assert layout.db_uri(root, "s", "c") == f"sqlite:///{db / 'manifest.db'}"
    assert layout.contract_stats_path(root, "s", "c") == db / "contract_stats.json"
    assert layout.splits_csv(db) == db / "splits.csv"
    assert layout.splits_source_json(db) == db / "splits_source.json"
    assert layout.analysis_dir(db) == db / "analysis"


def test_both_stages_of_a_comparison_share_its_folder():
    root = Path("/data/studies")
    comparison = root / "s" / "comparisons" / "robustness"
    assert layout.nested_cv_comparison_dir(root, "s", "robustness") == comparison / "nested_cv"
    assert layout.resplit_comparison_dir(root, "s", "robustness", "r1", "os") == comparison / "resplit_r1" / "os"


def test_a_replaced_dir_holds_only_the_new_files_and_survives_a_failed_write(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    (out / "stale.csv").write_text("old")

    with layout.replaced_dir(out) as tmp:
        (tmp / "summary.md").write_text("new")
    assert sorted(p.name for p in out.iterdir()) == ["summary.md"]

    with pytest.raises(RuntimeError):
        with layout.replaced_dir(out) as tmp:
            (tmp / "half.csv").write_text("x")
            raise RuntimeError("the writer failed")
    assert sorted(p.name for p in out.iterdir()) == ["summary.md"]
    assert not (tmp_path / "results.partial").exists()
