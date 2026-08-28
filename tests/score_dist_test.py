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
"""Tests for dlux.eval.score_dist: the pure class-conditional grouping + that the beeswarm renders."""

from __future__ import annotations

import numpy as np
from dlux.eval.score_dist import (
    binary_score_groups,
    ovr_score_groups,
    plot_binary_score_distribution,
    plot_multiclass_score_distribution,
)


def test_binary_score_groups_splits_by_label():
    g = binary_score_groups([0.1, 0.9, 0.4, 0.8], [0, 1, 0, 1])
    assert g["n"] == 4 and g["n_pos"] == 2
    assert sorted(g["neg"]) == [0.1, 0.4]
    assert sorted(g["pos"]) == [0.8, 0.9]


def test_ovr_score_groups_is_one_vs_rest():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    g = ovr_score_groups(probs, [0, 1, 2], 1)  # column 1, true-class-1 vs rest
    assert list(g["in"]) == [0.8]
    assert sorted(g["out"]) == [0.2, 0.2]
    assert g["n"] == 3 and g["n_in"] == 1


def test_plot_binary_writes_png(tmp_path):
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.uniform(0, 0.5, 20), rng.uniform(0.5, 1, 20)])
    labels = np.array([0] * 20 + [1] * 20)
    path = tmp_path / "score_dist_bin.png"
    plot_binary_score_distribution(scores, labels, field="demo", path=path)
    assert path.is_file() and path.stat().st_size > 0


def test_plot_binary_empty_is_noop(tmp_path):
    path = tmp_path / "empty.png"
    plot_binary_score_distribution([], [], field="demo", path=path)
    assert not path.exists()  # nothing to draw -> no file


def test_plot_multiclass_writes_png(tmp_path):
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 3, 30)
    probs = rng.dirichlet([1, 1, 1], 30)
    path = tmp_path / "score_dist_mc.png"
    plot_multiclass_score_distribution(probs, labels, field="demo", path=path, num_classes=3)
    assert path.is_file() and path.stat().st_size > 0
