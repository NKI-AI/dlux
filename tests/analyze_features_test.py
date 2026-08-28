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
"""Tests for the feature-cache report: the num_tiles recommendation, the padding arithmetic it is
justified by, and the rendered summary."""

from __future__ import annotations

import pytest
from dlux.data.analyze_features import (
    cost_ladder,
    experiment_snippet,
    padding_cost,
    recommend_num_tiles,
    render_summary,
    spread,
)

_MANIFEST = {
    "manifest_name": "demo_cohort",
    "model_name": "uni2",
    "feature_dim": 1536,
    "dtype": "float16",
    "created_at": "2026-07-30T00:00:00+00:00",
}
_EXTRACTION = {
    "grid": {"mpp": 2.0, "tile_size": [224, 224], "tile_overlap": [0, 0], "mask_threshold": 0.5},
    "counts": {"ok": 100, "insufficient_tiles": 3, "failed": 0, "not_attempted": 0},
}


def test_recommendation_sits_on_the_ladder_at_or_below_p25():
    counts = list(range(50, 150))  # p25 ~= 74.75
    assert recommend_num_tiles(counts) == 50  # 100 would exceed p25, 50 is the next rung down


def test_recommendation_floors_at_the_smallest_bag():
    # A grid this coarse leaves single-digit bags; the answer is a finer mpp, not a 4-tile recipe.
    assert recommend_num_tiles([2, 3, 4, 5]) == 10


def test_recommendation_needs_counts():
    with pytest.raises(ValueError, match="without any per-slide tile counts"):
        recommend_num_tiles([])


def test_padding_cost_counts_only_slides_below_the_bag():
    cost = padding_cost([50, 50, 100, 200], num_tiles=100)
    assert cost.n_padded == 2 and cost.fraction == pytest.approx(0.5)
    assert cost.mean_factor == pytest.approx(2.0)  # 100/50 for both padded slides
    # A slide with exactly num_tiles samples without replacement, so it is not padded.
    assert padding_cost([100, 100], num_tiles=100).n_padded == 0
    assert padding_cost([100, 100], num_tiles=100).mean_factor == 1.0


def test_coverage_is_the_share_of_tiles_an_epoch_visits():
    # 400 tiles cached; a bag of 100 visits 50 + 50 + 100 + 100 = 300 of them.
    assert padding_cost([50, 50, 100, 200], num_tiles=100).coverage == pytest.approx(300 / 400)
    # A bag at or above the largest slide sees everything.
    assert padding_cost([50, 200], num_tiles=200).coverage == pytest.approx(1.0)


def test_padding_and_coverage_move_in_opposite_directions():
    counts = list(range(20, 400))
    sizes = (32, 64, 128, 256)
    assert [padding_cost(counts, n).fraction for n in sizes] == sorted(
        [padding_cost(counts, n).fraction for n in sizes]
    )
    assert [padding_cost(counts, n).coverage for n in sizes] == sorted(
        [padding_cost(counts, n).coverage for n in sizes]
    )


def test_spread_separates_one_population_from_two():
    assert spread(list(range(80, 120))) < 2.0  # one specimen type
    assert spread([5] * 50 + [2000] * 50) > 100.0  # biopsies and resections in one sheet


def test_cost_ladder_spans_the_distribution():
    counts = [5] * 50 + [2000] * 50
    rows = cost_ladder(counts)
    assert [r.num_tiles for r in rows] == sorted(r.num_tiles for r in rows)
    # Spans both populations in a handful of rows rather than walking the low end densely.
    assert rows[0].num_tiles <= 10 and rows[-1].num_tiles >= 1000
    assert len(rows) <= 4


def test_snippet_is_the_experiment_override_path():
    assert experiment_snippet(64) == "augmentations:\n  tile_features:\n    fit:\n      num_tiles: 64"


def test_render_summary_recommends_on_a_single_population(tmp_path):
    counts = list(range(50, 150))
    summary = render_summary(counts, _MANIFEST, _EXTRACTION, tmp_path)
    assert summary.name == "summary.md" and (tmp_path / "tiles.png").exists()
    text = summary.read_text()
    assert "num_tiles: 50" in text and "```yaml" in text
    assert "demo_cohort" in text and "uni2" in text
    assert "insufficient_tiles" in text and "| 100 | 3 | 0 | 0 |" in text
    assert "tiles seen" in text


def test_render_summary_declines_to_recommend_on_two_populations(tmp_path):
    summary = render_summary([5] * 50 + [2000] * 50, _MANIFEST, _EXTRACTION, tmp_path)
    text = summary.read_text()
    assert "Two populations" in text
    assert "```yaml" not in text  # no snippet to paste, because no number is defensible
    assert "tiles seen" in text  # the table is still there to choose from
