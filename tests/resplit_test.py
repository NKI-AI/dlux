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
"""Tests for dlux.eval.resplit: the trial rows train_resplit writes and compare_resplits reads, the trial
training seed, the invariants checked before any number is read, and the paired deltas and pooled null."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dlux.data.errors import BuildDbError
from dlux.eval.resplit import (
    MARGIN_PERCENTILE,
    Trial,
    check_invariants,
    compare_resplits,
    read_rows,
    row_path,
    train_seed,
    write_row,
)


def _row(experiment: str, seed: int, rep: int, metric: float, split: str | None = None) -> dict:
    return {
        "experiment": experiment,
        "field": "er",
        "seed": seed,
        "rep": rep,
        "metric_key": "auroc",
        "metric": metric,
        "n_declared": 10,
        "n_test": 10,
        "split_hash": split or f"split{seed}",
        "train_seed": train_seed(experiment, seed, rep),
        "git_sha": "abc",
    }


def _trial(experiment: str, seed: int, rep: int, metric: float, split: str | None = None) -> Trial:
    row = _row(experiment, seed, rep, metric, split)
    return Trial(**{k: row[k] for k in Trial.__dataclass_fields__})


def test_a_written_row_reads_back_as_its_trial_and_leaves_no_temp_file(tmp_path: Path):
    path = row_path(tmp_path, "baseline", "er", 7, 2)
    assert path.name == "baseline__er__s0007_r2.json"
    write_row(path, _row("baseline", 7, 2, 0.75))
    (trial,) = read_rows(tmp_path)
    assert (trial.experiment, trial.seed, trial.rep, trial.metric) == ("baseline", 7, 2, 0.75)
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


def test_a_row_with_a_string_metric_is_refused(tmp_path: Path):
    write_row(row_path(tmp_path, "baseline", "er", 0, 0), {**_row("baseline", 0, 0, 0.7), "metric": "0.7"})
    with pytest.raises(BuildDbError, match="stores 'metric' as a string"):
        read_rows(tmp_path)


def test_the_training_seed_differs_across_arm_seed_and_replicate_and_repeats_for_a_trial():
    seeds = {train_seed(e, s, r) for e in ("baseline", "meanmil") for s in range(20) for r in range(3)}
    assert len(seeds) == 2 * 20 * 3
    assert train_seed("baseline", 3, 1) == train_seed("baseline", 3, 1)


def _sweep() -> list[Trial]:
    """Arm b beats reference a by 0.1 at every seed. Each arm's null pair differs by its own amount."""
    trials = []
    for seed in range(4):
        trials += [_trial("a", seed, 0, 0.70), _trial("a", seed, 1, 0.70 + 0.01 * seed), _trial("a", seed, 2, 0.70)]
        trials += [_trial("b", seed, 0, 0.80), _trial("b", seed, 1, 0.78), _trial("b", seed, 2, 0.80)]
    return trials


def test_the_real_delta_is_paired_by_seed_and_the_null_pools_every_arm():
    comparison = compare_resplits(_sweep(), arms=["a", "b"], reference="a", field_name="er")
    (arm,) = comparison.arms
    np.testing.assert_allclose(arm.delta_real, [0.1] * 4)
    np.testing.assert_allclose(comparison.null_by_arm["a"], [0.0, 0.01, 0.02, 0.03])
    np.testing.assert_allclose(comparison.null_by_arm["b"], [-0.02] * 4)
    assert len(comparison.delta_null) == 8
    assert comparison.margin == pytest.approx(np.percentile(np.abs(comparison.delta_null), MARGIN_PERCENTILE))


def test_a_seed_missing_a_replicate_is_dropped_for_every_arm_and_counted():
    trials = [t for t in _sweep() if not (t.experiment == "b" and t.seed == 3 and t.rep == 2)]
    comparison = compare_resplits(trials, arms=["a", "b"], reference="a", field_name="er")
    assert comparison.n_seeds == 3 and comparison.arms[0].seeds == [0, 1, 2]
    assert comparison.dropped == {"a": 1, "b": 0}


def test_an_arm_without_its_null_replicates_is_refused():
    trials = [t for t in _sweep() if not (t.experiment == "b" and t.rep == 2)]
    with pytest.raises(BuildDbError, match="no seed has all of reps"):
        compare_resplits(trials, arms=["a", "b"], reference="a", field_name="er")


def test_arms_that_drew_different_splits_at_one_seed_are_refused():
    trials = _sweep() + [_trial("c", 0, 0, 0.75, split="another")]
    with pytest.raises(BuildDbError, match=r"seed\(s\) \[0\] drew more than one split"):
        check_invariants(trials)


def test_replicates_that_share_a_training_seed_are_refused():
    twin = _trial("a", 0, 1, 0.7)
    twin.train_seed = train_seed("a", 0, 0)
    with pytest.raises(BuildDbError, match="training seed"):
        check_invariants([_trial("a", 0, 0, 0.7), twin])
