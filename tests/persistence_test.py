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
"""Tests for dlux.eval.persistence: canonical dir, sentinel refuse, collision guard, copy."""

from __future__ import annotations

import json

import pytest
from dlux.eval.persistence import (
    claim_hydra_run_dir,
    persist_run_artifacts,
    persistent_run_dir,
    run_already_persisted,
)


def test_persistent_run_dir_layout():
    p = persistent_run_dir("/studies", "demo_study", "er_abmil_v1", "demo_cohort", "cancer_type_cv_o0_i0")
    assert p.as_posix() == "/studies/demo_study/runs/er_abmil_v1/demo_cohort/cancer_type_cv_o0_i0"


def test_persistent_run_dir_requires_names():
    for bad in (
        ("/runs", "", "exp", "cohort", "split"),  # empty study
        ("/runs", "study", "", "cohort", "split"),  # empty experiment
        ("/runs", "study", "exp", "", "split"),  # empty cohort
        ("/runs", "study", "exp", "cohort", ""),  # empty split
    ):
        with pytest.raises(ValueError):
            persistent_run_dir(*bad)


def test_run_already_persisted(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert run_already_persisted(run_dir) is False  # no sentinel yet
    (run_dir / "metadata.json").write_text("{}")
    assert run_already_persisted(run_dir) is True  # sentinel present -> complete


def test_claim_marker_written(tmp_path):
    out = tmp_path / "hydra_out"
    marker = claim_hydra_run_dir(out, experiment_name="exp", split_version="s0")
    written = json.loads((out / "run_marker.json").read_text())
    assert written["experiment_name"] == "exp" and written["split_version"] == "s0"
    assert marker["pid"] == written["pid"]


def test_claim_collision_raises(tmp_path):
    out = tmp_path / "hydra_out"
    claim_hydra_run_dir(out, experiment_name="exp", split_version="s0")
    # second claim on the SAME dir -> collision
    with pytest.raises(RuntimeError, match="collision"):
        claim_hydra_run_dir(out, experiment_name="exp2", split_version="s1")
    # a different dir is fine
    claim_hydra_run_dir(tmp_path / "other", experiment_name="exp2", split_version="s1")


def test_persist_copies_and_writes_sentinel_last(tmp_path):
    src = tmp_path / "eph"
    src.mkdir()
    (src / "best.ckpt").write_bytes(b"weights")
    (src / "test_predictions.npz").write_bytes(b"npz")
    run_dir = tmp_path / "canonical" / "exp" / "s0"

    persist_run_artifacts(
        run_dir=run_dir,
        artifacts={
            "best.ckpt": src / "best.ckpt",
            "test_predictions.npz": src / "test_predictions.npz",
            "missing.csv": src / "does_not_exist.csv",  # warned, not fatal
        },
        metadata={"experiment_name": "exp", "split_version": "s0", "auroc": 0.94},
        force=False,
    )
    assert (run_dir / "best.ckpt").read_bytes() == b"weights"
    assert (run_dir / "test_predictions.npz").exists()
    assert not (run_dir / "missing.csv").exists()
    meta = json.loads((run_dir / "metadata.json").read_text())
    assert meta["auroc"] == 0.94


def test_persist_force_clears_existing(tmp_path):
    run_dir = tmp_path / "canonical"
    run_dir.mkdir()
    (run_dir / "stale.txt").write_text("old")
    src = tmp_path / "eph"
    src.mkdir()
    (src / "best.ckpt").write_bytes(b"w")

    persist_run_artifacts(
        run_dir=run_dir,
        artifacts={"best.ckpt": src / "best.ckpt"},
        metadata={"ok": True},
        force=True,
    )
    assert not (run_dir / "stale.txt").exists()  # cleared by force
    assert (run_dir / "best.ckpt").exists()
    assert (run_dir / "metadata.json").exists()
