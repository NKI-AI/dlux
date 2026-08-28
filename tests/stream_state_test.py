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
"""Tests for persisting and replaying fit-derived stream state (dlux.modalities.state + BulkRNA)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dlux.modalities.bulk_rna import BulkRNA
from dlux.modalities.state import (
    apply_stream_state,
    collect_stream_state,
    read_stream_state,
    write_stream_state,
)

_GENES = ["7503", "1234", "9999"]


def _rna(key="rna", ids=None, panel="hvg2000"):
    ids = _GENES if ids is None else ids
    return BulkRNA(
        key=key,
        matrix_path="/nonexistent/matrix.parquet",
        gene_means=np.array([1.0, 2.0, 3.0][: len(ids)], dtype=np.float32),
        gene_stds=np.array([0.5, 1.5, 2.5][: len(ids)], dtype=np.float32),
        gene_ids=ids,
        panel_name=panel,
    )


class _Task:
    """Minimal stand-in for the `streams` surface the collector uses."""

    def __init__(self, *streams):
        self.streams = streams


class _Stateless:
    key = "tiles"  # a modality with no fit-derived state defines no fit_state()


# -- what gets recorded -------------------------------------------------------
def test_gene_identities_travel_with_the_statistics():
    # A panel NAME does not pin down the genes: hvg2000 is computed per cohort, so two cohorts' hvg2000
    # are different gene sets. Replaying by name would score a model on the wrong genes.
    state = _rna().fit_state()
    assert [str(g) for g in state["gene_ids"]] == _GENES
    assert state["panel_name"] == "hvg2000"
    assert list(state["gene_means"]) == pytest.approx([1.0, 2.0, 3.0])


def test_streams_without_fit_state_are_omitted():
    assert set(collect_stream_state(_Task(_rna(), _Stateless()))) == {"rna"}
    assert collect_stream_state(_Task(_Stateless())) == {}


# -- round trip ---------------------------------------------------------------
def test_write_read_round_trip_preserves_every_stream(tmp_path):
    task = _Task(_rna(key="rna_a"), _rna(key="rna_b", ids=["7503", "1234"], panel="hallmark"), _Stateless())
    assert write_stream_state(task, tmp_path) is not None

    back = read_stream_state(tmp_path)
    assert set(back) == {"rna_a", "rna_b"}
    assert [str(g) for g in back["rna_a"]["gene_ids"]] == _GENES
    assert back["rna_b"]["panel_name"] == "hallmark"  # 0-d string survives the npz round trip as a scalar
    assert list(back["rna_b"]["gene_stds"]) == pytest.approx([0.5, 1.5])


def test_no_state_writes_no_file(tmp_path):
    # An empty file would read back as "recorded nothing", which is true but wasteful -- and worse, it
    # would look like a run that DID record state for a stream that has none.
    assert write_stream_state(_Task(_Stateless()), tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_absent_file_reads_as_nothing_recorded(tmp_path):
    # Not an error here: whether MISSING state is fatal is the consuming modality's call.
    assert read_stream_state(tmp_path) == {}


# -- replay -------------------------------------------------------------------
def _matrix(tmp_path, columns):
    path = tmp_path / "matrix.parquet"
    pd.DataFrame(np.arange(2 * len(columns), dtype=float).reshape(2, len(columns)), columns=columns).to_parquet(path)
    return path


def test_replay_restores_recorded_stats_and_ignores_matrix_order(tmp_path):
    # The external matrix carries the recorded genes in a DIFFERENT order plus extras; the recorded
    # order is what the model expects, so that is what comes back.
    path = _matrix(tmp_path, ["9999", "extra", "7503", "1234"])
    ids, means, stds = BulkRNA.replay_stats(_rna().fit_state(), key="rna", matrix_path=path, gene_panel="hvg2000")
    assert ids == _GENES  # recorded order, NOT the matrix's
    assert list(means) == pytest.approx([1.0, 2.0, 3.0])
    assert list(stds) == pytest.approx([0.5, 1.5, 2.5])


def test_replay_refuses_when_a_recorded_gene_is_absent(tmp_path):
    path = _matrix(tmp_path, ["7503", "1234"])  # 9999 missing
    with pytest.raises(ValueError, match="recorded genes are absent"):
        BulkRNA.replay_stats(_rna().fit_state(), key="rna", matrix_path=path, gene_panel="hvg2000")


def test_replay_refuses_a_different_panel_than_the_run_trained_on(tmp_path):
    path = _matrix(tmp_path, _GENES)
    with pytest.raises(ValueError, match="recorded gene_panel"):
        BulkRNA.replay_stats(_rna().fit_state(), key="rna", matrix_path=path, gene_panel="hallmark")


# -- per-fold swapping --------------------------------------------------------
# Scoring an external cohort runs many folds' models over ONE dataset, and each fold fitted its own
# statistics. Only the statistics may move between folds.
def test_swapping_statistics_between_folds():
    stream = _rna()
    fold3 = {
        "gene_ids": np.array(_GENES),
        "gene_means": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "gene_stds": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "panel_name": "hvg2000",
    }
    apply_stream_state(_Task(stream), {"rna": fold3})
    assert list(stream.gene_means.numpy()) == pytest.approx([10.0, 20.0, 30.0])
    assert list(stream.gene_stds.numpy()) == pytest.approx([1.0, 2.0, 3.0])


def test_swapping_refuses_a_changed_gene_set():
    # THE guard that makes swapping safe. The adapter -- and so which matrix columns the dataset reads,
    # in which order -- was fixed at construction and outlives the swap, so a different gene list here
    # would standardise the right numbers in the wrong places.
    stream = _rna()
    other = {
        "gene_ids": np.array(["1111", "2222", "3333"]),
        "gene_means": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "gene_stds": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        "panel_name": "hvg2000",
    }
    with pytest.raises(ValueError, match="gene set cannot"):
        apply_stream_state(_Task(stream), {"rna": other})


def test_zero_std_stays_clamped_after_a_swap():
    # A constant gene must not divide by zero -- the clamp lives in one place so construction and replay
    # cannot drift apart.
    stream = _rna()
    apply_stream_state(
        _Task(stream),
        {
            "rna": {
                "gene_ids": np.array(_GENES),
                "gene_means": np.zeros(3, dtype=np.float32),
                "gene_stds": np.array([0.0, 2.0, 0.0], dtype=np.float32),
                "panel_name": "hvg2000",
            }
        },
    )
    assert list(stream.gene_stds.numpy()) == pytest.approx([1.0, 2.0, 1.0])


def test_recorded_state_a_stream_cannot_consume_is_an_error():
    # The record and the modality have diverged; scoring would silently use whatever it was built with.
    with pytest.raises(ValueError, match="no .*load_fit_state"):
        apply_stream_state(_Task(_Stateless()), {"tiles": {"anything": np.array([1.0])}})


def test_state_for_an_undeclared_stream_is_left_alone():
    stream = _rna()
    apply_stream_state(_Task(stream), {"some_other_stream": {"gene_ids": np.array([])}})
    assert list(stream.gene_means.numpy()) == pytest.approx([1.0, 2.0, 3.0])


# -- a fold that recorded nothing must not pass silently -----------------------
# `apply_stream_state` with an empty payload is a NO-OP, so an unguarded loop would leave the previous
# fold's statistics installed and score this fold's model against another fold's standardisation.
def test_empty_payload_is_a_no_op_hence_must_be_caught_upstream():
    stream = _rna()
    apply_stream_state(_Task(stream), {})  # no exception, and...
    assert list(stream.gene_means.numpy()) == pytest.approx([1.0, 2.0, 3.0])  # ...nothing changed

    # so a fold whose state is missing keeps whatever the PREVIOUS fold installed:
    apply_stream_state(_Task(stream), {"rna": {**_rna().fit_state(), "gene_means": np.array([9.0, 9.0, 9.0])}})
    apply_stream_state(_Task(stream), {})  # the "missing state" fold
    assert list(stream.gene_means.numpy()) == pytest.approx([9.0, 9.0, 9.0])  # still fold N-1's -- the hazard


def test_read_stream_state_of_a_fold_that_recorded_nothing(tmp_path):
    """The signal callers must act on: absent file -> {}, indistinguishable from a stateless task."""
    assert read_stream_state(tmp_path) == {}
