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
"""Persisting a run's fit-derived stream state, so a cohort with no fit split can still be scored.

Some streams carry state fitted on this fold's fit split, bulk RNA's per-gene μ/σ, for instance. Three
kinds exist and they behave differently when scoring an external cohort:

- **training-only** (survival's time bin edges): scoring never needs it.
- **inference-relevant and already recorded** (a z-scored regression target's μ/σ, in ``metadata.json``).
- **inference-relevant and otherwise lost**, what this module exists for. Recomputing such state on the
  cohort being scored would silently change the model's input distribution, so it is replayed instead.

A modality opts in by defining ``fit_state() -> dict``. One without it simply has nothing to record.
This module never learns what any particular state means, it visits a task's streams, writes what they
hand back, and reads it back keyed by stream name. What the payload is, and how to restore it, belongs
to the modality (see ``BulkRNA.fit_state`` / ``BulkRNA.replay_stats``).

The file is a single ``.npz`` because these payloads are arrays: a 2 000-gene panel is ~16 KB per fold.
Callers choose the directory, training writes it beside the fold's other artifacts, external eval reads
it from the fold it is scoring, so this module knows nothing about run layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

STATE_FILENAME = "stream_state.npz"

# Payload keys are namespaced "<stream>/<field>" so one flat npz can hold every stream's state without
# the writer or reader knowing any stream's field names.
_SEP = "/"


def collect_stream_state(task: Any) -> Dict[str, dict]:
    """Every stream's fit-derived state, keyed by stream name. Streams that record nothing are omitted."""
    out: Dict[str, dict] = {}
    for stream in task.streams:
        fit_state = getattr(stream, "fit_state", None)
        if fit_state is None:
            continue
        state = fit_state()
        if state:
            out[stream.key] = state
    return out


def apply_stream_state(task: Any, state: Dict[str, dict]) -> None:
    """Install recorded state onto a task's streams, the mirror of ``collect_stream_state``.

    Why this exists rather than rebuilding the task: fit-derived state is per fold, but scoring an
    external cohort runs many folds' models over one dataset. Everything else about a stream is
    fold-independent, so the dataset is built once and only the state is swapped between folds.

    **This is sound only while no modality's adapter captures fit-derived state**, an adapter is
    constructed with the dataset and outlives the swap, so state baked into one would silently keep its
    original value. Today that holds (``BulkRNA.adapter`` captures the matrix path, gene ids and panel
    name; the statistics are applied later, at forward time) and ``BulkRNA.load_fit_state`` enforces the
    part that could drift by refusing a changed gene set. A future modality that standardises inside its
    adapter must not rely on this path.

    A stream with recorded state that cannot consume it is an error, not something to skip: it means the
    record and the code have diverged, and scoring would silently use whatever the stream was built with.
    """
    by_key = {stream.key: stream for stream in task.streams}
    for stream_key, payload in state.items():
        stream = by_key.get(stream_key)
        if stream is None:
            continue  # the record covers a stream this task does not declare; not this function's call
        loader = getattr(stream, "load_fit_state", None)
        if loader is None:
            raise ValueError(
                f"stream '{stream_key}' ({type(stream).__name__}) has recorded fit state but no "
                f"load_fit_state() to consume it — the record and the modality have diverged."
            )
        loader(payload)


def write_stream_state(task: Any, out_dir: str | Path) -> Optional[Path]:
    """Write the task's fit-derived stream state into ``out_dir``. Returns the path, or None when no
    stream had any, the common case, and one that must not leave an empty file behind to be read back."""
    state = collect_stream_state(task)
    if not state:
        return None
    flat: Dict[str, Any] = {}
    for stream_key, payload in state.items():
        for field, value in payload.items():
            flat[f"{stream_key}{_SEP}{field}"] = np.asarray(value)
    path = Path(out_dir) / STATE_FILENAME
    np.savez(path, **flat)
    return path


def read_stream_state(run_dir: str | Path) -> Dict[str, dict]:
    """Read back what ``write_stream_state`` wrote. An absent file means the run recorded nothing, which
    is not an error here: whether a missing state is fatal is the consuming modality's call, since only
    it knows whether that state is needed to score."""
    path = Path(run_dir) / STATE_FILENAME
    if not path.exists():
        return {}
    out: Dict[str, dict] = {}
    with np.load(path, allow_pickle=True) as data:
        for flat_key in data.files:
            stream_key, _, field = flat_key.partition(_SEP)
            value = data[flat_key]
            # 0-d arrays are scalars that survived a round trip through np.asarray (e.g. a panel name).
            out.setdefault(stream_key, {})[field] = value.item() if value.ndim == 0 else value
    return out
