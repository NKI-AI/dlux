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
"""Per-tile attention logits captured during the test pass.

Stores raw logits, not softmax weights: softmax weights sum to 1 per slide, so their scale depends on
the tile count and is not comparable across slides. Softmax, temperature and top-k are applied
downstream.

Two constraints on consumers: the logit scale is arbitrary (softmax is shift-invariant, so only
relative comparisons are meaningful), and logits from different models carry independently arbitrary
offsets, so they must be normalised per model before any cross-model aggregation.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn

_ACCESSOR = "get_raw_attention_logits"

COORDS_KEY = "coords"
LENGTH_KEY = "length"
SLIDE_ID_KEY = "slide_id"


def find_attention_module(lit_module: nn.Module) -> nn.Module | None:
    """Returns the model's declared ``attention_encoder``, or None if it has no attention branch.

    Raises:
        ValueError: the declared encoder cannot report logits.
    """
    model = getattr(lit_module, "_model", lit_module)
    encoder = getattr(model, "attention_encoder", None)
    if encoder is not None and not hasattr(encoder, _ACCESSOR):
        raise ValueError(
            f"{type(model).__name__}.attention_encoder returned {type(encoder).__name__}, which has no "
            f"{_ACCESSOR}() — the attention contract is broken."
        )
    return encoder


@contextlib.contextmanager
def capture_input(module: nn.Module) -> Iterator[list[torch.Tensor]]:
    """Records the tensor passed to ``module`` on each forward, for the duration of the block.

    The logit accessor needs the encoder's input features, which the task assembles internally.
    """
    seen: list[torch.Tensor] = []

    def _hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
        if args:
            seen.append(args[0])

    handle = module.register_forward_pre_hook(_hook)
    try:
        yield seen
    finally:
        handle.remove()


def find_tile_modality(batch: dict[str, Any]) -> str | None:
    """Returns the batch key of the modality carrying tile ``coords``, or None.

    Raises:
        ValueError: several modalities carry coords, so the pairing is ambiguous.
    """
    named = [k for k, v in batch.items() if isinstance(v, dict) and COORDS_KEY in v]
    if not named:
        return None
    if len(named) > 1:
        raise ValueError(f"several modalities carry '{COORDS_KEY}' ({named}); cannot tell which owns the attention")
    return named[0]


def extract_records(
    attention_logits: torch.Tensor, batch: dict[str, Any], modality: str
) -> list[tuple[str, np.ndarray]]:
    """Pairs one batch's attention logits with tile coordinates, per slide.

    Args:
        attention_logits: ``(B, N, 1)`` or ``(B, N)`` raw logits over the *padded* bag.
        batch: the collated batch the logits came from.
        modality: batch key of the tile modality (see :func:`find_tile_modality`).

    Returns:
        ``(slide_id, array)`` per sample, where array is ``(n_tiles, 3) float32`` holding
        ``x, y`` in slide level-0 pixels and the tile's raw logit.

    Bags are zero-padded to the batch maximum (``shape_policy="padded"``), so each sample is sliced
    back to its true ``length`` from the adapter metadata.
    """
    logits = attention_logits.detach().float().cpu()
    if logits.ndim == 3:
        logits = logits.squeeze(-1)
    coords = batch[modality][COORDS_KEY].detach().cpu()
    metas = batch[modality]["meta"]

    records: list[tuple[str, np.ndarray]] = []
    for i, meta in enumerate(metas):
        n_tiles = int(meta[LENGTH_KEY])
        slide_coords = coords[i, :n_tiles].numpy().astype(np.float32)
        slide_logits = logits[i, :n_tiles].numpy().astype(np.float32)
        records.append((str(meta[SLIDE_ID_KEY]), np.column_stack([slide_coords, slide_logits])))
    return records


def write_attention_npz(
    records: list[tuple[str, np.ndarray]], path: Path, *, tiling: dict[str, Any] | None = None
) -> Path:
    """Writes ``<slide_id> -> (n_tiles, 3)`` plus the tile geometry, so consumers can place and size
    every tile without opening the feature cache."""
    payload: dict[str, Any] = {f"slide/{slide_id}": array for slide_id, array in records}
    payload["slide_ids"] = np.array([slide_id for slide_id, _ in records], dtype=object)
    if tiling:
        payload["mpp"] = np.float32(tiling["mpp"])
        payload["tile_size"] = np.asarray(tiling["tile_size"], dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def read_attention_npz(path: Path) -> dict[str, np.ndarray]:
    """Returns ``slide_id -> (n_tiles, 3)``, the inverse of :func:`write_attention_npz`."""
    with np.load(path, allow_pickle=True) as handle:
        return {key[len("slide/") :]: handle[key] for key in handle.files if key.startswith("slide/")}
