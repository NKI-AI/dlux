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
"""Score-distribution beeswarm: predicted score by true class, over the pooled OOF or an external cohort.

Shows what an AUROC/AP summarises away, class overlap, where the 0.5 threshold falls, and the wrong-side
tail (one dot per patient). Classification only: binary (P(positive) split by label, dashed 0.5 line) and
multiclass one-vs-rest (P(class k), true-k vs not-k, no threshold line). Points are coloured by true class
only, so the panel carries no correct/incorrect verdict. Rendered at report time by aggregate and
evaluate_external into ``figures/score_dist_<field>.png``.

The grouping helpers are pure (arrays in, arrays out) and unit-test without a display. The renderers use
seaborn's beeswarm for point placement. The beeswarm saturates near 0/1 at large N (~800), that costs only
fine-grained density in the crowded extreme, not the overlap/threshold/tail the plot is for; markers shrink
with N and N is in the title so a dense block still means "many"."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

_NEG_COLOR, _POS_COLOR = "#4C72B0", "#C44E52"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _marker_size(n: int) -> float:
    """Shrink markers as N grows so fewer points clip in the crowded 0/1 extremes."""
    if n > 400:
        return 3.0
    if n > 150:
        return 4.5
    return 6.0


def binary_score_groups(scores, labels) -> dict:
    """Split predicted positive-class scores by true label. Pure, the plot's data core."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    return {
        "neg": scores[labels == 0],
        "pos": scores[labels == 1],
        "n": int(scores.size),
        "n_pos": int((labels == 1).sum()),
    }


def ovr_score_groups(probs, labels, k: int) -> dict:
    """One-vs-rest split of P(class k) into true-k vs not-k. Pure."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    col = probs[:, k]
    return {"in": col[labels == k], "out": col[labels != k], "n": int(labels.size), "n_in": int((labels == k).sum())}


def _swarm(ax, groups_xy, *, size: float, palette: dict) -> None:
    """Beeswarm of (category -> y-values) pairs onto ``ax``, one column per category in given order."""
    import warnings

    import seaborn as sns

    cats = [cat for cat, ys in groups_xy for _ in ys]
    ys = [float(y) for _, ys in groups_xy for y in ys]
    order = [cat for cat, _ in groups_xy]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # seaborn warns when a dense swarm cannot place every point
        sns.swarmplot(x=cats, y=ys, hue=cats, order=order, palette=palette, size=size, ax=ax, legend=False)


def plot_binary_score_distribution(scores, labels, *, field: str, path: Path, title_suffix: str = "") -> None:
    """Beeswarm of predicted P(positive) split by true class, with a dashed 0.5 threshold line."""
    g = binary_score_groups(scores, labels)
    if g["n"] == 0:
        return
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(4.6, 5.0))
    _swarm(
        ax,
        [("true neg", g["neg"]), ("true pos", g["pos"])],
        size=_marker_size(g["n"]),
        palette={"true neg": _NEG_COLOR, "true pos": _POS_COLOR},
    )
    ax.axhline(0.5, ls="--", lw=1.1, color="0.35")
    ax.annotate(
        "thr 0.5", xy=(1.0, 0.5), xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=8, color="0.35"
    )
    pct = 100.0 * g["n_pos"] / g["n"]
    ax.set(ylim=(-0.03, 1.03), xlabel="", ylabel="predicted P(positive)")
    ax.set_title(f"{field} — score by true class{title_suffix}\nN={g['n']}, pos={g['n_pos']} ({pct:.0f}%)")
    ax.margins(x=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_multiclass_score_distribution(
    probs, labels, *, field: str, path: Path, num_classes: Optional[int] = None, title_suffix: str = ""
) -> None:
    """One-vs-rest beeswarm small-multiples: per class k, P(class k) split by true-k vs not-k."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return
    k = num_classes or probs.shape[1]
    n = int(labels.size)
    plt = _pyplot()
    fig, axes = plt.subplots(1, k, figsize=(max(4.0, 3.0 * k), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for c, ax in zip(range(k), axes):
        g = ovr_score_groups(probs, labels, c)
        _swarm(
            ax,
            [(f"not {c}", g["out"]), (f"true {c}", g["in"])],
            size=_marker_size(n),
            palette={f"not {c}": _NEG_COLOR, f"true {c}": _POS_COLOR},
        )
        ax.set(ylim=(-0.03, 1.03), xlabel="", ylabel="predicted P(class)" if c == 0 else "")
        ax.set_title(f"class {c} (n={g['n_in']})")
        ax.margins(x=0.25)
    fig.suptitle(f"{field} — one-vs-rest score by true class{title_suffix} (N={n})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
