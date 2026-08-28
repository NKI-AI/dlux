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
"""Readable report over a finished tile-feature cache, written next to its ``manifest.json``.

The counterpart of ``analyze_db`` for the extraction stage: ``manifest.json`` and
``extraction_report.json`` already hold the numbers, but only as JSON. This renders them as
``summary.md`` + ``tiles.png``.

The report also recommends ``num_tiles`` for the fit-time bag subsampling, because that knob is
unreadable without this distribution: ``SelectRandomTiles`` pads a slide that has fewer tiles than
``num_tiles`` by sampling with replacement, so a value above the low end of the distribution buys
duplicate vectors rather than information. The recommendation is emitted as a snippet to paste into an
experiment config, since the right value depends on the cohort's tile counts and belongs with the rest
of a study's recipe.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
# silence matplotlib's superfluous INFO logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

# Ladder of round bag sizes. The recommendation is one of these, not a raw percentile,
# so two cohorts with near-identical distributions get the same number.
_BAG_SIZES = (10, 20, 50, 100, 200, 500, 1000, 2000, 5000)

_SNIPPET_TARGET = "catalog/experiment/<study>/<arm>.yaml"

# p75/p25 above this and the cohort holds two tissue-area populations, so a single bag size costs one
# of them dearly and the report declines to name one. An order of magnitude apart is not a tail.
_SPREAD_LIMIT = 10.0


@dataclass(frozen=True)
class PaddingCost:
    """What a given ``num_tiles`` costs on this distribution, from both directions.

    Too large and small slides pad: ``fraction`` is the share sampled with replacement
    (count < num_tiles) and ``mean_factor`` the mean ``num_tiles / count`` over them (1.0 when nothing
    pads). Too small and large slides are truncated: ``coverage`` is the share of all extracted tiles
    an epoch actually visits. The two move in opposite directions, which is why no single number is
    right without seeing both.
    """

    num_tiles: int
    n_padded: int
    fraction: float
    mean_factor: float
    coverage: float


def padding_cost(counts: list[int], num_tiles: int) -> PaddingCost:
    """The cost of a bag size on this distribution: what pads, and what never gets sampled."""
    array = np.asarray(counts, dtype=np.float64)
    padded = array[array < num_tiles]
    return PaddingCost(
        num_tiles=int(num_tiles),
        n_padded=int(padded.size),
        fraction=float(padded.size / array.size) if array.size else 0.0,
        mean_factor=float(np.mean(num_tiles / padded)) if padded.size else 1.0,
        coverage=float(np.minimum(array, num_tiles).sum() / array.sum()) if array.sum() else 0.0,
    )


def recommend_num_tiles(counts: list[int]) -> int:
    """The largest ladder size at or below the 25th percentile of tiles-per-slide.

    p25 leaves roughly three quarters of slides sampling without replacement. Never returns less than
    the smallest ladder size: below that the bag is too small to be a bag, and the answer is to tile at
    a finer mpp rather than to shrink the recipe.
    """
    if not counts:
        raise ValueError("cannot recommend num_tiles without any per-slide tile counts")
    p25 = float(np.percentile(np.asarray(counts, dtype=np.float64), 25))
    eligible = [size for size in _BAG_SIZES if size <= p25]
    return eligible[-1] if eligible else _BAG_SIZES[0]


def experiment_snippet(num_tiles: int) -> str:
    """The `num_tiles` override, in the shape an experiment config sets it."""
    return "augmentations:\n  tile_features:\n    fit:\n      num_tiles: " + str(num_tiles)


def _percentiles(counts: list[int]) -> Dict[str, float]:
    array = np.asarray(counts, dtype=np.float64)
    p5, p25, median, p75, p95 = (float(v) for v in np.percentile(array, [5, 25, 50, 75, 95]))
    return {
        "n_slides": float(array.size),
        "total": float(array.sum()),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p5": p5,
        "p25": p25,
        "median": median,
        "p75": p75,
        "p95": p95,
        "max": float(array.max()),
    }


def spread(counts: list[int]) -> float:
    """``p75 / p25`` of tiles-per-slide, how far apart the small and large slides are.

    A cohort of one specimen type sits near 1; mixing resections with biopsies pushes it into the tens,
    and no single bag size then serves both ends.
    """
    p25, p75 = (float(v) for v in np.percentile(np.asarray(counts, dtype=np.float64), [25, 75]))
    return p75 / max(p25, 1.0)


def cost_ladder(counts: list[int]) -> list[PaddingCost]:
    """Round bag sizes spanning this distribution, so a choice can be compared rather than trusted.

    One rung per quartile-ish landmark (p25/p50/p75/p95): the largest round size at or below that
    percentile, i.e. the size at which that share of slides still samples without replacement. Spans
    the distribution in a few rows however skewed it is, rather than walking the ladder densely at the
    bottom.
    """
    array = np.asarray(counts, dtype=np.float64)
    sizes = set()
    for percentile in (25, 50, 75, 95):
        cut = float(np.percentile(array, percentile))
        below = [size for size in _BAG_SIZES if size <= cut]
        sizes.add(below[-1] if below else _BAG_SIZES[0])
    return [padding_cost(counts, size) for size in sorted(sizes)]


def _plot_tiles(counts: list[int], recommended: Optional[int], path: Path) -> None:
    """Two panels: the tiles-per-slide histogram, and the share of slides retaining at least N tiles,
    the curve a bag size is read off. ``recommended`` is marked when one was named.

    Both axes are log-scaled, since tile counts routinely span three orders of magnitude within one
    cohort (a needle biopsy and a nephrectomy in the same sheet).
    """
    array = np.asarray(sorted(counts), dtype=np.float64)
    fig, (ax_hist, ax_curve) = plt.subplots(1, 2, figsize=(11, 4.2))

    edges = np.geomspace(max(array.min(), 1.0), max(array.max(), 2.0), num=40)
    ax_hist.hist(array, bins=edges, color="#4c72b0")
    ax_hist.set(xlabel="tiles per slide", ylabel="slides", title="tiles per slide", xscale="log")

    # P(count >= N): read a bag size off the y-value you are willing to pad.
    retained = 1.0 - np.arange(array.size, dtype=np.float64) / array.size
    ax_curve.step(array, retained, where="post", color="#4c72b0", lw=2)
    ax_curve.set(
        xlabel="num_tiles",
        ylabel="share of slides with >= num_tiles",
        ylim=(0.0, 1.02),
        title="slides sampled without replacement",
        xscale="log",
    )
    for axis in (ax_hist, ax_curve):
        axis.grid(True, lw=0.3, alpha=0.4)
        # Plain tick labels on the log axis. The default log formatter emits mathtext, which resolves
        # font families matplotlib does not ship here and warns once per family on every figure.
        axis.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.xaxis.set_minor_formatter(NullFormatter())
        if recommended is not None:
            axis.axvline(recommended, color="#55a868", lw=1.6, label=f"num_tiles {recommended}")
            axis.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _wh(size) -> str:
    """``[224, 224]`` -> ``224 x 224``; the raw pair for anything unexpected."""
    pair = list(size) if isinstance(size, (list, tuple)) else []
    return f"{pair[0]} x {pair[1]}" if len(pair) == 2 else str(size)


def _stat_table(stats: Dict[str, float]) -> list[str]:
    order = ("min", "p5", "p25", "median", "p75", "p95", "max", "mean")
    return [
        "| " + " | ".join(order) + " |",
        "| " + " | ".join("---" for _ in order) + " |",
        "| " + " | ".join(f"{stats[key]:.0f}" if key != "mean" else f"{stats[key]:.1f}" for key in order) + " |",
    ]


def render_summary(counts: list[int], manifest: dict, extraction: dict, out_dir: Path) -> Path:
    """Write ``summary.md`` + ``tiles.png`` into ``out_dir`` from already-read cache metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = _percentiles(counts)
    dispersion = spread(counts)
    # A dispersed cohort gets no recommendation, so nothing is marked on the figure either.
    recommended = None if dispersion > _SPREAD_LIMIT else recommend_num_tiles(counts)
    _plot_tiles(counts, recommended, out_dir / "tiles.png")

    grid = extraction.get("grid", {})
    # fp16 on disk and in the bag, so the per-row bag arithmetic below is exact.
    feature_dim = int(manifest.get("feature_dim") or 0)

    counts_block = extraction.get("counts", {})
    outcome = ("ok", "insufficient_tiles", "failed", "not_attempted")
    lines = [
        f"# {manifest.get('manifest_name')} / {manifest.get('model_name')}",
        "",
        "| | |",
        "| --- | --- |",
        f"| mpp | {grid.get('mpp')} |",
        f"| tile | {_wh(grid.get('tile_size'))} |",
        f"| overlap | {_wh(grid.get('tile_overlap'))} |",
        f"| mask threshold | {grid.get('mask_threshold')} |",
        f"| features | {feature_dim}-d {manifest.get('dtype')} |",
        f"| extracted | {str(manifest.get('created_at', ''))[:10]} |",
        "",
        "| " + " | ".join(outcome) + " |",
        "| " + " | ".join("---" for _ in outcome) + " |",
        "| " + " | ".join(str(counts_block.get(key, 0)) for key in outcome) + " |",
        "",
        "## tiles per slide",
        "",
        f"{stats['n_slides']:.0f} slides, {stats['total']:.0f} tiles, p75/p25 = {dispersion:.0f}",
        "",
        *_stat_table(stats),
        "",
        "![tiles per slide](tiles.png)",
        "",
        "## num_tiles",
        "",
        "| num_tiles | slides padded | copies per tile | tiles seen | bag size |",
        "| --- | --- | --- | --- | --- |",
    ]
    for cost in cost_ladder(counts):
        lines.append(
            f"| {cost.num_tiles} | {cost.n_padded} ({cost.fraction * 100:.0f}%) | "
            f"{cost.mean_factor:.1f}x | {cost.coverage * 100:.0f}% | "
            f"{cost.num_tiles * feature_dim * 2 / 1024:.0f} KiB |"
        )
    lines.append("")

    if dispersion > _SPREAD_LIMIT:
        lines += [
            f"**Two populations (p25 {stats['p25']:.0f}, p75 {stats['p75']:.0f}). No single value fits.**",
            "",
        ]
    else:
        lines += [
            f"**Recommended: {recommended}** "
            f"({(1 - padding_cost(counts, recommended).fraction) * 100:.0f}% of slides unpadded)",
            "",
            f"`{_SNIPPET_TARGET}`",
            "",
            "```yaml",
            experiment_snippet(recommended),
            "```",
            "",
        ]
    summary = out_dir / "summary.md"
    summary.write_text("\n".join(lines))
    return summary


def analyze_features(cache_path: Path) -> Optional[Path]:
    """Report on the finished feature cache at ``cache_path``.

    Reads ``manifest.json``, ``extraction_report.json`` and the per-slide tile counts (an attrs-only
    read per ``.h5``, so no features are loaded). Returns the ``summary.md`` path, or None when the
    cache holds no slide with tiles, there is nothing to describe and nothing to recommend.
    """
    from ahcore.data.stores.per_file_h5 import PerFileH5FeatureStore

    cache_path = Path(cache_path)
    manifest = json.loads((cache_path / "manifest.json").read_text())
    extraction = json.loads((cache_path / "extraction_report.json").read_text())

    store = PerFileH5FeatureStore(cache_path)
    counts = [n for sid in store.iter_ids() if (n := store.num_tiles(sid))]
    if not counts:
        return None
    return render_summary(counts, manifest, extraction, cache_path)
