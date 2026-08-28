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
"""Aggregate a nested-CV sweep's per-fold predictions into per-outer-fold metrics.

Pure numpy/sklearn (no ahcore/torch), so trivially testable. Reads only the canonical
run dirs a completed sweep left behind, each fold's ``test_predictions.npz`` gated by
its ``metadata.json`` sentinel, so a partially-finished sweep aggregates whatever exists.

**Aggregation unit = the outer-fold ensemble.** For each outer fold ``o`` the ``n_inner``
inner-replicate models predict the same held-out TEST patients (shared-TEST invariant). We
roll each replicate's slide probs to patient level (mean), then average across replicates
(mean-of-probs) to get one out-of-fold prediction per patient. Metrics are computed per
outer fold (→ mean ± std) and on the pooled out-of-fold set (the outer folds partition the
cohort). Binary, multiclass, regression, expression and survival endpoints each have their own
aggregation core (survival's label is a coupled ``(time, event)`` pair scored by Harrell's C).

**Metrics.** Binary: AUROC + AP are threshold-free; accuracy, balanced accuracy, sensitivity,
specificity, precision and F1 at a fixed threshold 0.5, deliberately not tuned (tuning on the
held-out folds would be a double-dip). Multiclass: macro one-vs-rest AUROC, accuracy, balanced
accuracy (mean per-class recall) and macro-F1 in the per-fold table, with a per-class breakdown
(AUROC/precision/recall/F1) + a confusion matrix on the pooled OOF; class assignment is argmax
(no tuned threshold).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from dlux.config.cohort import Objective
from dlux.data.splits import parse_cv_split
from dlux.eval._common import (
    AggregateResult,
    _aggregate_field,
    _EndpointHandler,
    _fmt,
    load_fold_predictions,
)
from dlux.eval.binary import AGGREGATE_HANDLER as _BINARY_AGGREGATE_HANDLER
from dlux.eval.binary import BINARY_METRIC_LABELS
from dlux.eval.multiclass import AGGREGATE_HANDLER as _MULTICLASS_AGGREGATE_HANDLER
from dlux.eval.multiclass import _aggregate_field_multiclass
from dlux.eval.regression import AGGREGATE_HANDLER as _REGRESSION_AGGREGATE_HANDLER
from dlux.eval.regression_vector import AGGREGATE_HANDLER as _REGRESSION_VECTOR_AGGREGATE_HANDLER
from dlux.eval.regression_vector import _aggregate_field_expression
from dlux.eval.survival import AGGREGATE_HANDLER as _SURVIVAL_AGGREGATE_HANDLER
from dlux.eval.survival import _aggregate_field_survival

logger = logging.getLogger(__name__)

_SENTINEL = "metadata.json"
_PREDICTIONS = "test_predictions.npz"


def _read_meta(fold_dir: Path) -> dict:
    """A fold's metadata.json as a dict, or ``{}`` when it cannot be read."""
    try:
        return json.loads((fold_dir / _SENTINEL).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _grid_of(meta: dict) -> Optional[tuple[int, int]]:
    """Expected (n_outer, n_inner) stamped in a fold's metadata, or None if not recorded."""
    if "n_outer" in meta and "n_inner" in meta:
        return int(meta["n_outer"]), int(meta["n_inner"])
    return None


def _discover_folds(
    experiment_dir: Path,
) -> tuple[
    Dict[tuple[str, int], List[tuple[int, Path]]],
    Dict[str, tuple[int, int]],
    Dict[tuple[str, int], Optional[dict]],
]:
    """Scan for completed nested-CV folds -> (folds, grids, coverage).

    ``folds`` maps (field, outer) -> [(inner, npz)]. ``grids`` carries the expected CV shape stamped in
    metadata (for the strict completeness check). ``coverage`` carries each outer fold's declared-vs-scored
    TEST patients, read from the first inner replicate seen: the shared-TEST invariant makes the
    replicates of one outer fold score the same patients, so counting every replicate would multiply the
    study's N by ``n_inner``. ``None`` when a fold recorded no coverage."""
    folds: Dict[tuple[str, int], List[tuple[int, Path]]] = {}
    grids: Dict[str, tuple[int, int]] = {}
    coverage: Dict[tuple[str, int], Optional[dict]] = {}
    for d in sorted(p for p in experiment_dir.iterdir() if p.is_dir()):
        if not (d / _SENTINEL).exists():  # sentinel absent -> fold incomplete, skip
            continue
        parsed = parse_cv_split(d.name)
        if parsed is None:  # not a nested-CV fold (e.g. an all_test split)
            continue
        field, outer, inner = parsed
        npz = d / _PREDICTIONS
        if not npz.exists():
            continue
        folds.setdefault((field, outer), []).append((inner, npz))
        if field not in grids or (field, outer) not in coverage:
            meta = _read_meta(d)
            if field not in grids and _grid_of(meta) is not None:
                grids[field] = _grid_of(meta)  # type: ignore[assignment]
            coverage.setdefault((field, outer), meta.get("test_coverage"))
    return folds, grids, coverage


def _roll_up_coverage(per_outer: Dict[int, Optional[dict]]) -> Optional[dict]:
    """Union a field's per-outer TEST coverage into one study-level statement.

    None if any outer fold recorded no coverage: a partial union would understate the loss and read as
    full coverage. Outer folds partition the cohort, so declared and scored sum across them."""
    if not per_outer or any(c is None for c in per_outer.values()):
        return None
    declared = sum(int(c["declared"]) for c in per_outer.values())  # type: ignore[index]
    scored = sum(int(c["scored"]) for c in per_outer.values())  # type: ignore[index]
    lost = sorted({str(p) for c in per_outer.values() for p in c["lost"]})  # type: ignore[index]
    return {"declared": declared, "scored": scored, "lost": lost}


def _verify_completeness(
    field: str,
    outer_to_reps: Dict[int, List[tuple[int, Path]]],
    stamped: Optional[tuple[int, int]],
    expected: tuple[int, int],
    strict: bool,
) -> None:
    """Completeness gate against the grid the study declares.

    The expected shape comes from config, not from a fold we happened to find: a grid read out of the
    data cannot detect that the data is short. The grid stamped in fold metadata is still required, it
    is cross-checked against the declared one, which catches a sweep whose config changed midway."""
    if stamped is None:
        raise ValueError(
            f"[{field}] fold metadata is missing the CV grid (n_outer/n_inner). Re-run the sweep so its "
            f"metadata records the grid."
        )
    if stamped != expected:
        raise ValueError(
            f"[{field}] the folds were run on a {stamped[0]}x{stamped[1]} grid but the study declares "
            f"{expected[0]}x{expected[1]} — the sweep and the config disagree; re-run or fix the study."
        )
    n_outer, n_inner = expected
    expected = {(o, i) for o in range(n_outer) for i in range(n_inner)}
    present = {(o, i) for o, reps in outer_to_reps.items() for (i, _npz) in reps}
    missing = sorted(expected - present)
    if missing:
        msg = (
            f"[{field}] INCOMPLETE sweep: {len(missing)}/{len(expected)} folds missing vs grid {n_outer}×{n_inner} "
            f"(missing: {['o%d_i%d' % m for m in missing[:12]]}{'…' if len(missing) > 12 else ''})"
        )
        if strict:
            raise ValueError(msg + ". Re-run the missing folds, or pass strict=false to aggregate the partial sweep.")
        logger.warning(msg + ". Proceeding because strict=false.")


def aggregate_experiment(
    experiment_dir: str | Path,
    *,
    expected_fields: Sequence[str],
    n_outer: int,
    n_inner: int,
    strict: bool = True,
    random_experiment_dir: str | Path | None = None,
) -> List[AggregateResult]:
    """Aggregate the study's declared endpoints under ``experiment_dir`` (one result per field).

    ``expected_fields`` and the grid come from the study, not from what happens to be on disk. That is
    the difference that lets this detect an endpoint with zero completed folds: such an endpoint is
    simply absent from discovery, so a check driven by the data can never notice it is gone.

    ``strict`` (default): refuse to report anything short of the declared shape. ``strict=False``
    aggregates what exists, warning about the gaps. ``random_experiment_dir`` (expression only): a
    random-baseline sweep's dir, enables the conservative significant-gene count (trained vs random)."""
    experiment_dir = Path(experiment_dir)
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"experiment dir does not exist: {experiment_dir}")
    folds, grids, coverage = _discover_folds(experiment_dir)
    if not folds:
        raise FileNotFoundError(
            f"no completed nested-CV folds (with {_SENTINEL} + {_PREDICTIONS}) under {experiment_dir}"
        )

    discovered = {field for field, _ in folds}
    missing_endpoints = [f for f in expected_fields if f not in discovered]
    if missing_endpoints:
        msg = (
            f"declared endpoint(s) {missing_endpoints} have NO completed folds under {experiment_dir} "
            f"(study.targets = {list(expected_fields)}). Re-run them, or narrow study.targets."
        )
        if strict:
            raise ValueError(msg + " Pass strict=false to report the endpoints that did complete.")
        logger.warning(msg + " Proceeding because strict=false.")
    undeclared = sorted(discovered - set(expected_fields))
    if undeclared:
        logger.warning(
            "runs dir contains endpoint(s) %s that study.targets does not declare — not aggregated.", undeclared
        )
    random_folds: Dict[tuple[str, int], List[tuple[int, Path]]] = {}
    random_grids: Dict[str, tuple[int, int]] = {}
    if random_experiment_dir is not None:
        random_folds, random_grids, _ = _discover_folds(Path(random_experiment_dir))
        if not random_folds:
            logger.warning("random_experiment_dir %s has no completed folds; ignoring.", random_experiment_dir)
    expected_grid = (int(n_outer), int(n_inner))
    results: List[AggregateResult] = []
    # Declared set, sorted order: which endpoints are reported is the study's call, but their order in
    # the report is not a thing this step should change.
    for field in sorted(f for f in expected_fields if f in discovered):
        field_folds = {o: reps for (f, o), reps in folds.items() if f == field}
        _verify_completeness(field, field_folds, grids.get(field), expected_grid, strict)
        first_npz = sorted(next(iter(field_folds.values())))[0][1]  # peek endpoint type to dispatch
        endpoint_type = load_fold_predictions(first_npz)["endpoint_type"]
        # Core dispatch stays an explicit if-chain (not a registry lookup): the expression core alone
        # takes a `random_reps` arg (the SEQUOIA random baseline), so the core signatures aren't uniform.
        if endpoint_type == Objective.regression_vector:
            rand_reps = {o: reps for (f, o), reps in random_folds.items() if f == field} or None
            # The null feeds a reported statistic (the trained-vs-random gene count), so it is held to the
            # same declared grid as the trained arm. Checking only "are there any folds at all" let a
            # short null through silently, changing that count without saying so.
            if random_experiment_dir is not None:
                if rand_reps is None:
                    logger.warning(
                        "[%s] random_experiment_dir has no completed folds for this endpoint — reporting "
                        "without the trained-vs-random comparison.",
                        field,
                    )
                else:
                    _verify_completeness(
                        f"{field} (random baseline)", rand_reps, random_grids.get(field), expected_grid, strict
                    )
            res = _aggregate_field_expression(field, field_folds, random_reps=rand_reps)
        elif endpoint_type == Objective.multiclass:
            res = _aggregate_field_multiclass(field, field_folds)
        elif endpoint_type == Objective.survival:
            res = _aggregate_field_survival(field, field_folds)
        else:
            res = _aggregate_field(field, field_folds, _handler(endpoint_type))
        # Coverage comes from run metadata rather than from the predictions, so it is attached here and
        # the cores stay unaware of it, they only ever see patients that were scored.
        res.coverage = _roll_up_coverage({o: c for (f, o), c in coverage.items() if f == field})
        if res.coverage and res.coverage["lost"]:
            logger.warning(
                "[%s] %d of %d declared TEST patients were never scored: %s",
                field,
                len(res.coverage["lost"]),
                res.coverage["declared"],
                res.coverage["lost"],
            )
        results.append(res)
    return results


# -- reporting (csv + markdown + figures) ------------------------------------
def write_reports(results: List[AggregateResult], experiment_name: str, out_dir: str | Path) -> Path:
    """Write per-patient + per-fold CSVs, a summary.md (with a per-fold metric table), and figures."""
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _write_per_patient_csv(results, out_dir / "per_patient_ensemble.csv")
    _write_per_fold_csv(results, out_dir / "per_fold_metrics.csv")
    for res in results:
        _handler(res.endpoint_type).write_artifacts(res, out_dir, figures_dir)
    # Combined cross-field per-fold strips: group results by the handler's declared strip_group (so this
    # driver holds no per-objective membership lists), then plot each group with its metric + filename.
    strips: Dict[str, List[AggregateResult]] = {}
    for r in results:
        strips.setdefault(_handler(r.endpoint_type).strip_group, []).append(r)
    if strips.get("classification"):
        _plot_auroc_by_fold(strips["classification"], figures_dir / "auroc_by_fold.png")
    if strips.get("regression"):
        _plot_metric_by_fold(strips["regression"], "r2", "R²", figures_dir / "r2_by_fold.png")
    if strips.get("expression"):
        _plot_metric_by_fold(
            strips["expression"], "gene_pearson_mean", "mean gene Pearson", figures_dir / "gene_pearson_by_fold.png"
        )
    if strips.get("survival"):
        _plot_metric_by_fold(strips["survival"], "c_index", "C-index", figures_dir / "c_index_by_fold.png")
    # Written last so it can link every figure the handlers and strip plotters just produced.
    _write_summary_md(results, experiment_name, out_dir / "summary.md", figures_dir)
    return out_dir


def _write_per_patient_csv(results: List[AggregateResult], path: Path) -> None:
    import csv

    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "outer_fold", "patient_code", "ensemble_prediction", "label", "n_replicates"])
        for res in results:
            patient_row = _handler(res.endpoint_type).patient_row  # (pred, label) formatting per endpoint
            for p in res.patient_predictions:
                pred, label = patient_row(p)
                w.writerow([p.field, p.outer_fold, p.patient_code, pred, label, p.n_replicates])


def _write_per_fold_csv(results: List[AggregateResult], path: Path) -> None:
    import csv

    labels = _handler(results[0].endpoint_type).metric_labels if results else BINARY_METRIC_LABELS
    with Path(path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["field", "outer_fold", "n_patients", "n_positive", *labels])
        for res in results:
            keys = _handler(res.endpoint_type).metric_labels
            for m in res.per_outer:
                w.writerow([m.field, m.outer_fold, m.n_patients, m.n_positive, *(f"{m.metrics[k]:.6f}" for k in keys)])


def _figure_links(figures_dir: Path, field: str | None, claimed: set[Path]) -> List[str]:
    """Markdown image links for ``figures_dir``.

    With a field, returns that field's own figures (stem ending in ``_<field>``); with None, whatever
    is left over, the cross-field per-fold strips. Paths are relative to summary.md.
    """
    if not figures_dir.is_dir():
        return []
    picked = sorted(
        p for p in figures_dir.glob("*.png") if p not in claimed and (p.stem.endswith(f"_{field}") if field else True)
    )
    claimed.update(picked)
    return [f"![{p.stem}]({figures_dir.name}/{p.name})" for p in picked]


def _coverage_lines(res: AggregateResult) -> List[str]:
    """State the TEST coverage whenever it is not total, and say so when it is not known.

    Without this line a reduced N is silent: the table's own n is the number of patients that were
    scored, so an endpoint that lost patients to `min_tiles` or to a missing feature cache reports
    a smaller cohort with nothing marking the difference. Full coverage needs no line."""
    if res.coverage is None:
        return ["**Coverage unknown** — these runs recorded no declared-vs-scored coverage.", ""]
    declared, lost = res.coverage["declared"], res.coverage["lost"]
    if not lost:
        return []
    shown = ", ".join(f"`{p}`" for p in lost[:12]) + ("…" if len(lost) > 12 else "")
    return [
        f"**Coverage: {res.coverage['scored']}/{declared} declared TEST patients scored** "
        f"({len(lost)} never scored: {shown}). Every metric above is on the scored patients only.",
        "",
    ]


def _write_summary_md(
    results: List[AggregateResult], experiment_name: str, path: Path, figures_dir: Path | None = None
) -> None:
    claimed: set[Path] = set()
    lines = [f"# Aggregate — {experiment_name}", ""]
    for res in results:
        handler = _handler(res.endpoint_type)
        labels = handler.metric_labels
        cols = list(labels.values())
        has_pos = handler.has_positive_count  # only classification carries a positive count
        pos_col, pos_sep = (" pos |", "--:|") if has_pos else ("", "")
        rows = [
            f"| fold | n |{pos_col} " + " | ".join(cols) + " |",
            f"|:--|--:|{pos_sep}" + "|".join(["--:"] * len(cols)) + "|",
        ]
        for m in res.per_outer:
            pos = f" {m.n_positive} |" if has_pos else ""
            rows.append(
                f"| o{m.outer_fold} | {m.n_patients} |{pos} " + " | ".join(_fmt(m.metrics[k]) for k in labels) + " |"
            )
        mean_pos = "  |" if has_pos else ""
        rows.append(
            f"| **mean±std** |  |{mean_pos} "
            + " | ".join(f"{_fmt(res.mean[k])}±{_fmt(res.std[k])}" for k in labels)
            + " |"
        )
        pooled_pos = f" {res.pooled_positive} |" if has_pos else ""
        rows.append(
            f"| **pooled OOF** | {res.pooled_n} |{pooled_pos} " + " | ".join(_fmt(res.pooled[k]) for k in labels) + " |"
        )

        cover = ", ".join(f"o{o}={n}" for o, n in sorted(res.inner_coverage.items()))
        lines += [
            f"## `{res.field}` ({res.endpoint_type.value})",
            "",
            f"{res.n_outer_folds} outer folds | inner replicates/fold: {cover}",
            "",
            *rows,
            "",
            *_coverage_lines(res),
            *handler.summary_extra(res),
            handler.footer,
            "",
        ]
        if figures_dir is not None:
            field_figures = _figure_links(figures_dir, res.field, claimed)
            if field_figures:
                lines += [*field_figures, ""]
    if figures_dir is not None:
        shared = _figure_links(figures_dir, None, claimed)
        if shared:
            lines += ["## figures", "", *shared, ""]
    Path(path).write_text("\n".join(lines))


def _plot_auroc_by_fold(results: List[AggregateResult], path: Path) -> None:
    """Strip plot of per-outer-fold AUROCs, one column per field, with a mean line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    all_aurocs: List[float] = []
    for idx, res in enumerate(results):
        aurocs = np.asarray([m.metrics["auroc"] for m in res.per_outer if np.isfinite(m.metrics["auroc"])])
        if aurocs.size == 0:
            continue
        all_aurocs.extend(aurocs.tolist())
        jitter = np.linspace(-0.08, 0.08, aurocs.size)
        ax.scatter(np.full(aurocs.size, idx) + jitter, aurocs, s=40, alpha=0.85)
        ax.hlines(aurocs.mean(), idx - 0.22, idx + 0.22, colors="black", lw=1.5)
        ax.text(idx + 0.28, aurocs.mean(), f"{aurocs.mean():.3f} ± {aurocs.std(ddof=0):.3f}", va="center", fontsize=8)

    # Fixed, comparable y-axis [0.4, 1.0], extended downward only if a fold actually dips below 0.4,
    # so runs are visually comparable and tiny spreads aren't exaggerated, without ever clipping outliers.
    y_lo = min(0.4, min(all_aurocs) - 0.03) if all_aurocs else 0.4
    ax.axhline(0.5, color="grey", lw=0.8, linestyle="--", alpha=0.7)  # chance
    ax.text(len(results) - 0.5 + 0.55, 0.5, "chance", color="grey", fontsize=7, va="bottom", ha="right")

    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([r.field for r in results])
    ax.set(
        xlim=(-0.5, len(results) - 0.5 + 0.6), ylim=(max(0.0, y_lo), 1.01), ylabel="AUROC", title="Per-outer-fold AUROC"
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_metric_by_fold(results: List[AggregateResult], key: str, label: str, path: Path) -> None:
    """Strip of per-outer-fold values of one metric, one column per field (regression analog of the
    per-fold AUROC strip)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(4.5, 1.6 * len(results)), 4.0))
    all_vals: List[float] = []
    for idx, res in enumerate(results):
        vals = np.asarray([m.metrics[key] for m in res.per_outer if np.isfinite(m.metrics[key])])
        if vals.size == 0:
            continue
        all_vals.extend(vals.tolist())
        jitter = np.linspace(-0.08, 0.08, vals.size)
        ax.scatter(np.full(vals.size, idx) + jitter, vals, s=40, alpha=0.85)
        ax.hlines(vals.mean(), idx - 0.22, idx + 0.22, colors="black", lw=1.5)
        ax.text(idx + 0.28, vals.mean(), f"{vals.mean():.3f} ± {vals.std(ddof=0):.3f}", va="center", fontsize=8)

    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([r.field for r in results])
    lo = (min(all_vals) - 0.05) if all_vals else 0.0
    hi = (max(all_vals) + 0.05) if all_vals else 1.0
    ax.set(xlim=(-0.5, len(results) - 0.5 + 0.6), ylim=(lo, hi), ylabel=label, title=f"Per-outer-fold {label}")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -- endpoint handlers (dispatched on objective via _HANDLERS) ----------------
# A handler owns one endpoint kind's reporting: metric column labels, the summary footer, whether a
# per-fold positive count applies, the summary's extra callout lines, and the per-field artifacts
# (figure + any csv). The numeric aggregation cores (_aggregate_field / _aggregate_field_expression)
# stay separate: scalar and per-gene-vector pooling are different shapes.
_HANDLERS: Dict[str, _EndpointHandler] = {
    "binary": _BINARY_AGGREGATE_HANDLER,
    "multiclass": _MULTICLASS_AGGREGATE_HANDLER,
    "regression": _REGRESSION_AGGREGATE_HANDLER,
    "regression_vector": _REGRESSION_VECTOR_AGGREGATE_HANDLER,
    "survival": _SURVIVAL_AGGREGATE_HANDLER,
}


def _handler(endpoint_type: Objective) -> _EndpointHandler:
    try:
        return _HANDLERS[endpoint_type]  # str-enum keys the string-keyed registry transparently (Step 3 re-keys)
    except KeyError:
        raise NotImplementedError(f"aggregate reporting supports {sorted(_HANDLERS)} (got '{endpoint_type}').")


def endpoint_headline(res: AggregateResult) -> str:
    """One-line stdout summary for a result, dispatched on its endpoint type (used by bin/aggregate)."""
    return _handler(res.endpoint_type).headline(res)
