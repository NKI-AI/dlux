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
"""Stage-2 arm comparison against a measured noise floor, from persisted resplit rows.

Each arm was trained R times on seed-drawn splits, three replicates per seed (``rep`` 0/1/2). Per
seed, with no run shared between the two clouds:

    delta_real[arm] = m(arm, seed, 0) - m(reference, seed, 0)
    delta_null      = { m(a, seed, 1) - m(a, seed, 2)  for every arm a }

``delta_real`` carries the configuration effect plus training noise. ``delta_null`` carries training
noise alone, so the question "is the difference bigger than noise" is answered by comparing two
clouds rather than by assuming a distribution. See ``docs/specs/COMPARE_SPEC.md`` sections 6/6b.

The null pools every arm's within-arm pair. A null measured on the reference alone has variance
``2*var_ref``, while ``delta_real`` carries ``var_ref + var_arm``, equal only if the arms are equally
noisy, and too low (so too eager to declare an effect) whenever the arm under test is noisier. Pooling
gives the mean of the per-arm variances, which is exactly ``var_ref + var_arm``. No cross-arm null is
possible: any reference-vs-arm difference contains the effect by construction.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from dlux.data.errors import BuildDbError

# rep 0 is the arm's measurement; 1 and 2 are its within-arm null pair. Fixed rather than discovered
# from the files so a half-finished sweep is a loud error instead of a quietly different design.
REAL_REP = 0
NULL_REPS = (1, 2)
REQUIRED_REPS = (REAL_REP, *NULL_REPS)

# Margin percentile. The claim it licenses is "bigger than what training randomness alone produces 95%
# of the time", calibrated to the measured floor rather than picked.
MARGIN_PERCENTILE = 95.0


@dataclass
class Trial:
    """One persisted row: one arm, one seed, one replicate."""

    experiment: str
    field: str
    seed: int
    rep: int
    metric: float
    metric_slide: float
    n_declared: int
    n_test: int
    split_hash: str
    train_seed: int
    git_sha: str
    metric_key: str | None = None  # which metric `metric` is (None if unset)


@dataclass
class ArmResult:
    """One test arm's paired difference against the reference, over the seeds both scored."""

    arm: str
    seeds: List[int]
    delta_real: np.ndarray
    #: This arm's own within-arm differences. Contributed to the pooled null, and reported per arm so
    #: a much noisier arm is visible rather than averaged away.
    delta_own_null: np.ndarray


@dataclass
class ResplitComparison:
    """Every arm against one reference on one endpoint, plus the pooled null they share."""

    field: str
    reference: str
    metric_name: str
    n_seeds: int
    arms: List[ArmResult]
    delta_null: np.ndarray
    #: arm -> the within-arm differences that went into ``delta_null`` (reference included).
    null_by_arm: Dict[str, np.ndarray] = field(default_factory=dict)
    #: Per-arm absolute metric at rep 0, for the "what were the numbers" table.
    absolute: Dict[str, np.ndarray] = field(default_factory=dict)
    #: arm -> rep -> absolute metric per seed. The within-arm spread across reps IS the null, so
    #: showing it beside the arms' separation makes the whole comparison readable in one panel.
    absolute_by_rep: Dict[str, Dict[int, np.ndarray]] = field(default_factory=dict)
    #: seed -> (declared, scored) summed over arms; a systematic coverage loss must not hide.
    coverage: Dict[str, tuple] = field(default_factory=dict)
    #: arm -> seeds it had complete but that the intersection discarded. Never silent: an R smaller
    #: than the sweep looks like "we ran R" unless the report says what was dropped and why.
    dropped: Dict[str, int] = field(default_factory=dict)

    @property
    def margin(self) -> float:
        """delta = P95(|delta_null|): the measured noise floor."""
        return float(np.percentile(np.abs(self.delta_null), MARGIN_PERCENTILE))


def read_rows(rows_dir: Path) -> List[Trial]:
    """Every results row under ``rows_dir``.

    Metrics must be numeric: a string-valued metric is a stale or malformed writer, so it is an error
    rather than something to silently coerce.
    """
    rows_dir = Path(rows_dir)
    paths = sorted(rows_dir.glob("*.json"))
    if not paths:
        raise BuildDbError(
            f"no resplit rows under {rows_dir}. compare_resplits reads persisted trials only. Run the "
            f"sweep first (slurm/resplit_sweep.sh)."
        )
    trials = []
    for path in paths:
        row = json.loads(path.read_text())
        for key in ("metric", "metric_slide"):
            if isinstance(row.get(key), str):
                raise BuildDbError(
                    f"{path.name} stores '{key}' as a string ({row[key]!r}), but metrics must be numeric. "
                    f"Re-run the trial to regenerate the row."
                )
        trials.append(
            Trial(
                experiment=str(row["experiment"]),
                field=str(row["field"]),
                seed=int(row["seed"]),
                rep=int(row["rep"]),
                metric=float(row["metric"]),
                metric_slide=float(row["metric_slide"]),
                n_declared=int(row["n_declared"]),
                n_test=int(row["n_test"]),
                split_hash=str(row["split_hash"]),
                train_seed=int(row["train_seed"]),
                git_sha=str(row["git_sha"]),
                metric_key=(str(row["metric_key"]) if row.get("metric_key") else None),
            )
        )
    return trials


def check_invariants(trials: Sequence[Trial]) -> List[str]:
    """Everything that must hold before a number means anything. Returns human-readable notes.

    Raises on a violation rather than dropping the offending seed. A silently smaller R would look
    like "we ran R" when we did not.
    """
    if not trials:
        raise BuildDbError("no trials to check.")

    bad_metric = [t for t in trials if not np.isfinite(t.metric)]
    if bad_metric:
        raise BuildDbError(
            f"{len(bad_metric)} trial(s) have a non-finite metric, first: "
            f"{bad_metric[0].experiment} seed={bad_metric[0].seed} rep={bad_metric[0].rep}."
        )

    # One drawn split per seed, shared by every arm. This makes the comparison paired.
    hashes: Dict[int, set] = defaultdict(set)
    for t in trials:
        hashes[t.seed].add(t.split_hash)
    disagreeing = sorted(s for s, h in hashes.items() if len(h) > 1)
    if disagreeing:
        raise BuildDbError(
            f"seed(s) {disagreeing[:5]} drew more than one split. The arms did not train on the same "
            f"data, so nothing here is paired."
        )

    # A repeated training seed means two 'independent' runs are the same run.
    train_seeds = [t.train_seed for t in trials]
    if len(set(train_seeds)) != len(train_seeds):
        seen, dupes = set(), set()
        for t in trials:
            (dupes if t.train_seed in seen else seen).add(t.train_seed)
        raise BuildDbError(
            f"{len(dupes)} training seed(s) are reused across trials. Replicates that share a torch "
            f"seed are not replicates, and the null cloud collapses toward zero."
        )

    notes = []
    shas = {t.git_sha for t in trials}
    if len(shas) > 1:
        counts = defaultdict(int)
        for t in trials:
            counts[t.git_sha[:8]] += 1
        notes.append(
            "Trials span more than one commit: "
            + ", ".join(f"`{sha}` x{n}" for sha, n in sorted(counts.items(), key=lambda kv: -kv[1]))
            + ". Check the difference is inert before reading the deltas."
        )
    lost = sum(t.n_declared - t.n_test for t in trials)
    if lost:
        notes.append(
            f"{lost} patient-slots were declared by the draw but not scored, across {len(trials)} "
            f"trials (mean {lost / len(trials):.2f}/trial). These are patients whose slides all fell "
            f"below min_tiles. Present in every arm's split equally."
        )
    return notes


def compare_resplits(
    trials: Sequence[Trial],
    *,
    arms: Sequence[str],
    reference: str,
    field_name: str,
    metric_name: str = "AUROC",
) -> ResplitComparison:
    """Pair every arm against ``reference`` within seed, and pool the arms' within-arm nulls."""
    if reference not in arms:
        raise BuildDbError(f"reference '{reference}' is not among arms {list(arms)}.")
    others = [a for a in arms if a != reference]
    if not others:
        raise BuildDbError("a comparison needs at least one arm besides the reference.")

    by_key = {(t.experiment, t.seed, t.rep): t for t in trials if t.field == field_name}
    if not by_key:
        raise BuildDbError(f"no trials for endpoint '{field_name}'.")

    present = {a: sorted({s for (e, s, _r) in by_key if e == a}) for a in arms}
    missing_arms = [a for a in arms if not present[a]]
    if missing_arms:
        raise BuildDbError(f"no trials for arm(s) {missing_arms} on '{field_name}'.")

    # Seeds usable by every arm at every required rep. Computed once so all arms are compared on one
    # seed set, an arm evaluated on a different subset is not comparable to the others. The null is
    # restricted too, even though a marginal distribution needs no pairing: pooling unequal counts
    # would weight the better-covered arm more heavily, and the pooled variance only equals
    # var_ref + var_arm when every arm contributes the same number of pairs.
    complete = {a: {s for s in present[a] if all((a, s, r) in by_key for r in REQUIRED_REPS)} for a in arms}
    seeds = sorted(set.intersection(*complete.values()))
    if not seeds:
        short = {a: sorted({r for (e, _s, r) in by_key if e == a}) for a in arms}
        raise BuildDbError(
            f"no seed has all of reps {list(REQUIRED_REPS)} for every arm on '{field_name}'. "
            f"Reps present per arm: {short}. Every arm needs REPS=3. Rep 0 is its measurement, "
            f"reps 1/2 are its own null pair."
        )

    def metric(arm: str, seed: int, rep: int) -> float:
        return by_key[(arm, seed, rep)].metric

    null_by_arm = {
        arm: np.array([metric(arm, s, NULL_REPS[0]) - metric(arm, s, NULL_REPS[1]) for s in seeds]) for arm in arms
    }
    results = [
        ArmResult(
            arm=arm,
            seeds=seeds,
            delta_real=np.array([metric(arm, s, REAL_REP) - metric(reference, s, REAL_REP) for s in seeds]),
            delta_own_null=null_by_arm[arm],
        )
        for arm in others
    ]
    coverage = {
        arm: (
            sum(by_key[(arm, s, REAL_REP)].n_declared for s in seeds),
            sum(by_key[(arm, s, REAL_REP)].n_test for s in seeds),
        )
        for arm in arms
    }
    return ResplitComparison(
        field=field_name,
        reference=reference,
        metric_name=metric_name,
        n_seeds=len(seeds),
        arms=results,
        delta_null=np.concatenate([null_by_arm[a] for a in arms]),
        null_by_arm=null_by_arm,
        absolute={a: np.array([metric(a, s, REAL_REP) for s in seeds]) for a in arms},
        absolute_by_rep={a: {r: np.array([metric(a, s, r) for s in seeds]) for r in REQUIRED_REPS} for a in arms},
        coverage=coverage,
        dropped={a: len(complete[a]) - len(seeds) for a in arms},
    )


def margin_convergence(comparison: ResplitComparison, steps: int = 5) -> List[tuple]:
    """delta recomputed on the first k seeds, for k spanning the sweep -> ``[(n_seeds, delta), ...]``.

    A single delta invites the reader to trust a number that may still be moving, when it can keep
    drifting well past small R, visible only as a curve. Uses the same seed order the arms were paired
    in, so the k-th point is the answer the sweep would have given had it stopped at k seeds.
    """
    per_arm = len(comparison.delta_null) // max(len(comparison.null_by_arm), 1)
    out = []
    seen = set()
    for raw in np.linspace(per_arm / steps, per_arm, steps):
        # Deduped: below ~R=16 the linspace points round onto the same seed count, and a table
        # repeating one seed-count row looks as if the margin had stopped moving.
        k = int(round(raw))
        if k < 2 or k in seen:
            continue
        seen.add(k)
        pooled = np.concatenate([v[:k] for v in comparison.null_by_arm.values()])
        out.append((k, float(np.percentile(np.abs(pooled), MARGIN_PERCENTILE))))
    return out


def summarize_arm(result: ArmResult, margin: float, delta_null: np.ndarray) -> Dict[str, float]:
    """The reportable statistics for one arm. No p-value against zero, see COMPARE_SPEC section 6."""
    real = result.delta_real
    return {
        "median": float(np.median(real)),
        "mean": float(np.mean(real)),
        "sd": float(np.std(real, ddof=1)) if len(real) > 1 else float("nan"),
        # SEM of the paired per-split delta across seeds, shrinks with sqrt(R), unlike the replicate
        # margin. The number to compare the mean against for a small systematic effect.
        "sem": float(np.std(real, ddof=1) / np.sqrt(len(real))) if len(real) > 1 else float("nan"),
        "q25": float(np.percentile(real, 25)),
        "q75": float(np.percentile(real, 75)),
        "frac_above": float(np.mean(real > margin)),
        "frac_below": float(np.mean(real < -margin)),
        # Where the arm's median difference falls inside the noise cloud. 0.5 = indistinguishable.
        "percentile_in_null": float(np.mean(delta_null < np.median(real))),
        # 1.0 = the real cloud is exactly as wide as pure training noise.
        "sd_ratio": float(np.std(real, ddof=1) / np.std(delta_null, ddof=1)) if len(real) > 1 else float("nan"),
    }


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _jitter(n: int, spread: float = 0.13) -> np.ndarray:
    """Deterministic vertical offsets for a strip of points.

    A fixed sequence rather than an RNG draw so re-running the report on the same rows redraws the
    same figure, a plot that moves between runs is one nobody can diff.
    """
    if n <= 1:
        return np.zeros(n)
    # Golden-ratio stepping scatters neighbours in the sequence far apart, so consecutive seeds do
    # not fall in a visible diagonal the way linspace would.
    return ((np.arange(n) * 0.6180339887 % 1.0) - 0.5) * 2 * spread


def _violin(axis, values: np.ndarray, position: float, color: str, width: float = 0.7, vert: bool = False) -> None:
    """Density behind a strip, along whichever axis the strip runs.

    No-op below 5 points or on a constant series: a KDE over 2 values draws a shape with no meaning,
    and this figure is read at every R from the first trial onward.
    """
    if len(values) < 5 or np.allclose(values, values[0]):
        return
    parts = axis.violinplot(
        [values], positions=[position], vert=vert, widths=width, showextrema=False, showmedians=False
    )
    for body in parts["bodies"]:
        body.set_facecolor(color)
        body.set_alpha(0.30)
        body.set_zorder(1)


def _clouds_figure(comparison: ResplitComparison, path: Path) -> None:
    """Both clouds on one axis.

    Density behind, every point in front: at R=100 the null carries 2R values and a plain strip
    overplots into a solid bar, hiding exactly the shape the percentiles are read from. The raw points
    stay because a violin alone smooths away the tails, and the tails are where the margin lives.

    The null is drawn under every arm rather than once at the bottom: the question is always "is this
    arm's cloud outside the noise", and a single shared row invites reading the arms against each
    other, which the correlated delta_real clouds do not support.
    """
    plt = _pyplot()
    margin = comparison.margin
    rows = len(comparison.arms)
    fig, axes = plt.subplots(rows, 1, figsize=(7.8, 2.9 * rows), squeeze=False)
    for axis, result in zip(axes[:, 0], comparison.arms):
        axis.axvspan(-margin, margin, color="#d9dde3", alpha=0.5, zorder=0)
        for values, pos, color, label in (
            (comparison.delta_null, 0.0, "#7f8c9b", f"delta_null (n={len(comparison.delta_null)})"),
            (result.delta_real, 1.0, "#4c72b0", f"delta_real (n={len(result.delta_real)})"),
        ):
            _violin(axis, values, pos, color)
            axis.scatter(
                values,
                pos + _jitter(len(values)),
                s=16,
                alpha=0.45,
                color=color,
                edgecolors="none",
                zorder=3,
                label=label,
            )
            axis.scatter([np.median(values)], [pos], s=170, marker="|", color="#c44e52", zorder=4)
        axis.axvline(0.0, color="#9ca3af", lw=1.2, zorder=2)
        axis.set(
            yticks=[0, 1],
            yticklabels=["null\n(training noise)", result.arm],
            ylim=(-0.7, 1.7),
            xlabel=f"delta {comparison.metric_name} vs {comparison.reference}",
            title=f"{result.arm} vs {comparison.reference}: {comparison.field} "
            f"(R={comparison.n_seeds}, margin +-{margin:.4f})",
        )
        axis.grid(True, axis="x", lw=0.3, alpha=0.4)
        axis.legend(loc="upper right", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _metric_figure(comparison: ResplitComparison, chance: float | None, path: Path) -> None:
    """Each arm's absolute metric per split, one column per arm, replicates side by side.

    The delta figure places the arms relative to the reference. This places them absolutely, so a
    difference can be read against how good the arms are in the first place, +0.03 means something
    different at 0.55 than at 0.85. Same purpose as `compare`'s metric.png, so both stages read alike.

    Replicates are drawn as separate columns because the spread across them, at fixed arm and fixed
    splits, is the null itself. Seeing it beside the gap between arms is the whole comparison in one
    panel, with no arithmetic.
    """
    plt = _pyplot()
    # Reference first (leftmost), so the arms are read against it. The rest sorted for a stable order.
    ref = comparison.reference
    arms = ([ref] if ref in comparison.absolute_by_rep else []) + sorted(
        a for a in comparison.absolute_by_rep if a != ref
    )
    palette = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
    fig, axis = plt.subplots(figsize=(1.6 + 1.5 * len(arms) * len(REQUIRED_REPS), 4.4))

    # Medians throughout, never means: the summary table quotes the median of this same quantity, and
    # every other statistic in the report is a percentile (median delta, P95 margin). One centre
    # statistic per report, so letting it vary by panel is not cosmetic, the mean-median gap can
    # exceed the effects under discussion.
    ticks, labels, seen = [], [], []
    for arm_idx, arm in enumerate(arms):
        color = palette[arm_idx % len(palette)]
        positions = []
        for rep_idx, rep in enumerate(REQUIRED_REPS):
            values = comparison.absolute_by_rep[arm][rep]
            pos = arm_idx * (len(REQUIRED_REPS) + 0.9) + rep_idx
            positions.append(pos)
            seen.extend(values.tolist())
            _violin(axis, values, pos, color, vert=True)
            axis.scatter(pos + _jitter(len(values)), values, s=16, alpha=0.45, color=color, edgecolors="none")
            axis.hlines(np.median(values), pos - 0.3, pos + 0.3, colors="black", lw=1.3, zorder=5)
            ticks.append(pos)
            labels.append(f"rep{rep}")
        # The arm's median over every replicate: the number the summary table reports, spanning the
        # arm's columns so it belongs to the arm, not one replicate.
        pooled = np.concatenate([comparison.absolute_by_rep[arm][r] for r in REQUIRED_REPS])
        axis.hlines(
            np.median(pooled), positions[0] - 0.45, positions[-1] + 0.45, colors=color, lw=2.6, zorder=6, alpha=0.95
        )
        axis.text(
            (positions[0] + positions[-1]) / 2,
            1.005,
            f"{arm}\nmedian {np.median(pooled):.3f}",
            ha="center",
            va="top",
            fontsize=8,
            color=color,
            fontweight="bold",
        )
    if chance is not None:
        axis.axhline(chance, color="grey", lw=0.8, ls="--", alpha=0.7)
        # Left of the first column: at the right edge it collides with the axis spine.
        axis.text(ticks[0] - 0.55, chance, "chance", color="grey", fontsize=7, va="bottom", ha="left")
    floor = 0.0 if chance is None else chance
    lo = min(floor, min(seen)) if seen else floor
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels, fontsize=7)
    axis.set(
        ylabel=comparison.metric_name,
        ylim=(lo - 0.02, 1.04),  # from the metric's chance floor up to 1.04 (headroom for the per-arm labels)
        title=f"{comparison.field}: per-split {comparison.metric_name} (R={comparison.n_seeds}, bars are medians)",
    )
    axis.grid(True, axis="y", lw=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _convergence_figure(comparison: ResplitComparison, path: Path) -> None:
    """The margin as seeds accumulate. Whether R was enough is not visible in a single number."""
    plt = _pyplot()
    curve = margin_convergence(comparison, steps=8)
    fig, axis = plt.subplots(figsize=(6.2, 3.0))
    axis.plot([k for k, _ in curve], [d for _, d in curve], marker="o", color="#4c72b0")
    axis.axhline(comparison.margin, color="#c44e52", lw=1.0, ls="--", label=f"final {comparison.margin:.4f}")
    axis.set(
        xlabel="seeds used",
        ylabel=f"margin  P{MARGIN_PERCENTILE:.0f}|delta_null|",
        title=f"{comparison.field}: margin convergence",
        ylim=(0, None),
    )
    axis.grid(True, lw=0.3, alpha=0.4)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_resplit_comparison(
    comparison: ResplitComparison,
    name: str,
    out_dir: Path,
    notes: Sequence[str] = (),
    chance: float | None = 0.5,
) -> Path:
    """Write ``summary.md`` + ``clouds.png`` + ``metric.png`` + ``margin_convergence.png`` + CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _clouds_figure(comparison, out_dir / "clouds.png")
    _metric_figure(comparison, chance, out_dir / "metric.png")
    _convergence_figure(comparison, out_dir / "margin_convergence.png")

    with (out_dir / "per_seed.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "arm", "seed", "delta_real", "delta_own_null"])
        for result in comparison.arms:
            for i, seed in enumerate(result.seeds):
                writer.writerow(
                    [
                        comparison.field,
                        result.arm,
                        seed,
                        f"{result.delta_real[i]:.6f}",
                        f"{result.delta_own_null[i]:.6f}",
                    ]
                )
        for seed_i, seed in enumerate(comparison.arms[0].seeds):
            writer.writerow(
                [
                    comparison.field,
                    comparison.reference,
                    seed,
                    "",
                    f"{comparison.null_by_arm[comparison.reference][seed_i]:.6f}",
                ]
            )

    margin = comparison.margin
    metric = comparison.metric_name
    lines = [
        f"# resplit comparison: {name}",
        "",
        f"Endpoint `{comparison.field}`. Reference arm `{comparison.reference}`. "
        f"R = {comparison.n_seeds} independent splits. Metric {metric} (per patient).",
        "",
        f"Margin delta = P{MARGIN_PERCENTILE:.0f}(|delta_null|) = **{margin:.4f}**, pooled over "
        f"{len(comparison.null_by_arm)} arms and {len(comparison.delta_null)} within-arm replicate pairs.",
        "",
    ]
    # Stated before any statistic: an R smaller than the sweep looks like "we ran R" unless the report
    # says otherwise, and every number below is a percentile of a cloud whose size is the point.
    short = {a: n for a, n in comparison.dropped.items() if n}
    if short:
        worst = max(short, key=lambda a: short[a])
        lines += [
            f"> **Incomplete: {comparison.n_seeds} of {comparison.n_seeds + short[worst]} seeds.** "
            f"Arms are compared only on seeds where every arm has reps {list(REQUIRED_REPS)}. Dropped: "
            + ", ".join(f"{a} {n}" for a, n in sorted(short.items()))
            + ".",
            "",
        ]
    for note in notes:
        lines += [f"> {note}", ""]

    lines += [
        "## Arms",
        "",
        "| arm | median delta | mean±SEM | IQR | sd | > +delta | < -delta | pctile in null | sd ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in comparison.arms:
        s = summarize_arm(result, margin, comparison.delta_null)
        lines.append(
            f"| {result.arm} | {s['median']:+.4f} | {s['mean']:+.4f}±{s['sem']:.4f} | "
            f"[{s['q25']:+.4f}, {s['q75']:+.4f}] | {s['sd']:.4f} | "
            f"{s['frac_above']:.0%} | {s['frac_below']:.0%} | {s['percentile_in_null']:.2f} | "
            f"{s['sd_ratio']:.2f} |"
        )
    lines += [
        "",
        "## Per-arm noise",
        "",
        "| arm | own null sd | absolute " + metric + " (median) |",
        "| --- | --- | --- |",
    ]
    for arm in sorted(comparison.null_by_arm):
        own = comparison.null_by_arm[arm]
        lines.append(f"| {arm} | {np.std(own, ddof=1):.4f} | {np.median(comparison.absolute[arm]):.4f} |")
    lines += [
        "",
        "![clouds](clouds.png)",
        "",
        "![absolute metric](metric.png)",
        "",
        "![margin convergence](margin_convergence.png)",
        "",
        "## Definitions",
        "",
        f"- `delta` is m(arm) - m(`{comparison.reference}`) on the same split, replicate 0.",
        "- `delta_null` is replicate 1 minus replicate 2 within one arm, on the same split.",
        "- `mean±SEM` is the mean paired delta and its cross-seed SEM (sd/√R). The SEM shrinks with R, "
        "unlike the margin, so it resolves a small systematic effect the margin cannot — read it when the "
        "effect is smaller than the replicate noise.",
        "- `pctile in null` locates the arm's median delta in the null distribution. 0.5 is indistinguishable.",
        "- `sd ratio` is sd(delta) over sd(delta_null).",
        "- No p-value against zero is reported.",
        "",
    ]

    summary = out_dir / "summary.md"
    summary.write_text("\n".join(lines))
    return summary
