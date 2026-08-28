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
"""Qualitative comparison of experiment arms, from persisted aggregate output only.

Reads each arm's ``per_patient_ensemble.csv`` and reports, per endpoint: every arm's per-fold and
pooled metric, and the paired per-fold difference against a reference arm. Nothing is loaded, re-run
or re-fitted. If an arm has not been aggregated, it does not appear here.

**It computes no test, interval or p-value, deliberately.** The few outer folds share training data, so
their differences are not independent and no honest inference rests on that few. This answers "is
there maybe something here that justifies a more expensive experiment, and nothing more. See
``docs/specs/COMPARE_SPEC.md`` for the staged design this is stage 1 of.

Paired against a reference arm rather than all-vs-all: N arms give N(N-1)/2 pairs, which is quadratic
to read and, once testing ever arrives, a far worse multiplicity problem than N-1.

The predictions are each arm's mean over its inner replicates, so a comparison is between ensembled
procedures, not individual models, arm-level training noise is already partly averaged out.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from dlux.data.errors import BuildDbError
from dlux.eval._common import _roll_slides_to_patients, _roll_slides_to_patients_vec
from dlux.eval.binary import binary_metrics
from dlux.eval.gene_metrics import per_gene_pearson
from dlux.eval.multiclass import multiclass_metrics
from dlux.eval.regression import regression_metrics
from dlux.eval.survival_metrics import concordance_index

_TABLE = "per_patient_ensemble.csv"


@dataclass(frozen=True)
class ArmScores:
    """One arm's metric on one endpoint: per outer fold, and pooled over all of them."""

    arm: str
    per_fold: Dict[int, float]
    pooled: float


@dataclass(frozen=True)
class FieldComparison:
    """Every arm on one endpoint, plus each non-reference arm's paired deltas against the reference."""

    field: str
    reference: str
    n_patients: int  # patients all arms scored, the set every number here is computed on
    scores: List[ArmScores]
    # arm -> {outer_fold: metric(arm) - metric(reference)}; the reference itself is absent.
    deltas: Dict[str, Dict[int, float]]
    pooled_deltas: Dict[str, float]
    # arm -> how many patients that arm scored before intersection. Equal to n_patients for every arm
    # when they cover the same set; where they differ, the gap is the result rather than bookkeeping.
    coverage: Dict[str, int]

    @property
    def coverage_differs(self) -> bool:
        return len(set(self.coverage.values())) > 1


def binary_auroc(labels: np.ndarray, predictions: np.ndarray) -> float:
    """AUROC via the same call the binary aggregate uses. NaN for a single-class subset (undefined, not zero)."""
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(labels, predictions))


def multiclass_macro_auroc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Macro one-vs-rest AUROC over the (N, K) softmax probabilities. Class count comes from the
    array's width, so a fold missing a class still scores against the full set."""
    return float(multiclass_metrics(labels, probabilities, num_classes=probabilities.shape[1])["auroc"])


def regression_r2(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Coefficient of determination. 0 is the mean-predictor, and it goes negative for worse."""
    return float(regression_metrics(targets, predictions)["r2"])


def survival_c_index(target: np.ndarray, risk: np.ndarray) -> float:
    """Harrell's C. ``target`` is the (N, 2) [time, event] pair, since concordance needs the censoring
    indicator and cannot be computed from time alone."""
    return float(concordance_index(time=target[:, 0], risk=risk, event=target[:, 1]))


def expression_mean_gene_pearson(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Mean per-gene Pearson over the (N, G) patient x gene matrices, the same headline
    ``gene_pearson_mean`` the expression aggregate reports, so compare and aggregate agree. Correlation
    is computed per gene across patients (constant genes are undefined and dropped), then averaged.
    NaN when no gene is scorable (n < 2 or every gene constant)."""
    r = per_gene_pearson(predictions, labels)
    return float(np.nanmean(r)) if np.isfinite(r).any() else float("nan")


@dataclass(frozen=True)
class MetricSpec:
    """One selectable comparison metric: how to score it, its no-information level, its display label."""

    key: str
    fn: Callable[[np.ndarray, np.ndarray], float]
    chance: float | None  # drawn on the absolute plot so an arm reads against it; None omits the line
    label: str


def _binary(key: str) -> Callable[[np.ndarray, np.ndarray], float]:
    return lambda y, x: binary_metrics(y, x)[key]


def _multiclass(key: str) -> Callable[[np.ndarray, np.ndarray], float]:
    return lambda y, x: multiclass_metrics(y, x, num_classes=x.shape[1])[key]


def _regression(key: str) -> Callable[[np.ndarray, np.ndarray], float]:
    return lambda y, x: regression_metrics(y, x)[key]


# objective -> selectable metrics; the first entry is the endpoint default. Every metric delegates to the
# same function the aggregate reports with, so compare and aggregate cannot disagree on a number. Only
# higher-is-better metrics are offered, so the paired delta (arm - reference) reads "positive = arm better"
# for every choice, a comparison config picks one with `metric:` (e.g. metric=qwk for an ordinal endpoint).
_METRICS: Dict[str, Dict[str, MetricSpec]] = {
    "binary": {
        "auroc": MetricSpec("auroc", binary_auroc, 0.5, "AUROC"),
        "ap": MetricSpec("ap", _binary("ap"), None, "AP"),
        "accuracy": MetricSpec("accuracy", _binary("accuracy"), None, "accuracy"),
        "balanced_accuracy": MetricSpec("balanced_accuracy", _binary("balanced_accuracy"), 0.5, "balanced accuracy"),
        "f1": MetricSpec("f1", _binary("f1"), None, "F1"),
        "mcc": MetricSpec("mcc", _binary("mcc"), 0.0, "MCC"),
        "kappa": MetricSpec("kappa", _binary("kappa"), 0.0, "kappa"),
    },
    "multiclass": {
        "auroc": MetricSpec("auroc", multiclass_macro_auroc, 0.5, "macro AUROC"),
        "accuracy": MetricSpec("accuracy", _multiclass("accuracy"), None, "accuracy"),
        "balanced_accuracy": MetricSpec(
            "balanced_accuracy", _multiclass("balanced_accuracy"), None, "balanced accuracy"
        ),
        "macro_f1": MetricSpec("macro_f1", _multiclass("macro_f1"), None, "macro F1"),
        "qwk": MetricSpec("qwk", _multiclass("qwk"), 0.0, "QWK"),
    },
    "regression": {
        "r2": MetricSpec("r2", regression_r2, 0.0, "R²"),
        "pearson": MetricSpec("pearson", _regression("pearson"), 0.0, "Pearson r"),
        "spearman": MetricSpec("spearman", _regression("spearman"), 0.0, "Spearman ρ"),
    },
    "survival": {"c_index": MetricSpec("c_index", survival_c_index, 0.5, "C-index")},
    "regression_vector": {
        "gene_pearson_mean": MetricSpec("gene_pearson_mean", expression_mean_gene_pearson, 0.0, "mean gene Pearson")
    },
}


def default_metric_key(objective: str) -> str:
    """The endpoint's standard comparison metric, the first registered for the objective."""
    return next(iter(_METRICS[objective]))


def resolve_metric(objective: str, metric_key: str | None = None) -> MetricSpec:
    """The MetricSpec a comparison scores on. ``metric_key`` None -> the endpoint default. An unknown key
    or objective raises with the available choices listed."""
    if objective not in _METRICS:
        raise BuildDbError(f"compare supports objectives {sorted(_METRICS)}, got '{objective}'.")
    table = _METRICS[objective]
    key = metric_key or default_metric_key(objective)
    if key not in table:
        raise BuildDbError(
            f"metric '{key}' is not available for a '{objective}' comparison; choose from {sorted(table)}."
        )
    return table[key]


def score_predictions(
    predictions: Dict[str, Any], objective: str, metric_key: str | None = None
) -> tuple[float, int, int]:
    """Score one model's raw prediction dict -> ``(metric, n_patients, n_positive)``.

    ``metric_key`` selects which metric (None = the endpoint default). The resplit sweep sets it once so
    every trial is scored on the same metric compare_resplits then reports.

    Slides are rolled to patients by the mean, the same unit every other report uses, a metric over
    slides would weight multi-slide patients more heavily. Single model, so there is no inner
    ensemble to average over; that is the only difference from the aggregate path.

    Resolves through the same ``resolve_metric`` the arm comparison uses, so a resplit trial and a
    stage-1 comparison cannot disagree about what a number means.
    """
    codes, labels = np.asarray(predictions["patient_codes"]), np.asarray(predictions["labels"])
    values = np.asarray(predictions["probs"])
    if objective in ("multiclass", "survival", "regression_vector"):
        # multiclass: (K,) softmax per slide. survival: label is the (time, event) pair. expression:
        # prediction and label are both the (G,) gene vector. All three carry a vector the flat roller
        # cannot, so they ride the vec roller; only survival collapses its prediction back to a scalar.
        rolled, per_patient_label = _roll_slides_to_patients_vec(codes, values.reshape(len(codes), -1), labels)
        patients = sorted(rolled)
        y = np.stack([np.atleast_1d(per_patient_label[p]) for p in patients])
        x = np.stack([rolled[p] for p in patients])
        if objective == "survival":
            x = x.reshape(-1)
    else:
        rolled, per_patient_label = _roll_slides_to_patients(codes, values, labels)
        patients = sorted(rolled)
        y = np.asarray([per_patient_label[p] for p in patients])
        x = np.asarray([rolled[p] for p in patients])
    n_pos = int((y == 1).sum()) if objective == "binary" else int(y[:, 1].sum()) if objective == "survival" else 0
    return float(resolve_metric(objective, metric_key).fn(y, x)), len(patients), n_pos


# The logged/recorded key for each objective's headline metric. Mirrors the streamed names the tasks
# build (`test/auroc`, `test/r2`, `test/c_index`) so the `_patient` variant matches its counterpart.
METRIC_KEYS: Dict[str, str] = {
    "binary": "auroc",
    "multiclass": "auroc",  # macro one-vs-rest
    "regression": "r2",
    "survival": "c_index",
    "regression_vector": "gene_pearson_mean",  # the per_fold_metrics / aggregate column name
}

# Objectives whose per_patient_ensemble.csv cannot support their metric, and the richer per-endpoint
# table to read instead. Multiclass ensembles to an argmax class there, losing the probabilities an
# AUROC needs; survival records time as the label and drops the event indicator a C-index needs.
_PREDICTIONS_TABLE = {"multiclass", "survival"}


def _read_arm_expression(arm_dir: Path, arm: str) -> Dict[str, Dict[str, tuple]]:
    """``field -> patient_code -> (outer_fold, pred_vector, label_vector)`` from the per-patient NPZ.

    Expression's ensemble CSV is header-only (no per-patient scalar), so the field is discovered from
    the ``per_patient_<field>.npz`` files instead, and the ``(N, G)`` gene matrices are read straight
    back. float16 predictions are widened to float64 for the correlation."""
    out: Dict[str, Dict[str, tuple]] = {}
    for npz_path in sorted(arm_dir.glob("per_patient_*.npz")):
        field = npz_path.stem[len("per_patient_") :]
        data = np.load(npz_path, allow_pickle=True)
        codes, folds = data["patient_codes"], data["outer_folds"]
        preds, labels = data["preds"].astype(np.float64), data["labels"].astype(np.float64)
        out[field] = {str(codes[i]): (int(folds[i]), preds[i], labels[i]) for i in range(len(codes))}
    if not out:
        raise BuildDbError(
            f"arm '{arm}': no per_patient_*.npz under {arm_dir}, which an expression comparison needs. "
            f"Re-aggregate the arm."
        )
    return out


def read_arm(results_dir: Path, arm: str, cohort: str, objective: str = "binary") -> Dict[str, Dict[str, tuple]]:
    """``field -> patient_code -> (outer_fold, prediction, target)`` from one arm's aggregate output.

    ``prediction`` and ``target`` carry whatever the objective's metric consumes: scalars for binary
    and regression, an (N, K) probability row and a class index for multiclass, a risk scalar and a
    (time, event) pair for survival, and the (G,) gene vector for expression."""
    arm_dir = Path(results_dir) / arm / cohort
    if objective == "regression_vector":
        return _read_arm_expression(arm_dir, arm)
    table = arm_dir / _TABLE
    if not table.is_file():
        raise BuildDbError(
            f"no aggregate output for arm '{arm}': {table} is missing. compare reads persisted "
            f"results only, so the arm has to be aggregated first."
        )
    # The ensemble table is the index of which endpoints this arm scored, even where it cannot supply
    # the values the metric needs.
    with table.open() as handle:
        rows = list(csv.DictReader(handle))
    fields = sorted({row["field"] for row in rows})

    if objective not in _PREDICTIONS_TABLE:
        out: Dict[str, Dict[str, tuple]] = {}
        for row in rows:
            out.setdefault(row["field"], {})[row["patient_code"]] = (
                int(row["outer_fold"]),
                float(row["ensemble_prediction"]),
                float(row["label"]),
            )
        return out
    return {field: _read_predictions(arm_dir, arm, field, objective) for field in fields}


def _read_predictions(arm_dir: Path, arm: str, field: str, objective: str) -> Dict[str, tuple]:
    """One endpoint's per-patient rows from ``predictions_<field>.csv``."""
    table = arm_dir / f"predictions_{field}.csv"
    if not table.is_file():
        raise BuildDbError(
            f"arm '{arm}' has no {table.name}, which a '{objective}' comparison needs (the ensemble "
            f"table cannot supply it). Re-aggregate the arm."
        )
    out: Dict[str, tuple] = {}
    with table.open() as handle:
        for row in csv.DictReader(handle):
            if objective == "multiclass":
                # Sorted on the numeric suffix, not the string: prob_10 must not sort before prob_2.
                keys = sorted((k for k in row if k.startswith("prob_")), key=lambda k: int(k[5:]))
                probs = tuple(float(row[k]) for k in keys)
                out[row["patient_code"]] = (int(row["outer_fold"]), probs, float(row["label"]))
            else:  # survival
                out[row["patient_code"]] = (
                    int(row["outer_fold"]),
                    float(row["risk"]),
                    (float(row["time"]), float(row["event"])),
                )
    return out


def _assert_comparable(field: str, reference: str, ref: Dict[str, tuple], arm: str, other: Dict[str, tuple]) -> None:
    """Refuse arms whose shared patients carry different labels.

    Differing patient *sets* are not refused: an arm can legitimately score patients another cannot,
    e.g. a finer tiling grid clears ``min_tiles`` for slides a coarser one leaves with no tiles.
    Those are intersected by the caller and the gap reported as coverage. A shared patient with two
    different labels is a different thing entirely, one arm is reading another endpoint or another
    sheet, and no intersection makes that comparable.
    """
    disagreeing = [p for p in ref if p in other and not np.array_equal(ref[p][2], other[p][2])]
    if disagreeing:
        raise BuildDbError(
            f"'{field}': arms '{reference}' and '{arm}' disagree on the label of "
            f"{len(disagreeing)} patient(s), e.g. {disagreeing[:5]}. One of them is reading a "
            f"different endpoint or a different sheet."
        )


def _score(rows: Dict[str, tuple], metric: Callable[[np.ndarray, np.ndarray], float]) -> ArmScores:
    folds = sorted({row[0] for row in rows.values()})
    per_fold = {}
    for fold in folds:
        keys = [p for p, row in rows.items() if row[0] == fold]
        labels = np.array([rows[p][2] for p in keys])
        preds = np.array([rows[p][1] for p in keys])
        per_fold[fold] = metric(labels, preds)
    labels = np.array([row[2] for row in rows.values()])
    preds = np.array([row[1] for row in rows.values()])
    return ArmScores(arm="", per_fold=per_fold, pooled=metric(labels, preds))


def compare_arms(
    results_dir: Path,
    cohort: str,
    arms: Sequence[str],
    reference: str,
    objective: str = "binary",
    fields: Sequence[str] | None = None,
    metric_key: str | None = None,
) -> List[FieldComparison]:
    """Every arm's metric per endpoint, and each arm's paired per-fold delta against ``reference``.

    ``metric_key`` selects which of the objective's metrics to score on (None = the endpoint default)."""
    if reference not in arms:
        raise BuildDbError(f"reference '{reference}' is not among the arms {list(arms)}.")
    if len(arms) < 2:
        raise BuildDbError(f"comparison needs at least two arms, got {list(arms)}.")
    metric = resolve_metric(objective, metric_key).fn

    by_arm = {arm: read_arm(results_dir, arm, cohort, objective) for arm in arms}
    shared = set.intersection(*(set(tables) for tables in by_arm.values()))
    wanted = [f for f in (fields or sorted(shared)) if f in shared]
    if not wanted:
        raise BuildDbError(
            f"arms {list(arms)} share no endpoint (each has: { {a: sorted(t) for a, t in by_arm.items()} })."
        )

    comparisons = []
    for field in wanted:
        ref_rows = by_arm[reference][field]
        coverage = {arm: len(by_arm[arm][field]) for arm in arms}
        # Every number is computed on the patients all arms scored, so the metric and the paired
        # delta describe one population. What each arm covered on its own is kept and reported: the
        # gap between an arm's coverage and this set is a property of the arm, not an accounting
        # detail, and collapsing it into the metric would confound coverage with discrimination.
        common = set.intersection(*(set(by_arm[arm][field]) for arm in arms))
        if not common:
            raise BuildDbError(
                f"'{field}': arms {list(arms)} share no patient (each scored: {coverage}). "
                f"There is nothing to compare them on."
            )
        scores, deltas, pooled_deltas = [], {}, {}
        for arm in arms:
            rows = by_arm[arm][field]
            # The label-agreement guard assumes every arm predicts the same target. Expression arms
            # legitimately predict different gene panels (baseline full, hvg2000, hallmark), so their
            # label vectors differ by design and there is no single target to agree on, skip it.
            if arm != reference and objective != "regression_vector":
                _assert_comparable(field, reference, ref_rows, arm, rows)
            scored = _score({p: rows[p] for p in common}, metric)
            scores.append(ArmScores(arm=arm, per_fold=scored.per_fold, pooled=scored.pooled))
        ref_scores = next(s for s in scores if s.arm == reference)
        for scored in scores:
            if scored.arm == reference:
                continue
            deltas[scored.arm] = {f: scored.per_fold[f] - ref_scores.per_fold[f] for f in ref_scores.per_fold}
            pooled_deltas[scored.arm] = scored.pooled - ref_scores.pooled
        comparisons.append(
            FieldComparison(
                field=field,
                reference=reference,
                n_patients=len(common),
                scores=scores,
                deltas=deltas,
                pooled_deltas=pooled_deltas,
                coverage=coverage,
            )
        )
    return comparisons


def _fmt(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{value:.3f}"


def _absolute_figure(comparisons: List[FieldComparison], metric_name: str, chance: float | None, path: Path) -> None:
    """Per-fold metric per arm, one column per arm, mean +- std marked.

    The delta plot places the arms relative to a reference. This places them absolutely, so a
    difference can be read against how good the arms are in the first place.
    """
    plt = _pyplot()
    fig, axes = plt.subplots(1, len(comparisons), figsize=(4.0 + 1.4 * len(comparisons), 4.0), squeeze=False)
    for axis, comparison in zip(axes[0], comparisons):
        seen: List[float] = []
        for idx, scored in enumerate(comparison.scores):
            values = np.array([v for v in scored.per_fold.values() if np.isfinite(v)])
            if values.size == 0:
                continue
            seen.extend(values.tolist())
            jitter = np.linspace(-0.08, 0.08, values.size)
            axis.scatter(np.full(values.size, idx) + jitter, values, s=40, alpha=0.85)
            axis.hlines(values.mean(), idx - 0.22, idx + 0.22, colors="black", lw=1.5)
            axis.text(
                idx + 0.28, values.mean(), f"{values.mean():.3f} ± {values.std(ddof=0):.3f}", va="center", fontsize=8
            )
        n = len(comparison.scores)
        if chance is not None:
            axis.axhline(chance, color="grey", lw=0.8, linestyle="--", alpha=0.7)
            axis.text(n - 0.5 + 0.55, chance, "chance", color="grey", fontsize=7, va="bottom", ha="right")
        # Same fixed, comparable window the aggregate's strip uses, widened only if a fold falls outside.
        lo = min(0.4, min(seen) - 0.03) if seen else 0.4
        axis.set_xticks(range(n))
        axis.set_xticklabels([s.arm for s in comparison.scores], rotation=20, ha="right")
        axis.set(
            xlim=(-0.5, n - 0.5 + 0.6),
            ylim=(max(0.0, lo), 1.01),
            ylabel=metric_name,
            title=f"{comparison.field} (n={comparison.n_patients})",
        )
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _delta_figure(comparisons: List[FieldComparison], metric_name: str, path: Path) -> None:
    """Per-fold deltas against the reference, one row of points per arm per endpoint.

    The paired difference is plotted, not the two arms' marginals: both arms see the same patients in
    the same folds, so an easy fold lifts both and a hard one drops both. Marginals can overlap almost
    entirely while one arm wins every fold.
    """
    plt = _pyplot()
    fig, axes = plt.subplots(len(comparisons), 1, figsize=(7, 2.2 * len(comparisons)), squeeze=False)
    for axis, comparison in zip(axes[:, 0], comparisons):
        arms = sorted(comparison.deltas)
        for row, arm in enumerate(arms):
            values = [v for v in comparison.deltas[arm].values() if not np.isnan(v)]
            axis.scatter(values, [row] * len(values), s=44, alpha=0.75, color="#4c72b0", zorder=3)
            axis.scatter([comparison.pooled_deltas[arm]], [row], s=90, marker="|", color="#c44e52", zorder=4)
        axis.axvline(0.0, color="#9ca3af", lw=1.2, zorder=1)
        axis.set(
            yticks=range(len(arms)),
            yticklabels=arms,
            xlabel=f"per-fold delta {metric_name} vs {comparison.reference}",
            title=f"{comparison.field} (n={comparison.n_patients})",
        )
        axis.grid(True, axis="x", lw=0.3, alpha=0.4)
        axis.set_ylim(-0.6, len(arms) - 0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_comparison(
    comparisons: List[FieldComparison],
    name: str,
    out_dir: Path,
    metric_name: str = "AUROC",
    chance: float | None = 0.5,
) -> Path:
    """Write ``summary.md`` + ``metric.png`` + ``deltas.png`` + ``per_fold.csv`` for one comparison."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _absolute_figure(comparisons, metric_name, chance, out_dir / "metric.png")
    _delta_figure(comparisons, metric_name, out_dir / "deltas.png")

    with (out_dir / "per_fold.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["field", "arm", "outer_fold", metric_name.lower(), f"delta_vs_{comparisons[0].reference}"])
        for comparison in comparisons:
            for scored in comparison.scores:
                for fold, value in sorted(scored.per_fold.items()):
                    delta = comparison.deltas.get(scored.arm, {}).get(fold)
                    writer.writerow(
                        [comparison.field, scored.arm, fold, f"{value:.6f}", "" if delta is None else f"{delta:.6f}"]
                    )

    lines = [f"# comparison: {name}", "", f"Reference arm: `{comparisons[0].reference}`", ""]
    for comparison in comparisons:
        folds = sorted(comparison.scores[0].per_fold)
        lines += [f"## {comparison.field}", "", f"{comparison.n_patients} patients", ""]
        if comparison.coverage_differs:
            # Coverage is a result, not bookkeeping: an arm that scores patients another cannot is
            # better in a way no metric on the shared set can show. Stated before the metric so the
            # table below is never read as if it covered everyone.
            widest = max(comparison.coverage.values())
            lines += [
                f"**The arms did not score the same patients.** Every number below is computed on the "
                f"{comparison.n_patients} patients ALL arms scored. Coverage, out of the {widest} "
                f"scored by the widest arm:",
                "",
                "| arm | scored | not scored |",
                "| --- | --- | --- |",
            ]
            for arm in [s.arm for s in comparison.scores]:
                got = comparison.coverage[arm]
                lines.append(f"| {arm} | {got} | {widest - got} |")
            lines += [
                "",
                "A gap here is a property of the arm. A tiling that clears `min_tiles` on slides a "
                "coarser one leaves empty covers more patients, and that advantage does not appear in "
                "the metric below, which holds the population fixed on purpose.",
                "",
            ]
        lines += [
            "| arm | " + " | ".join(f"o{f}" for f in folds) + " | pooled |",
            "| --- | " + " | ".join("---" for _ in folds) + " | --- |",
        ]
        for scored in comparison.scores:
            marker = " (ref)" if scored.arm == comparison.reference else ""
            lines.append(
                f"| {scored.arm}{marker} | "
                + " | ".join(_fmt(scored.per_fold[f]) for f in folds)
                + f" | {_fmt(scored.pooled)} |"
            )
        lines += [
            "",
            f"Paired delta vs `{comparison.reference}`:",
            "",
            "| arm | " + " | ".join(f"o{f}" for f in folds) + " | pooled |",
            "| --- | " + " | ".join("---" for _ in folds) + " | --- |",
        ]
        for arm in sorted(comparison.deltas):
            lines.append(
                f"| {arm} | "
                + " | ".join(_fmt(comparison.deltas[arm][f]) for f in folds)
                + f" | {_fmt(comparison.pooled_deltas[arm])} |"
            )
        lines.append("")
    n_outer_folds = len(comparisons[0].scores[0].per_fold)
    lines += [
        f"![per-fold {metric_name}](metric.png)",
        "",
        "![per-fold deltas](deltas.png)",
        "",
        f"No test, interval or p-value is computed here. The {n_outer_folds} outer folds share training data, so their",
        f"differences are not independent, and {n_outer_folds} of them cannot support inference. This says whether a",
        "difference looks worth a more expensive experiment, nothing more.",
        "",
    ]
    summary = out_dir / "summary.md"
    summary.write_text("\n".join(lines))
    return summary
