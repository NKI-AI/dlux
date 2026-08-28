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
"""External-cohort scoring: ensemble a development sweep's models over a held-out cohort.

Pure numpy (no torch/ahcore), so trivially testable. The inference driver
(``bin/evaluate_external``) runs each of the development sweep's K×J models over the
external cohort and rolls each model's slide-probs to patient level. This module then
ensembles those per-model patient-probs into the two numbers we report:

- **grand ensemble**, mean-of-probs over all K×J models → a single point estimate. This
  is the *headline*: it's the deployable model, and it has no leakage (no model saw the
  external cohort).
- **per-outer stability**, for each outer fold, mean over its J inner models → one score,
  reported as mean ± std across the K outer ensembles. This is a *stability diagnostic*
  ("does the result survive which outer split trained it"), not an uncertainty band, the
  K scores are the same external patients, so the spread is model-variability, not sampling
  variance. It must not be compared apples-to-apples with the internal per-outer band.

Endpoint-varying behaviour (metric fn, scalar-vs-vector prob shape, per-patient CSV, figure,
per-class table) is dispatched through ``_HANDLERS`` keyed on ``endpoint_type``, the same
per-subsystem registry pattern ``aggregate`` uses, so no function re-branches on the type.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from dlux.config.cohort import Objective
from dlux.eval._common import _ExternalHandler, _fmt
from dlux.eval.binary import EXTERNAL_HANDLER as _BINARY_EXTERNAL_HANDLER
from dlux.eval.multiclass import EXTERNAL_HANDLER as _MULTICLASS_EXTERNAL_HANDLER
from dlux.eval.regression import EXTERNAL_HANDLER as _REGRESSION_EXTERNAL_HANDLER
from dlux.eval.regression_vector import EXTERNAL_HANDLER as _EXPRESSION_EXTERNAL_HANDLER
from dlux.eval.survival import EXTERNAL_HANDLER as _SURVIVAL_EXTERNAL_HANDLER

logger = logging.getLogger(__name__)

# per-model patient probabilities: {(outer, inner): {patient_code: prob}}, prob is a positive-class
# float (binary) or a (K,) per-class softmax array (multiclass). The endpoint handler interprets it.
PerModelProbs = Dict[Tuple[int, int], Dict[str, object]]


@dataclass
class ExternalResult:
    field: str
    cohort: str
    n_patients: int
    n_positive: int
    n_models: int
    grand: Dict[str, float]  # grand K×J ensemble metrics (headline)
    endpoint_type: Objective = Objective.binary
    num_classes: int = 1
    outer_folds: List[int] = dc_field(default_factory=list)  # sorted outer-fold ids (row labels)
    per_outer: List[Dict[str, float]] = dc_field(default_factory=list)  # per-outer-ensemble metrics
    per_outer_mean: Dict[str, float] = dc_field(default_factory=dict)  # stability band (not an uncertainty CI)
    per_outer_std: Dict[str, float] = dc_field(default_factory=dict)
    # per-patient ensembled probs: scalar (binary) or (K,) vector (multiclass); the handler interprets them.
    grand_patient_probs: Dict[str, object] = dc_field(default_factory=dict)
    per_outer_patient_probs: Dict[int, Dict[str, object]] = dc_field(default_factory=dict)
    # Per-patient label, shaped by the endpoint: a class index (binary/multiclass), a raw scalar
    # (regression), or a vector such as survival's (time, event). The handler interprets it.
    labels: Dict[str, Any] = dc_field(default_factory=dict)
    # Expression only: SEQUOIA's conservative significant-gene count vs a random-init model, when a
    # random-baseline arm was supplied. {n_significant, n_scored, alpha, fdr}; None otherwise.
    conservative: Any = None


def _mean_std(per_outer: List[Dict[str, float]], metric_labels: Dict[str, str]) -> Tuple[Dict, Dict]:
    """Mean and population std of each metric across outer-fold ensembles, ignoring NaNs."""
    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}
    for key in metric_labels:
        vals = np.asarray([m[key] for m in per_outer], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        mean[key] = float(finite.mean()) if finite.size else float("nan")
        std[key] = float(finite.std(ddof=0)) if finite.size else float("nan")
    return mean, std


def external_ensemble(
    field: str,
    cohort: str,
    per_model: PerModelProbs,
    labels: Dict[str, Any],
    *,
    endpoint_type: Objective = Objective.binary,
    num_classes: int = 1,
) -> ExternalResult:
    """Ensemble per-model patient-probs on the external cohort into grand + per-outer metrics.

    Every model must cover exactly the same external patients (the whole cohort). The endpoint
    handler supplies the metric fn + prob shape (scalar for binary, (K,) softmax for multiclass;
    class = argmax).
    """
    if not per_model:
        raise ValueError(f"[{field}/{cohort}] no development models supplied to ensemble")
    endpoint_type = Objective(endpoint_type)  # normalize str|Objective -> enum (stored on the result)

    # The all_test_<target> split filters to valid-label patients up front, so every scored patient
    # must carry a valid label here. A missing label is not something to quietly drop. It means the
    # cohort DB is stale or the wrong split was scored, so fail loudly rather than paper over it. What
    # counts as missing is the endpoint's business (a negative regression target is data, not a
    # sentinel), so the rule comes from the handler.
    handler = _handler(endpoint_type)
    patients = sorted(labels)
    invalid = [p for p in patients if handler.is_missing(labels[p])]
    if invalid:
        raise ValueError(
            f"[{field}/{cohort}] {len(invalid)} scored patient(s) carry a missing label "
            f"(e.g. {invalid[:8]}) — the all_test_{field} split must exclude these before inference. "
            f"Rebuild the cohort DB."
        )
    if not patients:
        raise ValueError(f"[{field}/{cohort}] no patients to score.")
    patient_set = set(patients)
    for (o, i), probs in per_model.items():
        if not patient_set <= set(probs):
            raise ValueError(
                f"[{field}/{cohort}] model o{o}_i{i} is missing scored patients "
                f"(absent: {sorted(patient_set - set(probs))[:8]})"
            )

    label_arr = handler.label_array([labels[p] for p in patients])
    outer_folds = sorted({outer for (outer, _inner) in per_model})

    def _ensemble(models: list) -> Dict[str, object]:
        """Mean-of-probs over the given models, per patient (scalar or (K,) vector per the handler)."""
        if handler.vector_probs:
            return {p: np.mean(np.stack([np.asarray(per_model[m][p]) for m in models]), axis=0) for p in patients}
        return {p: float(np.mean([per_model[m][p] for m in models])) for p in patients}

    def _metrics(prob_map: Dict[str, object]) -> Dict[str, float]:
        if handler.vector_probs:
            return handler.metrics(label_arr, np.stack([prob_map[p] for p in patients]), num_classes)
        return handler.metrics(label_arr, np.asarray([prob_map[p] for p in patients], dtype=np.float64), num_classes)

    grand_probs = _ensemble(list(per_model))
    grand = _metrics(grand_probs)

    per_outer: List[Dict[str, float]] = []
    per_outer_probs: Dict[int, Dict[str, object]] = {}
    for o in outer_folds:
        inner_models = [m for m in per_model if m[0] == o]
        o_prob_map = _ensemble(inner_models)
        per_outer_probs[o] = o_prob_map
        per_outer.append(_metrics(o_prob_map))
    mean, std = _mean_std(per_outer, handler.metric_labels)

    return ExternalResult(
        field=field,
        cohort=cohort,
        n_patients=len(patients),
        n_positive=handler.count_positive(label_arr) if handler.has_positive_count else 0,
        n_models=len(per_model),
        grand=grand,
        endpoint_type=endpoint_type,
        num_classes=num_classes,
        outer_folds=outer_folds,
        per_outer=per_outer,
        per_outer_mean=mean,
        per_outer_std=std,
        grand_patient_probs=grand_probs,
        per_outer_patient_probs=per_outer_probs,
        labels=labels,
    )


# -- reporting ---------------------------------------------------------------
def _field_section(result: ExternalResult) -> list[str]:
    """One markdown section for a field: per-outer rows (o0…oK) + stability + grand, plus any
    endpoint-specific extra (multiclass per-class table). Columns come from the endpoint handler."""
    handler = _handler(result.endpoint_type)
    labels = handler.metric_labels
    cols = list(labels.values())
    rows = ["| fold | " + " | ".join(cols) + " |", "|:--|" + "|".join(["--:"] * len(cols)) + "|"]
    for outer, m in zip(result.outer_folds, result.per_outer):
        rows.append(f"| o{outer} | " + " | ".join(_fmt(m[k]) for k in labels) + " |")
    rows.append(
        "| **stability (mean±std)** | "
        + " | ".join(f"{_fmt(result.per_outer_mean[k])}±{_fmt(result.per_outer_std[k])}" for k in labels)
        + " |"
    )
    rows.append("| **GRAND ensemble** | " + " | ".join(f"**{_fmt(result.grand[k])}**" for k in labels) + " |")
    positive = f" ({result.n_positive} positive)" if handler.has_positive_count else ""
    return [
        f"## `{result.field}`",
        "",
        f"{result.n_patients} patients{positive} | {result.n_models} development models",
        "",
        *rows,
        "",
        *handler.section_extra(result),
    ]


def write_external_report(results: list[ExternalResult], experiment_name: str, out_dir: str | Path) -> Path:
    """Write per-patient CSV(s) + external_summary.md + figures for all scored endpoints.

    Mirrors ``dlux.eval.aggregate.write_reports``: a section per endpoint, a per-endpoint figure, and
    one combined per-outer AUROC strip. Per-patient CSV + figure are dispatched via ``_HANDLERS``."""
    out_dir = Path(out_dir)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not results:
        logger.warning("No external results to write under %s.", out_dir)
        return out_dir
    cohort = results[0].cohort

    # Per-patient CSV: each endpoint handler owns its file strategy (binary → one combined long CSV;
    # multiclass → per-field per-class CSVs). Group by type, hand each group to its handler.
    by_type: Dict[str, List[ExternalResult]] = {}
    for res in results:
        by_type.setdefault(res.endpoint_type, []).append(res)
    for endpoint_type, group in by_type.items():
        _handler(endpoint_type).write_patient_csv(group, out_dir)

    lines = [f"# External validation — {experiment_name} on `{cohort}`", ""]
    for res in results:
        lines += _field_section(res)
    lines += [
        "_**GRAND ensemble** (mean-of-probs over all K×J models) is the HEADLINE — the deployable, "
        "leakage-free point estimate. The per-outer rows + **stability (mean±std)** are a "
        "*stability diagnostic* (does the result survive which outer split trained it), NOT an "
        "uncertainty interval: the K scores are the same external patients, so the spread is "
        "model-variability, and must not be compared apples-to-apples with the internal per-outer band._",
        "",
    ]
    path = out_dir / "external_summary.md"
    path.write_text("\n".join(lines))

    for res in results:
        _handler(res.endpoint_type).write_artifact(res, figures_dir)
    # One combined strip per metric family: AUROC and C-index are both [0,1] with chance at 0.5, but
    # they are not the same quantity and must not share an axis.
    by_strip: Dict[tuple, List[ExternalResult]] = {}
    for res in results:
        h = _handler(res.endpoint_type)
        if not h.strip_group:  # opted out. Its scores do not share an axis with anything (regression)
            continue
        by_strip.setdefault((h.strip_group, h.strip_metric), []).append(res)
    for (_group, metric), group_results in by_strip.items():
        _plot_external_strip(group_results, metric, figures_dir / f"{metric}_by_outer.png")

    logger.info("Wrote external report (%d endpoint(s)) -> %s", len(results), path)
    return out_dir


def _plot_external_strip(results: list[ExternalResult], metric: str, path: Path) -> None:
    """Combined strip: one column per endpoint, per-outer scores (stability) as points, the
    grand-ensemble score as a bar, chance dashed. One call per metric family (see write_external_report):
    ``auroc`` for binary/multiclass, ``c_index`` for survival."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(max(4.5, 1.7 * len(results)), 4.5))
    all_vals: list[float] = []
    for idx, res in enumerate(results):
        scores = np.asarray([m[metric] for m in res.per_outer if np.isfinite(m[metric])])
        if scores.size:
            jitter = np.linspace(-0.08, 0.08, scores.size)
            ax.scatter(np.full(scores.size, idx) + jitter, scores, s=40, alpha=0.85, color="C0", zorder=3)
            all_vals.extend(scores.tolist())
        grand = res.grand[metric]
        if np.isfinite(grand):
            ax.hlines(grand, idx - 0.24, idx + 0.24, colors="C3", lw=2.2, zorder=4)  # grand ensemble
            ax.text(idx + 0.28, grand, f"{grand:.3f}", va="center", fontsize=8, color="C3")
            all_vals.append(grand)
    ax.axhline(0.5, color="grey", lw=0.8, linestyle="--", alpha=0.7)  # chance

    handles = [
        Line2D([0], [0], marker="o", color="C0", linestyle="", label="per-outer ensemble"),
        Line2D([0], [0], color="C3", lw=2.2, label="GRAND ensemble"),
        Line2D([0], [0], color="grey", lw=0.8, linestyle="--", label="chance"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    y_lo = min(0.4, min(all_vals) - 0.03) if all_vals else 0.4
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels([res.field for res in results])
    label = _handler(results[0].endpoint_type).metric_labels.get(metric, metric)
    ax.set(xlim=(-0.5, len(results) - 0.5 + 0.6), ylim=(max(0.0, y_lo), 1.01), ylabel=label)
    ax.set_title(f"External per-outer {label} — {results[0].cohort}")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -- endpoint handlers (dispatched on endpoint_type via _HANDLERS) ------------
# One handler per endpoint kind owns external eval's endpoint-varying behaviour: the metric fn, the
# per-patient prob shape, whether a positive count applies, the per-patient CSV strategy, the figure,
# and the summary's extra lines. external_ensemble + write_external_report look the handler up once.
# Capability is not gated on whether we ship an example study for it: a consumer brings their own
# external cohort. Each handler carries its own label model (shape, missing rule, positive count), so
# adding an objective here needs no change to the scoring core. All five objectives are now wired.
_HANDLERS: Dict[str, _ExternalHandler] = {
    "binary": _BINARY_EXTERNAL_HANDLER,
    "multiclass": _MULTICLASS_EXTERNAL_HANDLER,
    "regression": _REGRESSION_EXTERNAL_HANDLER,
    "regression_vector": _EXPRESSION_EXTERNAL_HANDLER,
    "survival": _SURVIVAL_EXTERNAL_HANDLER,
}


def supported_objectives() -> list[str]:
    """The objectives external eval can actually report on, read from the handler registry, so a caller
    cannot advertise a capability that is not wired."""
    return sorted(_HANDLERS)


def _handler(endpoint_type: Objective) -> _ExternalHandler:
    try:
        return _HANDLERS[endpoint_type]  # str-enum keys the string-keyed registry transparently (Step 3 re-keys)
    except KeyError:
        raise NotImplementedError(f"external eval supports {sorted(_HANDLERS)} (got '{endpoint_type}').")


def uses_vector_probs(endpoint_type: Objective) -> bool:
    """Whether this endpoint's per-patient prediction is a vector (drives the slide→patient roll)."""
    return _handler(endpoint_type).vector_probs


def uses_vector_labels(endpoint_type: Objective) -> bool:
    """Whether this endpoint's per-patient label is a vector. Independent of ``uses_vector_probs``:
    multiclass has vector predictions and a scalar label, survival the other way round."""
    return _handler(endpoint_type).vector_labels


def cast_scalar_label(endpoint_type: Objective, value: float):
    """What a scalar label is for this endpoint (class index vs raw value), see the handler field."""
    return _handler(endpoint_type).scalar_label_cast(value)


def headline(result: ExternalResult) -> str:
    """One-line stdout summary on the endpoint's primary metric (``strip_metric``), AUROC for
    classification, C-index for survival, so the driver never names a metric an endpoint lacks."""
    h = _handler(result.endpoint_type)
    key = h.strip_metric
    label = h.metric_labels.get(key, key)
    return (
        f"External {result.field} on {result.cohort}: GRAND {label}={_fmt(result.grand[key])} "
        f"(stability {_fmt(result.per_outer_mean[key])}±{_fmt(result.per_outer_std[key])}, "
        f"{result.n_models} models)."
    )
