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
"""Reads what one model of a sweep left behind, for inspection.

A model is addressed by (study, experiment, cohort, field, inner). Each slide is tested by exactly one
outer fold, so fixing the inner replicate and merging outer folds covers the cohort with one model per
slide. Predictions here are that single model's own, not the cross-replicate ensemble the aggregate
reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

import numpy as np

from dlux.eval.survival_metrics import bin_hazards, bin_survival, kaplan_meier

_ATTENTION = "attention.npz"
_PREDICTIONS = "test_predictions.npz"


@dataclass(frozen=True)
class SurvivalCurve:
    """A slide's predicted survival function, as the step points a reader can plot.

    ``edges`` are the interior bin boundaries in the time column's units and ``surv[j]`` is
    P(surviving past ``edges[j]``), so the curve is ``1`` before the first edge and steps down at each
    one. ``tail`` is the survival at the end of the open last bin (past the final edge), a number
    rather than a step, since that bin has no right boundary to draw.
    """

    edges: list[float]
    surv: list[float]
    hazards: list[float]
    tail: float


@dataclass(frozen=True)
class SlidePrediction:
    """One slide's prediction from the model that tested it."""

    prob: float | list[float]
    label: float | list[float] | None
    endpoint_type: str
    fold: str
    survival: SurvivalCurve | None = None
    # Needed to roll slides up to patients, which survival statistics must do: several slides of one
    # patient share one (time, event) and would otherwise be counted as several patients.
    patient_code: str = ""


@dataclass(frozen=True)
class RegressionSummary:
    """Cohort-level context a single regression prediction is unreadable without.

    ``lo``/``hi`` span both labels and predictions, so neither marker ever needs clamping.
    """

    lo: float
    hi: float
    median_abs_error: float
    pearson_r: float
    label_sd: float
    n_slides: int


@dataclass(frozen=True)
class SurvivalSummary:
    """Cohort-level context a single predicted curve is unreadable without.

    The risk score ``-sum_j S_j`` has an arbitrary scale, so the only statement it supports about one
    slide is its rank among the others this model scored (``RunRecords.risk_percentile``).
    """

    n_slides: int
    n_events: int
    median_follow_up: float
    n_patients: int
    # Kaplan-Meier over the patients this model scored, as ``(t, S)`` step arrays. The reference a
    # single predicted curve is unreadable without: it says what the cohort does, so one patient's
    # curve can be read as a deviation from it rather than as an unscoreable number.
    km_times: list[float]
    km_surv: list[float]


@dataclass(frozen=True)
class RunRecords:
    """One model's outputs across the slides it scored.

    ``mpp`` and ``tile_size`` describe the extraction grid; a tile's footprint in a slide's level-0
    pixels is ``tile_size * mpp / slide_mpp``.
    """

    attention: dict[str, np.ndarray]
    predictions: dict[str, SlidePrediction]
    mpp: float
    tile_size: int
    folds: list[str] = dataclass_field(default_factory=list)
    aggregated: bool = False
    # outer fold -> inner replicates that contributed; identical for every slide of that fold.
    inners_by_outer: dict[str, list[int]] = dataclass_field(default_factory=dict)
    # slide -> median pairwise rank correlation between the contributing models.
    consistency: dict[str, float] = dataclass_field(default_factory=dict)
    # slide -> the outer fold that tested it.
    outer_of: dict[str, str] = dataclass_field(default_factory=dict)
    regression: RegressionSummary | None = None
    # slide -> fraction of slides this one beats on |error|; regression endpoints only.
    error_percentile: dict[str, float] = dataclass_field(default_factory=dict)
    survival: SurvivalSummary | None = None
    # slide -> fraction of slides it out-risks; survival endpoints only.
    risk_percentile: dict[str, float] = dataclass_field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.attention)


def class_names(field: str, mapping: dict[str, int] | None) -> dict[int, str]:
    """Names each class by the source values that map to it, e.g. ``{0: "grade 1, 2", 1: "grade 3"}``.

    Returns an empty dict when the endpoint has no categorical map, leaving callers to fall back to
    bare class indices.
    """
    if not mapping:
        return {}
    grouped: dict[int, list[str]] = {}
    for value, klass in mapping.items():
        grouped.setdefault(int(klass), []).append(str(value))
    return {klass: f"{field} {', '.join(sorted(values))}" for klass, values in grouped.items()}


def _survival_curve(logits: np.ndarray, edges: np.ndarray) -> SurvivalCurve:
    """One slide's hazard logits + this fold's bin edges -> the plottable survival curve."""
    hazards = bin_hazards(logits)
    surv = bin_survival(hazards)
    return SurvivalCurve(
        edges=[float(e) for e in edges],
        surv=[float(s) for s in surv[: len(edges)]],  # one step per interior edge
        hazards=[float(h) for h in hazards],
        tail=float(surv[-1]),  # the open last bin has no right edge to step at
    )


def survival_at(curve: SurvivalCurve, times: list[float]) -> np.ndarray:
    """Evaluate a curve's step function at arbitrary ``times``.

    The bridge between folds: each fold quantiles its own fit split, so two curves share no bin
    boundaries and their hazards are not comparable index-by-index. Evaluated on one grid they are.
    """
    edges = np.asarray(curve.edges, dtype=np.float64)
    # Survival before the first edge is 1; at or past edge j it is surv[j]; searchsorted picks the step.
    steps = np.concatenate([[1.0], np.asarray(curve.surv, dtype=np.float64)])
    return steps[np.searchsorted(edges, np.asarray(times, dtype=np.float64), side="right")]


def _combine_curves(curves: list[SurvivalCurve]) -> SurvivalCurve:
    """Ensemble survival curves by averaging S(t) on the union of their edges.

    Bin j spans a different interval in every fold, so curves are comparable only once evaluated on a
    shared grid. The union of the edges is the finest grid every curve can be read on.
    """
    grid = sorted({edge for curve in curves for edge in curve.edges})
    mean = np.mean([survival_at(curve, grid) for curve in curves], axis=0)
    return SurvivalCurve(
        edges=grid,
        surv=[float(s) for s in mean],
        # Per-bin hazards are not averageable across folds; the ensemble carries the curve only.
        hazards=[],
        tail=float(np.mean([curve.tail for curve in curves])),
    )


def _read_predictions(path: Path, fold: str) -> dict[str, SlidePrediction]:
    with np.load(path, allow_pickle=True) as handle:
        slide_ids = [str(s) for s in handle["slide_ids"]]
        patient_codes = [str(p) for p in handle["patient_codes"]]
        probs, labels, logits = handle["probs"], handle["labels"], handle["logits"]
        endpoint = str(handle["endpoint_type"]) if "endpoint_type" in handle.files else ""
        edges = handle["time_edges"] if "time_edges" in handle.files else None
    if endpoint == "survival" and edges is None:
        raise ValueError(
            f"{path} holds survival predictions without `time_edges`, so its per-bin hazards cannot be "
            "placed on a time axis. Re-run the fold so training persists the bin edges."
        )
    out: dict[str, SlidePrediction] = {}
    for i, slide_id in enumerate(slide_ids):
        prob = probs[i].tolist() if probs.ndim > 1 else float(probs[i])
        label = None
        if labels is not None and len(labels):
            label = labels[i].tolist() if labels.ndim > 1 else float(labels[i])
        curve = _survival_curve(logits[i], edges) if endpoint == "survival" else None
        out[slide_id] = SlidePrediction(
            prob=prob,
            label=label,
            endpoint_type=endpoint,
            fold=fold,
            survival=curve,
            patient_code=patient_codes[i],
        )
    return out


def _regression_context(
    predictions: dict[str, SlidePrediction],
) -> tuple[RegressionSummary | None, dict[str, float]]:
    """Summarises a regression run and ranks each slide by absolute error.

    Returns ``(None, {})`` unless the endpoint is scalar regression with labels.
    """
    usable = {
        k: v
        for k, v in predictions.items()
        if v.endpoint_type == "regression" and v.label is not None and not isinstance(v.prob, list)
    }
    if len(usable) < 2:
        return None, {}
    keys = list(usable)
    pred = np.array([float(usable[k].prob) for k in keys])
    true = np.array([float(usable[k].label) for k in keys])
    abs_error = np.abs(pred - true)
    # Fraction of slides with a larger error, i.e. how many this slide beats.
    order = abs_error.argsort().argsort()
    percentile = {k: float((len(keys) - 1 - order[i]) / max(len(keys) - 1, 1)) for i, k in enumerate(keys)}
    summary = RegressionSummary(
        lo=float(min(true.min(), pred.min())),
        hi=float(max(true.max(), pred.max())),
        median_abs_error=float(np.median(abs_error)),
        pearson_r=float(np.corrcoef(pred, true)[0, 1]) if pred.std() and true.std() else 0.0,
        label_sd=float(true.std()),
        n_slides=len(keys),
    )
    return summary, percentile


def _survival_context(
    predictions: dict[str, SlidePrediction],
) -> tuple[SurvivalSummary | None, dict[str, float]]:
    """Summarises a survival run and ranks each slide by risk.

    Returns ``(None, {})`` unless the endpoint is survival with coupled ``[time, event]`` labels.
    """
    usable = {
        k: v
        for k, v in predictions.items()
        if v.endpoint_type == "survival" and isinstance(v.label, list) and len(v.label) == 2
    }
    if len(usable) < 2:
        return None, {}
    keys = list(usable)
    time = np.array([float(usable[k].label[0]) for k in keys])
    event = np.array([float(usable[k].label[1]) for k in keys])
    # Both statistics below are about patients, so a patient with several slides counts once: their
    # (time, event) is a patient property, and their risk is the mean over their slides, the rollup
    # the aggregate uses, so the rank shown here agrees with the reported numbers.
    risk_by_patient: dict[str, list[float]] = {}
    label_by_patient: dict[str, tuple[float, float]] = {}
    for k in keys:
        code = usable[k].patient_code or k
        risk_by_patient.setdefault(code, []).append(float(usable[k].prob))
        label_by_patient[code] = (float(usable[k].label[0]), float(usable[k].label[1]))
    codes = list(risk_by_patient)
    patient_risk = np.array([float(np.mean(risk_by_patient[c])) for c in codes])
    order = patient_risk.argsort().argsort()
    # Fraction of patients this one out-risks; every slide of a patient reports the patient's rank.
    rank_of = {c: float(order[i] / max(len(codes) - 1, 1)) for i, c in enumerate(codes)}
    percentile = {k: rank_of[usable[k].patient_code or k] for k in keys}
    pairs = np.asarray([label_by_patient[c] for c in codes], dtype=np.float64)
    km_times, km_surv = kaplan_meier(pairs[:, 0], pairs[:, 1])
    summary = SurvivalSummary(
        n_slides=len(keys),
        n_events=int(event.sum()),
        median_follow_up=float(np.median(time)),
        n_patients=len(codes),
        km_times=[float(t) for t in km_times],
        km_surv=[float(s) for s in km_surv],
    )
    return summary, percentile


def _weights(records: np.ndarray) -> np.ndarray:
    """Softmax over one slide's logits. Each model's logits carry an arbitrary offset, so they must be
    normalised individually before any combination."""
    logits = records[:, 2].astype(np.float64)
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def _rank_agreement(weights: list[np.ndarray]) -> float:
    """Median pairwise Spearman correlation between models' per-tile weights.

    NaN when there is nothing to correlate: fewer than two models, or a slide of a single tile (whose
    ranking is the same trivially, and whose correlation is undefined rather than perfect).
    """
    if len(weights) < 2 or any(w.size < 2 for w in weights):
        return float("nan")
    ranks = [w.argsort().argsort().astype(np.float64) for w in weights]
    scores = [float(np.corrcoef(ranks[i], ranks[j])[0, 1]) for i in range(len(ranks)) for j in range(i + 1, len(ranks))]
    return float(np.median(scores))


def _combine(stacked: list[np.ndarray]) -> np.ndarray:
    """Mean of the per-model softmax weights, re-expressed as logits.

    Storing ``log(mean weight)`` keeps the artifact shape identical to a single model's, so consumers
    softmax it and recover the mean weights exactly.
    """
    mean = np.mean([_weights(a) for a in stacked], axis=0)
    out = stacked[0].copy()
    out[:, 2] = np.log(np.maximum(mean, np.finfo(np.float64).tiny))
    return out


def _average_predictions(entries: list[SlidePrediction], fold: str) -> SlidePrediction:
    """Mean of the models' outputs, matching how the aggregate ensembles predictions."""
    first = entries[0]
    if isinstance(first.prob, list):
        prob: float | list[float] = np.mean([np.asarray(e.prob) for e in entries], axis=0).tolist()
    else:
        prob = float(np.mean([float(e.prob) for e in entries]))
    curves = [e.survival for e in entries if e.survival is not None]
    return SlidePrediction(
        prob=prob,
        label=first.label,
        endpoint_type=first.endpoint_type,
        fold=fold,
        survival=_combine_curves(curves) if curves else None,
        patient_code=first.patient_code,
    )


def load_run_records(runs_dir: Path, field: str, inner: int | None) -> RunRecords | None:
    """Collects attention and predictions for the model(s) that scored each slide.

    Args:
        runs_dir: ``.../runs/<experiment>/<cohort>``.
        field: endpoint name, the ``<field>`` in ``<field>_cv_o{k}_i{j}``.
        inner: a single inner replicate, or None to average every replicate present.

    Returns:
        The merged records, or None if no fold has an ``attention.npz``.

    Raises:
        ValueError: two outer folds claim the same slide, or replicates of one outer fold disagree on
            a slide's tile coordinates.
    """
    pattern = f"{field}_cv_o*_i*" if inner is None else f"{field}_cv_o*_i{inner}"
    by_outer: dict[str, list[Path]] = {}
    for path in sorted(runs_dir.glob(f"{pattern}/{_ATTENTION}")):
        outer = path.parent.name.rsplit("_i", 1)[0]
        by_outer.setdefault(outer, []).append(path.parent)

    attention: dict[str, np.ndarray] = {}
    predictions: dict[str, SlidePrediction] = {}
    consistency: dict[str, float] = {}
    inners_by_outer: dict[str, list[int]] = {}
    owner: dict[str, str] = {}
    folds: list[str] = []
    mpp, tile_size = 0.0, 0

    for outer, dirs in sorted(by_outer.items()):
        inners_by_outer[outer] = sorted(int(d.name.rsplit("_i", 1)[1]) for d in dirs)
        folds.append(outer)
        per_model: dict[str, list[np.ndarray]] = {}
        per_model_pred: dict[str, list[SlidePrediction]] = {}
        for directory in dirs:
            with np.load(directory / _ATTENTION, allow_pickle=True) as handle:
                if "mpp" in handle.files:
                    mpp, tile_size = float(handle["mpp"]), int(np.asarray(handle["tile_size"]).ravel()[0])
                for key in handle.files:
                    if key.startswith("slide/"):
                        per_model.setdefault(key[len("slide/") :], []).append(handle[key])
            prediction_path = directory / _PREDICTIONS
            if prediction_path.exists():
                for slide_id, entry in _read_predictions(prediction_path, outer).items():
                    per_model_pred.setdefault(slide_id, []).append(entry)

        for slide_id, stacked in per_model.items():
            if slide_id in attention:
                raise ValueError(f"slide {slide_id!r} has attention in both {owner[slide_id]} and {outer}")
            base = stacked[0][:, :2]
            if any(not np.array_equal(a[:, :2], base) for a in stacked[1:]):
                raise ValueError(f"replicates of {outer} disagree on tile coordinates for slide {slide_id!r}")
            attention[slide_id] = stacked[0] if len(stacked) == 1 else _combine(stacked)
            owner[slide_id] = outer
            if len(stacked) > 1:
                consistency[slide_id] = _rank_agreement([_weights(a) for a in stacked])
        for slide_id, entries in per_model_pred.items():
            if slide_id in per_model:
                predictions[slide_id] = entries[0] if len(entries) == 1 else _average_predictions(entries, outer)

    if not attention:
        return None
    scoped = {k: v for k, v in predictions.items() if k in attention}
    summary, percentile = _regression_context(scoped)
    survival, risk_percentile = _survival_context(scoped)
    return RunRecords(
        attention=attention,
        predictions=scoped,
        mpp=mpp,
        tile_size=tile_size,
        folds=folds,
        aggregated=inner is None,
        inners_by_outer=inners_by_outer,
        consistency=consistency,
        outer_of=owner,
        regression=summary,
        error_percentile=percentile,
        survival=survival,
        risk_percentile=risk_percentile,
    )
