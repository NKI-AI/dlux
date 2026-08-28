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
"""Driver tests for dlux.eval.aggregate: fold discovery, completeness/grid guards, the shared-TEST
invariant, the scalar outer-fold ensemble + rollup, and the report skeleton — exercised through the
binary endpoint as the reference scalar vehicle. The per-objective aggregation + reporting tests live
in eval_{regression,multiclass,survival,expression}_test.py (mirroring dlux/eval/<objective>.py).

Synthetic per-fold artifacts are written with the REAL NPZ writer (schema fidelity) into a
temp runs tree, each gated by a metadata.json sentinel — exactly what a sweep leaves behind.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from dlux.eval.aggregate import aggregate_experiment, write_reports
from dlux.eval.predictions import write_predictions_npz


def _declare(fields=("cancer_type",), n_outer=2, n_inner=2) -> dict:
    """What the study declares — aggregate now takes the expected shape from config, not from disk."""
    return {"expected_fields": list(fields), "n_outer": n_outer, "n_inner": n_inner}


def _write_fold(
    runs_root,
    field,
    outer,
    inner,
    patient_probs,
    patient_labels,
    *,
    complete=True,
    endpoint_type="binary",
    grid=None,
    coverage=None,
):
    """Write one fold dir: {field}_cv_o{outer}_i{inner}/test_predictions.npz (+ metadata sentinel).

    One slide per patient. ``patient_probs`` maps patient_code -> positive-class prob. ``grid``
    (n_outer, n_inner) stamps the expected CV shape into metadata for the strict check. ``coverage``
    stamps this fold's declared-vs-scored TEST patients; omitting it is what a pre-record run looks like.
    """
    d = runs_root / f"{field}_cv_o{outer}_i{inner}"
    d.mkdir(parents=True)
    pcs = list(patient_probs)
    probs = np.asarray([patient_probs[p] for p in pcs], dtype=np.float64)
    labels = np.asarray([patient_labels[p] for p in pcs], dtype=np.int64)
    preds = {
        "slide_ids": [f"{p}_s0" for p in pcs],
        "patient_codes": pcs,
        "logits": probs,
        "probs": probs,
        "labels": labels,
        "endpoint_type": endpoint_type,
        "num_classes": 1,
        "target_key": f"patient.{field}",
    }
    write_predictions_npz(preds, d / "test_predictions.npz")
    if complete:
        meta = {"n_outer": grid[0], "n_inner": grid[1]} if grid else {}
        if coverage is not None:
            meta["test_coverage"] = coverage
        (d / "metadata.json").write_text(json.dumps(meta))


def _two_fold_sweep(runs_root):
    """outer0 -> AUROC 1.0, outer1 -> AUROC 0.0 (after inner ensembling). field='cancer_type', grid 2x2."""
    labels = {"p0": 1, "p1": 0, "p2": 1, "p3": 0}
    # outer 0: both inner replicates rank correctly -> ensemble ranks correctly -> AUROC 1.0
    _write_fold(runs_root, "cancer_type", 0, 0, {"p0": 0.9, "p1": 0.1}, labels, grid=(2, 2))
    _write_fold(runs_root, "cancer_type", 0, 1, {"p0": 0.7, "p1": 0.3}, labels, grid=(2, 2))
    # outer 1: both inner replicates rank INVERSELY -> ensemble inverted -> AUROC 0.0
    _write_fold(runs_root, "cancer_type", 1, 0, {"p2": 0.4, "p3": 0.6}, labels, grid=(2, 2))
    _write_fold(runs_root, "cancer_type", 1, 1, {"p2": 0.45, "p3": 0.55}, labels, grid=(2, 2))


def test_outer_ensemble_and_metrics(tmp_path):
    _two_fold_sweep(tmp_path)
    (results,) = aggregate_experiment(tmp_path, **_declare())  # single field

    assert results.field == "cancer_type" and results.endpoint_type == "binary"
    # per-outer AUROC: o0 perfect ranking (1.0), o1 inverted (0.0)
    assert {m.outer_fold: round(m.metrics["auroc"], 3) for m in results.per_outer} == {0: 1.0, 1: 0.0}
    assert results.mean["auroc"] == pytest.approx(0.5)
    assert results.std["auroc"] == pytest.approx(0.5)
    # pooled OOF over p0..p3 with ensembled probs [0.8, 0.2, 0.425, 0.575], labels [1,0,1,0] -> 3/4 pairs
    assert results.pooled["auroc"] == pytest.approx(0.75)
    assert (results.pooled_n, results.pooled_positive) == (4, 2)
    assert results.inner_coverage == {0: 2, 1: 2}
    # thresholded @0.5: o0 preds [1,0] on labels [1,0] -> acc 1.0; o1 preds [0,1] -> acc 0.0
    assert {m.outer_fold: m.metrics["accuracy"] for m in results.per_outer} == {0: 1.0, 1: 0.0}
    assert results.mean["accuracy"] == pytest.approx(0.5)
    # o0 is perfect -> every metric = 1.0
    o0 = next(m for m in results.per_outer if m.outer_fold == 0)
    for k in ("sensitivity", "specificity", "f1", "balanced_accuracy", "precision", "ap", "npv", "mcc", "kappa"):
        assert o0.metrics[k] == pytest.approx(1.0)
    # ensemble = mean of the two inner replicate probs
    p0 = next(p for p in results.patient_predictions if p.patient_code == "p0")
    assert p0.ensemble_prob == pytest.approx(0.8) and p0.n_replicates == 2


def test_binary_metrics_degenerate_single_class():
    from dlux.eval.binary import binary_metrics

    m = binary_metrics(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.6]))  # all positive
    assert np.isnan(m["auroc"]) and np.isnan(m["specificity"])  # undefined without a negative
    assert m["sensitivity"] == pytest.approx(1.0)  # all predicted positive @0.5


def test_slide_rollup_mean(tmp_path):
    """Two slides for one patient -> patient prob is their mean, before inner ensembling."""
    d = tmp_path / "cancer_type_cv_o0_i0"
    d.mkdir()
    preds = {
        "slide_ids": ["p0_s0", "p0_s1", "p1_s0"],
        "patient_codes": ["p0", "p0", "p1"],
        "logits": np.array([0.8, 0.6, 0.2]),
        "probs": np.array([0.8, 0.6, 0.2]),  # p0 two slides -> mean 0.7
        "labels": np.array([1, 1, 0]),
        "endpoint_type": "binary",
        "num_classes": 1,
        "target_key": "patient.cancer_type",
    }
    write_predictions_npz(preds, d / "test_predictions.npz")
    (d / "metadata.json").write_text(json.dumps({"n_outer": 1, "n_inner": 1}))

    (results,) = aggregate_experiment(tmp_path, **_declare(n_outer=1, n_inner=1))
    p0 = next(p for p in results.patient_predictions if p.patient_code == "p0")
    assert p0.ensemble_prob == pytest.approx(0.7)


def test_incomplete_folds_skipped(tmp_path):
    labels = {"p0": 1, "p1": 0}
    _write_fold(tmp_path, "cancer_type", 0, 0, {"p0": 0.9, "p1": 0.1}, labels, grid=(1, 2))
    _write_fold(tmp_path, "cancer_type", 0, 1, {"p0": 0.8, "p1": 0.2}, labels, complete=False)  # no sentinel
    # i1 has no sentinel -> not discovered; strict would flag the gap, so aggregate the partial sweep
    (results,) = aggregate_experiment(tmp_path, **_declare(n_outer=1), strict=False)
    assert results.inner_coverage == {0: 1}  # only the completed replicate counted


def test_shared_test_invariant_violation_raises(tmp_path):
    labels = {"p0": 1, "p1": 0, "p9": 0}
    _write_fold(tmp_path, "cancer_type", 0, 0, {"p0": 0.9, "p1": 0.1}, labels, grid=(1, 2))
    _write_fold(tmp_path, "cancer_type", 0, 1, {"p0": 0.8, "p9": 0.2}, labels, grid=(1, 2))  # different patient set!
    with pytest.raises(ValueError, match="shared-TEST invariant"):
        aggregate_experiment(tmp_path, **_declare(n_outer=1))


def test_no_folds_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        aggregate_experiment(tmp_path, **_declare())


def test_strict_raises_on_missing_fold_with_grid(tmp_path):
    """A 2x2 grid stamped in metadata, but o1_i1 missing -> strict refuses; strict=False proceeds."""
    labels = {"p0": 1, "p1": 0, "p2": 1, "p3": 0}
    for o, i in [(0, 0), (0, 1), (1, 0)]:  # missing (1, 1)
        pats = {"p0": 0.9, "p1": 0.1} if o == 0 else {"p2": 0.8, "p3": 0.2}
        _write_fold(tmp_path, "cancer_type", o, i, pats, labels, grid=(2, 2))
    with pytest.raises(ValueError, match="INCOMPLETE"):
        aggregate_experiment(tmp_path, **_declare())  # strict=True default
    # strict=False aggregates the partial sweep
    (results,) = aggregate_experiment(tmp_path, **_declare(), strict=False)
    assert results.inner_coverage == {0: 2, 1: 1}


def test_missing_grid_stamp_raises(tmp_path):
    """A fold lacking the CV grid violates the contract -> refused (no legacy fallback), even non-strict."""
    labels = {"p0": 1, "p1": 0}
    _write_fold(tmp_path, "cancer_type", 0, 0, {"p0": 0.9, "p1": 0.1}, labels)  # grid=None -> metadata "{}"
    with pytest.raises(ValueError, match="missing the CV grid"):
        aggregate_experiment(tmp_path, **_declare())
    with pytest.raises(ValueError, match="missing the CV grid"):
        aggregate_experiment(
            tmp_path, **_declare(), strict=False
        )  # the grid is a contract requirement, not a strictness knob


def _write_vector_fold(runs_root, field, outer, inner, patients, *, n_genes=4, grid=(2, 2), seed=0):
    """One expression fold: (N, G) predictions + labels, the shape the random-baseline path aggregates."""
    d = runs_root / f"{field}_cv_o{outer}_i{inner}"
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed + 17 * outer + inner)
    labels = rng.normal(size=(len(patients), n_genes))
    write_predictions_npz(
        {
            "slide_ids": [f"{p}_s0" for p in patients],
            "patient_codes": list(patients),
            "logits": labels + rng.normal(scale=0.3, size=labels.shape),
            "probs": labels + rng.normal(scale=0.3, size=labels.shape),
            "labels": labels,
            "endpoint_type": "regression_vector",
            "num_classes": n_genes,
            "target_key": f"patient.{field}",
        },
        d / "test_predictions.npz",
    )
    (d / "metadata.json").write_text(json.dumps({"n_outer": grid[0], "n_inner": grid[1]}))


def _expression_sweep(root, *, skip=()):
    """A 2x2 expression sweep under ``root``; ``skip`` omits (outer, inner) folds."""
    outer_patients = {0: [f"p{i}" for i in range(6)], 1: [f"q{i}" for i in range(6)]}
    for o in (0, 1):
        for i in (0, 1):
            if (o, i) in skip:
                continue
            _write_vector_fold(root, "expression", o, i, outer_patients[o])


def test_strict_raises_on_incomplete_random_baseline(tmp_path):
    """The null feeds a REPORTED statistic, so it is held to the declared grid like the trained arm.

    Checking only "does the random dir have any folds at all" let a short null through silently and
    changed the trained-vs-random gene count without saying so."""
    trained, random_dir = tmp_path / "trained", tmp_path / "random"
    _expression_sweep(trained)
    _expression_sweep(random_dir, skip={(0, 1)})

    declared = _declare(fields=("expression",))
    with pytest.raises(ValueError, match=r"random baseline.*INCOMPLETE"):
        aggregate_experiment(trained, **declared, random_experiment_dir=random_dir)

    # strict=False still reports, having said what is missing — same escape hatch as the trained arm.
    (res,) = aggregate_experiment(trained, **declared, random_experiment_dir=random_dir, strict=False)
    assert res.n_outer_folds == 2


def test_complete_random_baseline_passes(tmp_path):
    trained, random_dir = tmp_path / "trained", tmp_path / "random"
    _expression_sweep(trained)
    _expression_sweep(random_dir)
    (res,) = aggregate_experiment(trained, **_declare(fields=("expression",)), random_experiment_dir=random_dir)
    assert res.n_outer_folds == 2


def test_strict_passes_on_complete_grid(tmp_path):
    labels = {"p0": 1, "p1": 0, "p2": 1, "p3": 0}
    for o, i in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        pats = {"p0": 0.9, "p1": 0.1} if o == 0 else {"p2": 0.8, "p3": 0.2}
        _write_fold(tmp_path, "cancer_type", o, i, pats, labels, grid=(2, 2))
    (results,) = aggregate_experiment(tmp_path, **_declare())  # strict passes -> no raise
    assert results.n_outer_folds == 2


def test_a_declared_endpoint_with_zero_folds_is_an_error(tmp_path):
    """The hole that motivated S5a. An endpoint whose every fold failed is simply ABSENT from discovery,
    so a completeness check driven by the data cannot see it — the report came out looking clean. Only
    the study's declaration knows it should have been there."""
    labels = {"p0": 1, "p1": 0}
    for o in (0, 1):
        for i in (0, 1):
            pats = {"p0": 0.9, "p1": 0.1} if o == 0 else {"p2": 0.4, "p3": 0.6}
            _write_fold(tmp_path, "cancer_type", o, i, pats, {**labels, "p2": 1, "p3": 0}, grid=(2, 2))

    # `grade` was declared and never produced a single fold.
    with pytest.raises(ValueError, match=r"declared endpoint\(s\) \['grade'\] have NO completed folds"):
        aggregate_experiment(tmp_path, **_declare(fields=("cancer_type", "grade")))

    # strict=false still reports what DID complete, but says what is missing.
    results = aggregate_experiment(tmp_path, **_declare(fields=("cancer_type", "grade")), strict=False)
    assert [r.field for r in results] == ["cancer_type"]


def test_an_undeclared_endpoint_on_disk_is_not_aggregated(tmp_path):
    """Discovery says what exists; the declaration says what counts. A stale endpoint left in the runs
    dir must not silently appear in the report."""
    labels = {"p0": 1, "p1": 0, "p2": 1, "p3": 0}
    for field in ("cancer_type", "leftover"):
        for o in (0, 1):
            for i in (0, 1):
                pats = {"p0": 0.9, "p1": 0.1} if o == 0 else {"p2": 0.4, "p3": 0.6}
                _write_fold(tmp_path, field, o, i, pats, labels, grid=(2, 2))
    results = aggregate_experiment(tmp_path, **_declare())
    assert [r.field for r in results] == ["cancer_type"]


def test_a_grid_mismatch_between_sweep_and_study_is_an_error(tmp_path):
    """The folds stamp the grid they ran on; the study declares the grid it wants. Disagreement means the
    config changed mid-sweep, which nothing caught before."""
    labels = {"p0": 1, "p1": 0}
    _write_fold(tmp_path, "cancer_type", 0, 0, {"p0": 0.9, "p1": 0.1}, labels, grid=(1, 1))
    with pytest.raises(ValueError, match="but the study declares 2x2"):
        aggregate_experiment(tmp_path, **_declare())


def test_write_reports_produces_files(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _two_fold_sweep(runs)
    results = aggregate_experiment(runs, **_declare())

    out = write_reports(results, "exp_v1", tmp_path / "results")
    assert (out / "per_patient_ensemble.csv").exists()
    assert (out / "per_fold_metrics.csv").exists()
    assert (out / "summary.md").exists()
    assert (out / "figures" / "roc_cancer_type.png").exists()
    assert (out / "figures" / "auroc_by_fold.png").exists()

    with (out / "per_fold_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # two outer folds
    assert {"auroc", "accuracy", "sensitivity", "f1"} <= set(rows[0].keys())  # richer metric columns
    md = (out / "summary.md").read_text()
    assert "| fold | n | pos |" in md  # markdown table header
    assert "0.500±0.500" in md  # mean±std cell (auroc & accuracy both 0.5±0.5)


# -- TEST coverage (declared vs scored) --------------------------------------
def _sweep_with_coverage(runs_root, o0_cov, o1_cov):
    """The 2x2 sweep, with each outer fold's coverage stamped on both of its inner replicates."""
    labels = {"p0": 1, "p1": 0, "p2": 1, "p3": 0}
    for inner in (0, 1):
        _write_fold(runs_root, "cancer_type", 0, inner, {"p0": 0.9, "p1": 0.1}, labels, grid=(2, 2), coverage=o0_cov)
        _write_fold(runs_root, "cancer_type", 1, inner, {"p2": 0.4, "p3": 0.6}, labels, grid=(2, 2), coverage=o1_cov)


def test_coverage_sums_over_outer_folds_not_over_replicates(tmp_path):
    """Inner replicates of one outer fold score the SAME patients, so counting each would multiply the
    study's N by n_inner. Outer folds partition the cohort, so those do sum."""
    _sweep_with_coverage(
        tmp_path,
        {"declared": 3, "scored": 2, "lost": ["lost_a"]},
        {"declared": 3, "scored": 2, "lost": ["lost_b"]},
    )
    (res,) = aggregate_experiment(tmp_path, **_declare())
    assert res.coverage == {"declared": 6, "scored": 4, "lost": ["lost_a", "lost_b"]}


def test_coverage_is_unknown_when_any_fold_predates_the_record(tmp_path):
    """A partial union would understate the loss, and an understated loss reads as full coverage."""
    _sweep_with_coverage(tmp_path, {"declared": 3, "scored": 2, "lost": ["lost_a"]}, None)
    (res,) = aggregate_experiment(tmp_path, **_declare())
    assert res.coverage is None


def test_summary_states_the_coverage_gap(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _sweep_with_coverage(
        runs, {"declared": 3, "scored": 2, "lost": ["lost_a"]}, {"declared": 2, "scored": 2, "lost": []}
    )
    out = write_reports(aggregate_experiment(runs, **_declare()), "exp_v1", tmp_path / "results")
    md = (out / "summary.md").read_text()
    assert "4/5 declared TEST patients scored" in md
    assert "`lost_a`" in md


def test_summary_is_silent_when_coverage_is_total(tmp_path):
    """Full coverage is the normal case; a line on every report would train the reader to skip it."""
    runs = tmp_path / "runs"
    runs.mkdir()
    _sweep_with_coverage(runs, {"declared": 2, "scored": 2, "lost": []}, {"declared": 2, "scored": 2, "lost": []})
    out = write_reports(aggregate_experiment(runs, **_declare()), "exp_v1", tmp_path / "results")
    assert "declared TEST patients scored" not in (out / "summary.md").read_text()


def test_summary_flags_unknown_coverage(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    _two_fold_sweep(runs)  # no coverage stamped anywhere
    out = write_reports(aggregate_experiment(runs, **_declare()), "exp_v1", tmp_path / "results")
    assert "Coverage unknown" in (out / "summary.md").read_text()
