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
"""Split-plan generation, pure (no ahcore/dlup), so it is trivially testable.

Given a ``Cohort`` and per-patient raw column values, produce named split versions
(``SplitVersion``) with per-patient category assignments. ``build_db`` turns these into
ahcore ``SplitDefinitions`` + ``Split`` rows.

Stratification is resolved here and only here (contract seam #4): the fold-balance variable
defaults to the modeled target (class index / binarized value / binned regression target),
or an overriding ``stratify.column``. Continuous binning goes through the shared
``bin_continuous``, the same math target ``binarize`` uses (binarize is the k=2 case).
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from dlux.config.cohort import (
    BinarizeSpec,
    Cohort,
    ContractField,
    DiscretizeSpec,
    Objective,
    SplitCategory,
    Splits,
    SplitStrategy,
    StratifyMethod,
)

# Local aliases for the split-category vocabulary (from config.cohort.SplitCategory).
FIT = SplitCategory.FIT
VALIDATE = SplitCategory.VALIDATE
TEST = SplitCategory.TEST
PREDICT = SplitCategory.PREDICT

# The nested-CV split-version string, shared by build-db (generation) and aggregate (parsing).
# Keep format and parse in lockstep.
_CV_SPLIT_RE = re.compile(r"^(?P<field>.+)_cv_o(?P<outer>\d+)_i(?P<inner>\d+)$")


def format_cv_split(field: str, outer: int, inner: int) -> str:
    """Nested-CV split version: ``{field}_cv_o{outer}_i{inner}``."""
    return f"{field}_cv_o{outer}_i{inner}"


def parse_cv_split(version: str) -> Optional[tuple[str, int, int]]:
    """Inverse of :func:`format_cv_split`; ``None`` if not a nested-CV version."""
    m = _CV_SPLIT_RE.match(version)
    if m is None:
        return None
    return m.group("field"), int(m.group("outer")), int(m.group("inner"))


def format_all_predict() -> str:
    """Whole-cohort PREDICT split: ``all_predict``. Field-agnostic, unlabeled inference has no label to
    filter on, so one split covers every patient for every endpoint (unlike ``all_test_<field>``)."""
    return "all_predict"


def format_all_test(field: str) -> str:
    """Whole-cohort TEST split: ``all_test_{field}``. The other split-version family, what a cohort
    scored end-to-end gets instead of nested-CV folds. Named here rather than inline at each use so
    both families are constructed here."""
    return f"all_test_{field}"


@dataclass
class SplitVersion:
    version: str
    description: str
    assignments: list[tuple[str, str]]  # (patient_code, category)


# -- continuous binning (shared by target binarize/discretize + fold stratify) ----------
def continuous_cuts(
    values, *, method: StratifyMethod, k: Optional[int] = None, edges: Optional[list[float]] = None
) -> list[float]:
    """The interior cut points for a binning. Factored out so build-db can record them as fold-invariant
    provenance (binarize threshold / discretize edges) and the target transform can replay the exact same
    cut in every fold, not just the bin indices ``bin_continuous`` returns."""
    arr = np.asarray(values, dtype=float)
    if method == StratifyMethod.median:
        return [float(np.median(arr))]
    if method == StratifyMethod.quantile:
        assert k is not None and k >= 2
        return [float(c) for c in np.quantile(arr, np.linspace(0.0, 1.0, k + 1)[1:-1])]
    if method == StratifyMethod.threshold:
        assert edges is not None
        return [float(e) for e in edges]
    raise ValueError(f"unknown bin method {method!r}")  # pragma: no cover - enum-exhaustive


def bin_continuous(values, *, method: StratifyMethod, k: Optional[int] = None, edges: Optional[list[float]] = None):
    """Bin continuous values into integer strata. Same math whether called for a target ``binarize`` (k=2),
    a target ``discretize`` (k>=3), or a fold ``stratify`` (arbitrary k), the cut points are identical."""
    cuts = continuous_cuts(values, method=method, k=k, edges=edges)
    return [int(b) for b in np.digitize(np.asarray(values, dtype=float), cuts)]


def _binarize_strata(values: list[float], binarize: BinarizeSpec) -> list[int]:
    """Target binarize -> {0,1}: the k=2 case of bin_continuous."""
    if binarize.method == StratifyMethod.threshold:
        return bin_continuous(values, method=StratifyMethod.threshold, edges=[binarize.value])
    if binarize.method == StratifyMethod.quantile:
        return bin_continuous(values, method=StratifyMethod.quantile, k=2)
    return bin_continuous(values, method=StratifyMethod.median)


def _discretize_strata(values: list[float], discretize: DiscretizeSpec) -> list[int]:
    """Target discretize -> {0..k-1}: the k>=3 case of bin_continuous (quantile or explicit edges)."""
    if discretize.method == StratifyMethod.threshold:
        return bin_continuous(values, method=StratifyMethod.threshold, edges=discretize.edges)
    return bin_continuous(values, method=StratifyMethod.quantile, k=discretize.k)


def _parse_float(value) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _valid_patients_and_strata(
    field: ContractField, patient_labels: dict[str, dict[str, str]]
) -> tuple[list[str], list[int]]:
    """Patients with a usable target for this field, plus their fold-stratification label.

    Valid = target present + usable (in the categorical map, or parses to a finite float).
    Stratum = the ``stratify`` facet: default → modeled target (class / binarized value /
    binned regression target); override → a named column (binned if continuous, else its
    distinct raw values)."""
    t = field.source.transform
    col = field.source.column
    is_categorical = t is not None and t.categorical is not None

    codes: list[str] = []
    raw_targets: list = []  # raw str (categorical) | float (binarize / regression)
    for code, row in patient_labels.items():
        raw = row.get(col)
        if raw is None or raw == "":
            continue
        if is_categorical:
            if raw in t.categorical:
                codes.append(code)
                raw_targets.append(raw)
        else:
            parsed = _parse_float(raw)
            if parsed is not None:
                codes.append(code)
                raw_targets.append(parsed)

    if not codes:
        return [], []

    strat = field.stratify
    if strat.column is not None:  # override: stratify on a different column
        col_raw = [patient_labels[c].get(strat.column) for c in codes]
        if strat.method is None:  # categorical override column -> distinct values as strata
            order = {v: i for i, v in enumerate(sorted({str(v) for v in col_raw}))}
            strata = [order[str(v)] for v in col_raw]
        else:  # continuous override column -> bin it
            strata = bin_continuous(
                [_parse_float(v) for v in col_raw], method=strat.method, k=strat.k, edges=strat.edges
            )
    elif is_categorical:
        strata = [t.categorical[v] for v in raw_targets]
    elif t is not None and t.binarize is not None:
        strata = _binarize_strata(raw_targets, t.binarize)
    elif t is not None and t.discretize is not None:
        strata = _discretize_strata(raw_targets, t.discretize)
    else:  # regression: bin the target by the stratify method (median default)
        strata = bin_continuous(raw_targets, method=strat.method, k=strat.k, edges=strat.edges)
    return codes, strata


def draw_split(
    codes: list[str],
    strata: Optional[list[int]],
    *,
    ratios: tuple[float, ...],
    seed: int,
) -> dict[str, list[str]]:
    """One patient-level stratified FIT/VALIDATE(/TEST) draw -> ``{stage: [patient_code]}``.

    Deterministic in ``(codes, strata, ratios, seed)`` and pure. Shared by the ``simple`` split strategy
    and the resplit trials, so both build a partition the same way. Splits at patient level. Slide-level
    would leak a patient across stages.
    """
    if len(ratios) == 2:
        train, val = train_test_split(codes, test_size=ratios[1], random_state=seed, stratify=strata)
        return {"fit": list(train), "validate": list(val)}
    # 3-way: peel off TEST, then split the remainder into FIT/VALIDATE
    idx = list(range(len(codes)))
    pool_idx, test_idx = train_test_split(idx, test_size=ratios[2], random_state=seed, stratify=strata)
    pool = [codes[i] for i in pool_idx]
    pool_strata = [strata[i] for i in pool_idx] if strata is not None else None
    test = [codes[i] for i in test_idx]
    val_fraction = ratios[1] / (ratios[0] + ratios[1])
    train, val = train_test_split(pool, test_size=val_fraction, random_state=seed, stratify=pool_strata)
    return {"fit": list(train), "validate": list(val), "test": list(test)}


def split_hash(drawn: dict[str, list[str]]) -> str:
    """Short stable digest of a patient->stage assignment.

    Recorded per resplit run: the draw is reproducible from its seed only while ``draw_split`` itself
    is unchanged, so the digest is what turns silent drift into a visible mismatch."""
    canonical = ";".join(f"{stage}:{','.join(sorted(codes))}" for stage, codes in sorted(drawn.items()))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


_STAGE_CATEGORY = {"fit": FIT, "validate": VALIDATE, "test": TEST}


def _simple_split(field_name: str, codes: list[str], strata: list[int], splits) -> SplitVersion:
    drawn = draw_split(codes, strata, ratios=splits.ratios, seed=splits.random_state)
    assignments = [(c, _STAGE_CATEGORY[stage]) for stage, members in drawn.items() for c in members]
    return SplitVersion(field_name, f"simple split ({field_name})", assignments)


def _nested_cv_splits(field_name: str, codes: list[str], strata: Optional[list[int]], splits) -> list[SplitVersion]:
    """Nested KFold: {field}_cv_o{k}_i{j}, TEST_k shared across inner j. Stratified when ``strata`` is
    given (categorical/binned target); plain KFold when None (regression_vector, a vector target has
    no stratification axis)."""
    n_outer, n_inner, seed = splits.n_outer, splits.n_inner, splits.random_state
    stratified = strata is not None
    outer = (
        StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
        if stratified
        else KFold(n_splits=n_outer, shuffle=True, random_state=seed)
    )
    versions: list[SplitVersion] = []
    outer_iter = outer.split(codes, strata) if stratified else outer.split(codes)
    for k, (pool_idx, test_idx) in enumerate(outer_iter):
        pool = [codes[i] for i in pool_idx]
        pool_strata = [strata[i] for i in pool_idx] if stratified else None
        test = [codes[i] for i in test_idx]  # computed once per outer k -> shared-TEST invariant is automatic
        inner = (
            StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1 + k)
            if stratified
            else KFold(n_splits=n_inner, shuffle=True, random_state=seed + 1 + k)
        )
        inner_iter = inner.split(pool, pool_strata) if stratified else inner.split(pool)
        for j, (train_idx, val_idx) in enumerate(inner_iter):
            train = [pool[i] for i in train_idx]
            val = [pool[i] for i in val_idx]
            assignments = [(c, FIT) for c in train] + [(c, VALIDATE) for c in val] + [(c, TEST) for c in test]
            versions.append(
                SplitVersion(
                    format_cv_split(field_name, k, j),
                    f"nested CV outer {k}/{n_outer} inner {j}/{n_inner} ({field_name})",
                    assignments,
                )
            )
    return versions


def _survival_codes_and_strata(
    field: ContractField, patient_labels: dict[str, dict[str, str]]
) -> tuple[list[str], list[int]]:
    """Valid patients + event-stratified folds for a survival field. Valid = a parseable event
    indicator (0/1) and a positive follow-up time. The stratum is the event indicator, so each fold
    carries a comparable observed-event rate (censored patients are kept, not dropped)."""
    event_col, time_col = field.source.column, field.source.time_column
    codes: list[str] = []
    strata: list[int] = []
    for code, row in patient_labels.items():
        event, time = _parse_float(row.get(event_col)), _parse_float(row.get(time_col))
        if event is None or event not in (0.0, 1.0) or time is None or time <= 0.0:
            continue
        codes.append(code)
        strata.append(int(event))
    return codes, strata


def field_codes_and_strata(
    field: ContractField,
    field_name: str,
    patient_labels: dict[str, dict[str, str]],
    rnaseq_covered: Optional[dict[str, set[str]]],
    require_coverage: Optional[set[str]] = None,
) -> tuple[list[str], Optional[list[int]]]:
    """Valid patients + fold strata for a field. ``regression_vector`` reads coverage from the RNA
    matrix (``rnaseq_covered``) and is unstratified (strata None). ``survival`` reads two columns
    (event + time) and stratifies on the event indicator. All other objectives read the single
    patients-sheet column via ``_valid_patients_and_strata``. ``require_coverage`` (a study's required-
    modality intersection) further restricts every field's valid patients, so a fusion study's arms all
    train/score on one patient set."""
    if field.objective == Objective.regression_vector:
        covered = (rnaseq_covered or {}).get(field_name, set())
        codes, strata = [c for c in patient_labels if c in covered], None  # manifest-order, RNA-covered
    elif field.objective == Objective.survival:
        codes, strata = _survival_codes_and_strata(field, patient_labels)
    else:
        codes, strata = _valid_patients_and_strata(field, patient_labels)
    if require_coverage is not None:  # gate on required-modality coverage (drop patients missing it)
        keep = [i for i, c in enumerate(codes) if c in require_coverage]
        codes = [codes[i] for i in keep]
        strata = [strata[i] for i in keep] if strata is not None else None
    return codes, strata


def generate_splits(
    cohort: Cohort,
    patient_labels: dict[str, dict[str, str]],
    splits: Splits,
    rnaseq_covered: Optional[dict[str, set[str]]] = None,
    require_coverage: Optional[set[str]] = None,
) -> list[SplitVersion]:
    """Produce all split versions for ``cohort`` under the concrete ``splits`` plan.

    ``splits`` is the study-composed plan (``Study.splits_for``), role → strategy + params.
    ``patient_labels`` maps ``patient_code -> {column_name: raw_value}`` (the patients.csv columns; a
    field reads its ``source.column`` and any ``stratify.column``). ``rnaseq_covered`` maps a
    regression_vector field to the set of patients with an RNA-matrix row (build_db supplies it).
    ``require_coverage`` (the study's required-modality intersection, build_db-resolved) restricts every
    field's patients to those who have the required input modalities, the fusion fair-comparison gate.
    """
    all_codes = [c for c in patient_labels if require_coverage is None or c in require_coverage]

    # all_predict is unlabeled inference: there are no labels to filter on, so a single split
    # over every patient (with the required modalities) is the only sensible thing.
    if splits.strategy == SplitStrategy.all_predict:
        return [
            SplitVersion(
                format_all_predict(), "all patients (unlabeled inference; PREDICT)", [(c, PREDICT) for c in all_codes]
            )
        ]

    # all_test is external validation, labels exist, so filter to each field's valid-value patients
    # up front (one `all_test_<field>` split each), exactly like the internal per-endpoint strategies.
    # This keeps missing-label patients out of inference entirely, rather than scoring then discarding.
    if splits.strategy == SplitStrategy.all_test:
        versions: list[SplitVersion] = []
        for field_name, field in cohort.contract.items():
            codes, _strata = field_codes_and_strata(field, field_name, patient_labels, rnaseq_covered, require_coverage)
            if not codes:
                print(f"[splits] no valid patients for '{field_name}', skipping")
                continue
            versions.append(
                SplitVersion(
                    format_all_test(field_name),
                    f"valid '{field_name}' patients (external validation; TEST)",
                    [(c, TEST) for c in codes],
                )
            )
        return versions

    # Per-endpoint strategies (internal cohorts): split each field's valid-value patients.
    versions = []
    for field_name, field in cohort.contract.items():
        codes, strata = field_codes_and_strata(field, field_name, patient_labels, rnaseq_covered, require_coverage)
        if not codes:
            print(f"[splits] no valid patients for '{field_name}', skipping")
            continue
        if splits.strategy == SplitStrategy.simple:
            versions.append(_simple_split(field_name, codes, strata, splits))
        elif splits.strategy == SplitStrategy.nested_cv:
            versions.extend(_nested_cv_splits(field_name, codes, strata, splits))
    return versions


# -- predefined splits: import an assignment instead of generating one --------
# A build_db can write its split_versions from a supplied patient->fold CSV instead of drawing them, so
# two cohorts (or runs) share an identical split. build_db also exports the resulting split to the same
# CSV format on every build, so it round-trips and doubles as provenance. See docs/specs/SPLITS_SPEC.md.
_IMPORT_HEADER = ["patient_id", "split_version", "category"]
_VALID_CATEGORIES = {FIT, VALIDATE, TEST, PREDICT}
_STRATEGY_CATEGORIES = {
    SplitStrategy.nested_cv: {FIT, VALIDATE, TEST},
    SplitStrategy.simple: {FIT, VALIDATE, TEST},
    SplitStrategy.all_test: {TEST},
    SplitStrategy.all_predict: {PREDICT},
}


def export_splits_csv(versions: list[SplitVersion], path: Path) -> None:
    """Write ``versions`` as a portable patient->fold CSV (``patient_id,split_version,category``).

    build_db emits this for every build, generated or imported alike: it is the import format (so a split
    round-trips) and it doubles as provenance. Rows are sorted by (split_version, patient_id) for a stable,
    diffable file."""
    rows = sorted((code, v.version, category) for v in versions for code, category in v.assignments)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_IMPORT_HEADER)
        writer.writerows(rows)


def import_splits_csv(path: Path) -> list[SplitVersion]:
    """Read a patient->fold CSV (``export_splits_csv`` format) into ``SplitVersion``s, keeping file order
    of versions. Structural checks only (header, column count, category vocabulary). The cohort-aware
    checks are :func:`validate_imported_splits`."""
    grouped: dict[str, list[tuple[str, str]]] = {}
    order: list[str] = []
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header != _IMPORT_HEADER:
            raise ValueError(f"split file {path} header {header} != {_IMPORT_HEADER}")
        for lineno, rec in enumerate(reader, start=2):
            if not rec:
                continue
            if len(rec) != 3:
                raise ValueError(f"split file {path}:{lineno}: expected 3 columns, got {len(rec)}")
            code, version, category = (c.strip() for c in rec)
            if category not in _VALID_CATEGORIES:
                raise ValueError(
                    f"split file {path}:{lineno}: category {category!r} not in {sorted(_VALID_CATEGORIES)}"
                )
            if version not in grouped:
                grouped[version] = []
                order.append(version)
            grouped[version].append((code, category))
    return [SplitVersion(v, f"imported split ({v})", grouped[v]) for v in order]


def expected_version_names(active_fields: list[str], splits: Splits) -> set[str]:
    """The split_version names ``generate_splits`` would emit for ``active_fields`` (fields with at least
    one valid patient) under ``splits``. Checking an imported file against this catches a wrong
    ``n_outer``/``n_inner`` or a missing/extra field as a set mismatch."""
    if splits.strategy == SplitStrategy.all_predict:
        return {format_all_predict()}
    if splits.strategy == SplitStrategy.all_test:
        return {format_all_test(f) for f in active_fields}
    if splits.strategy == SplitStrategy.simple:
        return set(active_fields)
    return {
        format_cv_split(f, k, j) for f in active_fields for k in range(splits.n_outer) for j in range(splits.n_inner)
    }


def _version_field(version: str, active_fields: list[str]) -> Optional[str]:
    """The contract field a split_version belongs to (``None`` for the field-agnostic all_predict)."""
    parsed = parse_cv_split(version)
    if parsed is not None:
        return parsed[0]
    if version.startswith("all_test_"):
        return version[len("all_test_") :]
    if version == format_all_predict():
        return None
    if version in active_fields:  # simple: the version IS the field name
        return version
    return None


def valid_patients_by_field(
    cohort: Cohort,
    patient_labels: dict[str, dict[str, str]],
    rnaseq_covered: Optional[dict[str, set[str]]] = None,
    require_coverage: Optional[set[str]] = None,
) -> dict[str, set[str]]:
    """Per contract-field, the patients a generated split would cover (the field's valid-label subset,
    gated by ``require_coverage``). This is the yardstick an imported split is checked against, every
    such patient must appear in that field's imported versions (unless ``allow_uncovered``)."""
    out: dict[str, set[str]] = {}
    for field_name, field in cohort.contract.items():
        codes, _strata = field_codes_and_strata(field, field_name, patient_labels, rnaseq_covered, require_coverage)
        out[field_name] = set(codes)
    return out


def validate_imported_splits(
    versions: list[SplitVersion],
    splits: Splits,
    *,
    cohort_patients: set[str],
    valid_by_field: dict[str, set[str]],
    allow_uncovered: bool = False,
) -> list[SplitVersion]:
    """Cohort-aware validation of an imported split; returns the versions reconciled to the cohort.

    Raises ``ValueError`` on: a version-name set that differs from what the study expects; a version whose
    category set is wrong for the strategy; and, by default, either side of a cohort/import mismatch (a
    file patient absent from the cohort, or a labelled cohort patient absent from the file), which catches
    a truncated or wrong file. ``allow_uncovered`` is the matched cross-cohort case where the import and
    the cohort legitimately meet only at their intersection: it downgrades both mismatches to a reported
    drop, file-only patients are dropped from the returned versions, cohort-only patients from the folds."""
    active_fields = [f for f, codes in valid_by_field.items() if codes]

    expected = expected_version_names(active_fields, splits)
    got = {v.version for v in versions}
    if got != expected:
        missing, extra = sorted(expected - got), sorted(got - expected)
        raise ValueError(
            "imported splits do not match the study's expected fold set. "
            f"missing {missing[:8]}{'…' if len(missing) > 8 else ''}; "
            f"extra {extra[:8]}{'…' if len(extra) > 8 else ''}"
        )

    file_patients = {code for v in versions for code, _ in v.assignments}
    unknown = sorted(file_patients - cohort_patients)
    if unknown:
        if not allow_uncovered:
            raise ValueError(
                f"imported splits name {len(unknown)} patient(s) not in the cohort (e.g. {unknown[:8]}) — "
                "wrong file or a typo. Pass splits.allow_uncovered=true to drop them (matched-intersection case)."
            )
        print(
            f"[splits] import: {len(unknown)} file patient(s) not in the cohort → dropped "
            f"(matched-intersection case; e.g. {unknown[:8]})."
        )
        drop = set(unknown)
        versions = [
            SplitVersion(v.version, v.description, [(c, cat) for c, cat in v.assignments if c not in drop])
            for v in versions
        ]

    allowed = _STRATEGY_CATEGORIES[splits.strategy]
    for v in versions:
        cats = {category for _, category in v.assignments}
        if not cats <= allowed:
            raise ValueError(
                f"split_version {v.version!r} has categories {sorted(cats)} outside {sorted(allowed)} "
                f"for strategy {splits.strategy.value}."
            )
        if splits.strategy == SplitStrategy.nested_cv and allowed - cats:
            raise ValueError(
                f"split_version {v.version!r} is missing categories {sorted(allowed - cats)} "
                "(a nested-CV fold needs fit + validate + test)."
            )

    covered_by_field: dict[str, set[str]] = {f: set() for f in active_fields}
    for v in versions:
        field = _version_field(v.version, active_fields)
        if field is not None:
            covered_by_field.setdefault(field, set()).update(code for code, _ in v.assignments)
    for field in active_fields:
        uncovered = valid_by_field[field] - covered_by_field.get(field, set())
        if not uncovered:
            continue
        if not allow_uncovered:
            raise ValueError(
                f"{len(uncovered)} labelled '{field}' patient(s) are absent from the imported splits "
                f"(e.g. {sorted(uncovered)[:8]}). Pass splits.allow_uncovered=true to drop them "
                "(matched-intersection case), or fix the file."
            )
        print(
            f"[splits] import: {len(uncovered)} labelled '{field}' patient(s) not in the file → dropped "
            "from splits (allow_uncovered)."
        )
    return versions
