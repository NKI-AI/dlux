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
"""The on-disk layout of a study directory.

Everything for one experiment lives under ``studies/<study>/`` as predictable directory names:

    db/                              the built ahcore manifest DB(s) + build-time sidecars, per cohort:
      <cohort>.db                      the sqlite manifest
      <cohort>_contract_stats.json     fold-invariant target stats (binarize cut / target μ,σ)
      <cohort>_splits.csv              the resulting CV split, portable + re-importable
      <cohort>_splits_source.json      provenance, only when the split was imported
      <cohort>_analysis/               dataset summary.md + figures
    runs/          per-experiment training outputs (persistence.py owns the run-dir layout below this)
    results/       per-experiment aggregate / external reports
    comparisons/   compare outputs
    resplits/      resplit-comparison outputs

The DB path is read by six call sites (build_db, train, predict, evaluate_external, serve_slides, plus
tasks/build.py for the stats sidecar); constructing it here rather than inline keeps them in lockstep.
Sidecars derive from ``db_path().parent`` so they always sit beside the DB."""

from __future__ import annotations

from pathlib import Path

_STATS_SUFFIX = "_contract_stats.json"
_SPLITS_SUFFIX = "_splits.csv"
_SPLITS_SOURCE_SUFFIX = "_splits_source.json"
_ANALYSIS_SUFFIX = "_analysis"


def study_dir(studies_dir, study) -> Path:
    return Path(studies_dir) / str(study)


def db_dir(studies_dir, study) -> Path:
    """The per-study directory that holds every cohort's built DB + build-time sidecars."""
    return study_dir(studies_dir, study) / "db"


def db_path(studies_dir, study, cohort) -> Path:
    return db_dir(studies_dir, study) / f"{cohort}.db"


def db_uri(studies_dir, study, cohort) -> str:
    return f"sqlite:///{db_path(studies_dir, study, cohort)}"


def contract_stats_path(studies_dir, study, cohort) -> Path:
    return db_dir(studies_dir, study) / f"{cohort}{_STATS_SUFFIX}"


def splits_csv_path(studies_dir, study, cohort) -> Path:
    return db_dir(studies_dir, study) / f"{cohort}{_SPLITS_SUFFIX}"


def splits_source_path(studies_dir, study, cohort) -> Path:
    return db_dir(studies_dir, study) / f"{cohort}{_SPLITS_SOURCE_SUFFIX}"


def analysis_dir(studies_dir, study, cohort) -> Path:
    return db_dir(studies_dir, study) / f"{cohort}{_ANALYSIS_SUFFIX}"
