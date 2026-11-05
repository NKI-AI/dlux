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
"""The on-disk layout of a study directory. Every path below ``studies/<study>/`` is built here, and nowhere
else, so the code that writes a file and the code that reads it cannot disagree.

    db/<cohort>/                       one cohort's build, written as a whole by build_db:
      manifest.db                        the sqlite manifest
      contract_stats.json                fold-invariant target stats (binarize cut / discretize edges)
      splits.csv                         the resulting CV split, portable + re-importable
      splits_source.json                 provenance, only when the split was imported
      analysis/                          dataset summary.md + figures
    runs/<experiment>/<dev_cohort>/<field>_cv_o{k}_i{j}/
                                       one trained fold: checkpoint, metadata, its own test predictions
    results/<experiment>/<cohort>/     the experiment's results on one cohort, the same shape for every
                                       role (aggregate on the development cohort, evaluate_external on a
                                       validation cohort, predict on an unlabelled one):
      summary.md
      per_patient_<field>.npz            the record every reader of these results reads
      per_patient_<field>.csv            the same for people (none for an expression endpoint)
      per_fold_metrics.csv               labelled cohorts, one row per outer fold
      figures/
    comparisons/<comparison>/          both stages of one comparison:
      nested_cv/                         compare, on the arms' nested-CV results
      resplit_<resplit>/<field>/         compare_resplits, on a resplit sweep's trials
    resplits/<resplit>/<cohort>/       the trials of a resplit sweep:
      rows/                              one JSON row per trial (arm, seed, replicate)
      predictions/                       each trial's test predictions
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def study_dir(studies_dir, study) -> Path:
    return Path(studies_dir) / str(study)


def db_dir(studies_dir, study, cohort) -> Path:
    """One cohort's build: the manifest DB and the files build_db writes with it."""
    return study_dir(studies_dir, study) / "db" / str(cohort)


def manifest_db(db: Path) -> Path:
    return Path(db) / "manifest.db"


def contract_stats_json(db: Path) -> Path:
    return Path(db) / "contract_stats.json"


def splits_csv(db: Path) -> Path:
    return Path(db) / "splits.csv"


def splits_source_json(db: Path) -> Path:
    return Path(db) / "splits_source.json"


def analysis_dir(db: Path) -> Path:
    return Path(db) / "analysis"


def db_path(studies_dir, study, cohort) -> Path:
    return manifest_db(db_dir(studies_dir, study, cohort))


def db_uri(studies_dir, study, cohort) -> str:
    return f"sqlite:///{db_path(studies_dir, study, cohort)}"


def contract_stats_path(studies_dir, study, cohort) -> Path:
    return contract_stats_json(db_dir(studies_dir, study, cohort))


def runs_dir(studies_dir, study, experiment, cohort) -> Path:
    """An experiment's trained folds on its development cohort."""
    return study_dir(studies_dir, study) / "runs" / str(experiment) / str(cohort)


def run_dir(studies_dir, study, experiment, cohort, split_version) -> Path:
    """One trained fold."""
    return runs_dir(studies_dir, study, experiment, cohort) / str(split_version)


def results_dir(studies_dir, study, experiment, cohort) -> Path:
    """The experiment's results on one cohort."""
    return study_dir(studies_dir, study) / "results" / str(experiment) / str(cohort)


def comparison_dir(studies_dir, study, comparison) -> Path:
    """Both stages of one comparison."""
    return study_dir(studies_dir, study) / "comparisons" / str(comparison)


def nested_cv_comparison_dir(studies_dir, study, comparison) -> Path:
    return comparison_dir(studies_dir, study, comparison) / "nested_cv"


def resplit_comparison_dir(studies_dir, study, comparison, resplit, field) -> Path:
    return comparison_dir(studies_dir, study, comparison) / f"resplit_{resplit}" / str(field)


def resplit_dir(studies_dir, study, resplit, cohort) -> Path:
    return study_dir(studies_dir, study) / "resplits" / str(resplit) / str(cohort)


def resplit_rows_dir(studies_dir, study, resplit, cohort) -> Path:
    return resplit_dir(studies_dir, study, resplit, cohort) / "rows"


def resplit_predictions_dir(studies_dir, study, resplit, cohort) -> Path:
    return resplit_dir(studies_dir, study, resplit, cohort) / "predictions"


def per_patient_npz(results: Path, field) -> Path:
    return Path(results) / f"per_patient_{field}.npz"


def per_patient_csv(results: Path, field) -> Path:
    return Path(results) / f"per_patient_{field}.csv"


def per_gene_csv(results: Path, field) -> Path:
    """An expression endpoint's per-gene Pearson, for people (an expression endpoint has no per-patient CSV)."""
    return Path(results) / f"per_gene_pearson_{field}.csv"


def summary_md(results: Path) -> Path:
    return Path(results) / "summary.md"


def per_fold_metrics_csv(results: Path) -> Path:
    return Path(results) / "per_fold_metrics.csv"


@contextmanager
def replaced_dir(final: Path) -> Iterator[Path]:
    """Write a whole output folder before it replaces the old one. Yields ``<final>.partial``. On success the
    old folder is removed and the new one takes its name, so no file of an earlier run survives next to the
    new ones. On an error the old folder stays as it was and the partial one is removed."""
    final = Path(final)
    partial = final.with_name(final.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        yield partial
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    if final.exists():
        shutil.rmtree(final)
    partial.rename(final)
