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
"""build_db integration tests for predefined CV splits (import/export).

Exercises the real split step through actual sqlite (probe=none, so no WSIs are opened): a generated
split is exported, re-imported into a fresh DB, and the DB-level split_versions must come back identical.
Also checks the strict/allow_uncovered branch at build_db level. The pure export/import/validate helpers
are unit-tested separately in splits_test.py."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ahcore.manifest import Split, open_db

from dlux.config.cohort import Cohort, ContractField, Splits, SplitStrategy, Storage
from dlux.data import layout
from dlux.data.build_db import build_db
from dlux.data.errors import BuildDbError

_BINARY = {"cancer_type": ContractField.model_validate({"type": "binary", "map": {"BRCA": 0, "COAD": 1}})}


def _codes(n_per_class: int) -> list[str]:
    return [f"brca{i}" for i in range(n_per_class)] + [f"coad{i}" for i in range(n_per_class)]


def _write_sheets(dirpath, codes: list[str], unlabelled: frozenset[str] = frozenset()) -> tuple:
    patients_csv, slides_csv = dirpath / "patients.csv", dirpath / "slides.csv"
    with open(patients_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient_id", "cancer_type"])
        for code in codes:
            label = "" if code in unlabelled else "BRCA" if code.startswith("brca") else "COAD"
            writer.writerow([code, label])
    with open(slides_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["patient_id", "image_path"])
        for code in codes:
            writer.writerow([code, f"{code}.svs"])
    return patients_csv, slides_csv


def _build(tmp_path, name: str, codes: list[str], unlabelled: frozenset[str] = frozenset(), **build_kw) -> str:
    cohort_dir = tmp_path / name
    cohort_dir.mkdir(exist_ok=True)
    patients_csv, slides_csv = _write_sheets(cohort_dir, codes, unlabelled)
    build_db(
        Cohort(name=name, storage=Storage(image_dir=str(cohort_dir)), contract=_BINARY),
        Splits(strategy=SplitStrategy.nested_cv, n_outer=2, n_inner=2),
        _db(tmp_path, name),
        patients_csv=patients_csv,
        slides_csv=slides_csv,
        probe="none",
        analyze=False,
        **build_kw,
    )
    return f"sqlite:///{layout.manifest_db(_db(tmp_path, name))}"


def _db(tmp_path, name: str) -> Path:
    return tmp_path / "db" / name


def _db_splits(uri: str) -> dict[str, set[tuple[str, str]]]:
    """{split_version -> {(patient_code, category)}} read back from the persisted DB."""
    out: dict[str, set[tuple[str, str]]] = {}
    with open_db(uri, ensure_exists=False) as session:
        for row in session.query(Split).all():
            category = getattr(row.category, "value", row.category)
            out.setdefault(row.split_definition.version, set()).add((str(row.patient.patient_code), str(category)))
    return out


def test_import_reproduces_generated_split(tmp_path):
    codes = _codes(3)
    uri_gen = _build(tmp_path, "gen", codes)
    emitted = layout.splits_csv(_db(tmp_path, "gen"))  # build_db always exports the split with the DB
    assert emitted.is_file()
    generated = _db_splits(uri_gen)
    assert len(generated) == 4  # 2 outer x 2 inner

    uri_imp = _build(tmp_path, "imp", codes, import_splits=emitted)
    assert _db_splits(uri_imp) == generated  # imported DB carries byte-identical folds


def test_import_strict_errors_on_uncovered_then_allows(tmp_path):
    codes = _codes(3)
    _build(tmp_path, "gen2", codes)
    emitted = layout.splits_csv(_db(tmp_path, "gen2"))

    bigger = codes + ["brca9"]  # an extra labelled patient the import file does not cover
    with pytest.raises(BuildDbError, match="absent from the imported splits"):
        _build(tmp_path, "sub_strict", bigger, import_splits=emitted)

    uri_ok = _build(tmp_path, "sub_ok", bigger, import_splits=emitted, import_allow_uncovered=True)
    placed = {code for members in _db_splits(uri_ok).values() for code, _ in members}
    assert "brca9" not in placed and placed == set(codes)


def test_import_strict_errors_on_a_patient_without_a_valid_label_then_drops_it(tmp_path):
    codes = _codes(8)
    _build(tmp_path, "gen3", codes)
    emitted = layout.splits_csv(_db(tmp_path, "gen3"))

    with pytest.raises(BuildDbError, match="no valid 'cancer_type' label"):
        _build(tmp_path, "nolabel_strict", codes, unlabelled=frozenset({"brca0"}), import_splits=emitted)

    uri_ok = _build(
        tmp_path,
        "nolabel_ok",
        codes,
        unlabelled=frozenset({"brca0"}),
        import_splits=emitted,
        import_allow_uncovered=True,
    )
    placed = {code for members in _db_splits(uri_ok).values() for code, _ in members}
    assert placed == set(codes) - {"brca0"}


def test_a_failed_rebuild_leaves_the_previous_build(tmp_path):
    codes = _codes(3)
    uri = _build(tmp_path, "keep", codes)
    before = _db_splits(uri)
    bad = tmp_path / "bad_splits.csv"
    bad.write_text("not,a,split,file\n")

    with pytest.raises(BuildDbError):
        _build(tmp_path, "keep", codes, import_splits=bad, overwrite=True)

    assert _db_splits(uri) == before
    assert layout.splits_csv(_db(tmp_path, "keep")).is_file()
    assert not _db(tmp_path, "keep").with_name("keep.partial").exists()


def test_a_failure_after_the_db_is_written_leaves_the_previous_build(tmp_path, monkeypatch):
    uri = _build(tmp_path, "late", _codes(4))
    before = _db_splits(uri)

    def fail(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("dlux.data.build_db.export_splits_csv", fail)
    with pytest.raises(OSError, match="disk full"):
        _build(tmp_path, "late", _codes(3), overwrite=True)

    assert _db_splits(uri) == before
    assert layout.splits_csv(_db(tmp_path, "late")).is_file()
    assert layout.contract_stats_json(_db(tmp_path, "late")).is_file()
    assert not _db(tmp_path, "late").with_name("late.partial").exists()


def test_a_drawn_rebuild_drops_the_import_record_of_an_imported_build(tmp_path):
    codes = _codes(3)
    _build(tmp_path, "src", codes)
    _build(tmp_path, "rebuilt", codes, import_splits=layout.splits_csv(_db(tmp_path, "src")))
    assert layout.splits_source_json(_db(tmp_path, "rebuilt")).is_file()

    _build(tmp_path, "rebuilt", codes, overwrite=True)

    assert not layout.splits_source_json(_db(tmp_path, "rebuilt")).exists()
    assert layout.splits_csv(_db(tmp_path, "rebuilt")).is_file()
