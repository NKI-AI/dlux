# Predefined CV splits — import a split instead of drawing one

> **How `build_db` writes a cohort's CV `split_versions` from a supplied patient→fold CSV instead of drawing them — so two cohorts (or two runs) share an identical split — and how every build exports the split it produced for reuse and provenance.**

Source of truth: the split step in `dlux/data/build_db.py` (the generate-or-import branch and the always-emit of the split CSV) and `dlux/data/splits.py` (`import_splits_csv`, `validate_imported_splits`, `export_splits_csv`, `expected_version_names`). The knobs live in `config/build_db.yaml` under `splits:`. Only the _source_ of the assignment changes — `generate_splits` and its stratification are untouched and stay the default.

---

## Part 1 — what it does

### 1.1 Motivation

- **Matched cross-cohort splits (the driving case).** The same patients scanned two ways (two scanners, or a rescan of the same glass) must share fold assignments, so a comparison of the two cohorts' out-of-fold results is fair and leakage-free: same patients, same folds, only the scans differ. Each cohort's `build_db` otherwise draws its own split, so they would not match.
- **Exact reproducibility and canonical splits.** Re-importing a split reproduces it byte-for-byte; an externally-defined or published split can be adopted wholesale.
- **Distinct from the in-memory resplit** (`mm_dataset`, used by `train_resplit`), which hands a datamodule ephemeral per-stage patient codes for throwaway resplit trials and never persists. Predefined splits **persist** into the DB, so the standard `train` → `aggregate` pipeline runs over them unchanged. The two are complementary.

### 1.2 The knobs

Two fields under `splits:` in the `build_db` config:

```yaml
splits:
  import_from: null # a patient→fold CSV; null → draw the split (default)
  allow_uncovered: false # tolerate a file that covers only a subset of labelled patients
```

When `import_from` names a CSV, `build_db` reads that assignment (`import_splits_csv`), validates it against the cohort (`validate_imported_splits`), and writes it verbatim into the DB. When it is null, `build_db` draws the split with `generate_splits` as before. Nothing else in `build_db` (geometry probe, analysis, contract-stat resolution) changes.

### 1.3 The CSV — one format for import and export

One row per (patient, split_version), header exactly `patient_id,split_version,category`:

```
patient_id,split_version,category
P001,recurrence_cv_o0_i0,FIT
P001,recurrence_cv_o0_i1,VALIDATE
P001,recurrence_cv_o1_i0,TEST
```

- `split_version` uses the canonical names the sweep and `aggregate` expect — `<field>_cv_o{k}_i{j}` for nested CV, `all_test_<field>` for a validation cohort, and the single all-predict version. The import must name exactly the set `generate_splits` would emit for that study (see §1.4).
- `category` is a `SplitCategory`: `FIT` / `VALIDATE` / `TEST` / `PREDICT`.
- Splits are per contract field, and `split_version` encodes the field, so one file covers all of a multi-target study's fields.

**Every build exports this file.** `build_db` always writes `splits.csv` in the cohort's DB folder — for a drawn split and an imported one alike — sorted by `(split_version, patient_id)` for a stable, diffable file. So export is a byproduct of every build, the CSV a sibling cohort imports is exactly the one `build_db` emitted, and the file doubles as the split's provenance.

### 1.4 Validation

`import_splits_csv` does structural checks only (header, three columns, the category vocabulary, keeping file order). `validate_imported_splits` is cohort-aware and returns the versions reconciled to the cohort:

- **Fold set.** The imported version-name set must equal `expected_version_names` for the study's active fields (fields with at least one valid-label patient) under its strategy and `n_outer × n_inner`. A mismatch — a wrong grid, or a missing/extra field — is an error naming what is missing and extra.
- **Categories.** Each version's category set must fit its strategy (`nested_cv` → fit+validate+test, `all_test` → test, `all_predict` → predict); a nested-CV fold missing any of fit/validate/test is an error.
- **Membership.** By default, a file patient absent from the cohort, or a labelled cohort patient absent from the file, is an error — this catches a wrong or truncated file. `allow_uncovered=true` is the matched cross-cohort case where the import and the cohort legitimately meet only at their intersection: it downgrades both to a reported drop (file-only patients dropped from the versions, cohort-only patients simply not in the folds). This is how the second cohort of a scanned pair gets the first cohort's split, restricted to the shared patients.
- **Provenance.** On import, `build_db` also writes `splits_source.json` — `{"imported_from": <path>, "sha256_16": <hash>}` — so a reader never mistakes an imported split for a freshly drawn one.

### 1.5 Export is a `build_db` side artifact — no separate bin

There is no `bin/export_splits.py`. Export is the always-emitted `splits.csv` of §1.3, not a pipeline stage — dlux avoids the CLI-dispatcher path that would absorb such read-only utilities, and the emitted CSV is already the portable, inspectable artifact. The round-trip is the test: build a cohort with a drawn split, re-import its emitted CSV into a fresh DB, and the `split_versions` come back identical.

### 1.6 Worked example — two scans of the same patients

1. `build_db` cohort A as the development cohort of study A. It emits `studies/<study A>/db/<cohort A>/splits.csv`.
2. `build_db` cohort B as the development cohort of study B with `splits.import_from=<that csv> splits.allow_uncovered=true`. Patients in both cohorts get the same fold assignment. Patients that only cohort B has are dropped from the imported split.
3. Sweep and `aggregate` each study. Restricted to the patients both cohorts have, the two out-of-fold scores are a matched comparison of the two scans: same patients, same folds.

Cohort B is not a `validation` cohort of study A. A validation cohort must share no patient with the development cohort: `evaluate_external` scores it with every development fold, and each development patient is in the fit or validate split of most folds, so it refuses a cohort with a shared patient (`trained_folds.assert_disjoint_from_development`). To score cohort B with cohort A's models, score fold by fold: fold `(o, i)`'s model on fold `(o, i)`'s TEST patients of cohort B, the only patients that model never saw. That step is a project's own driver.

## Part 2 — why this shape

- **DB-level, not driver-level.** Persisting the shared split into the DB keeps everything downstream on the standard rails: `train` and `aggregate` need no changes. A bespoke driver that re-ran folds from an in-memory split would re-implement checkpointing, best-checkpoint selection, and prediction persistence, and diverge from the pipeline. One localized `build_db` branch buys the whole matched-split workflow on rails.
- **Import a flat file, don't "reference another cohort's DB".** A CSV is decoupled and inspectable, works across studies, and makes the intersection explicit — the file simply omits patients that lack a scan.
- **Export as a byproduct, not a bin.** Emitting the split on every build gives reuse and provenance for free, without elevating a read-only utility to a pipeline stage.
