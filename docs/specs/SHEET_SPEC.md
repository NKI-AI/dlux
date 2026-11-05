# dlux — the build-db input sheets

> **The rigid input contract `build_db` ingests.** Part 1 is what build-db reads today, and nothing else. Part 2 is why it looks that way.

Source of truth: `dlux/data/sheets.py` (columns + validation) and `dlux/data/build_db.py` (ingest). A project's own preprocessing (its `scripts/`, below the contract line) produces these sheets; build-db never does locator / remap / blocklist logic.

---

# Part 1 — the sheets

## 1.1 Invocation

```
build_db study=<study> cohort=<cohort> [patients_csv=... slides_csv=...]
```

- **`study` and `cohort` are both required** and cross-checked — the study must list the cohort, and the cohort's role in that study selects its split strategy. One run builds one cohort's DB.
- **Sheet paths default** to `${paths.cohorts_dir}/<cohort>/sheets/patients.csv` and `.../slides.csv`; override with the `patients_csv=` / `slides_csv=` config keys only if they live elsewhere.
- **`masks.csv` is picked up from the same directory as `slides.csv`** — never a CLI/override argument.
- **Image paths in the sheets are relative to `cohort.storage.image_dir`**; mask paths to `cohort.storage.mask_dir`. For a remote reader (`SLIDESCORE`), `image_path` is the slide's URL and is used as-is, with no `image_dir` (§1.3).
- **The DB lands at `studies/<study>/db/<cohort>/manifest.db`** (`{paths.studies_dir}/<study>/db/<cohort>/manifest.db`, computed from study + cohort by `dlux.data.layout`). There is no `database_uri` field on the cohort.
- **Format: CSV only.** XLSX is deferred (add later; two-tab workbook `patients` / `slides`).
- **Sheets are type-agnostic.** A label column holds raw values only (category strings/codes, or floats). The cohort `contract` supplies semantics (`type`, `map`, `stratify`; see `CONTRACT_SPEC.md`). The same sheet shape carries binary / multiclass / continuous with no special columns.

## 1.2 `patients.csv` — one row per patient

| column                         | required       | maps to                                | notes                                                   |
| ------------------------------ | -------------- | -------------------------------------- | ------------------------------------------------------- |
| `patient_id`                   | yes            | `patient.patient_code`                 | unique; FK target for slides                            |
| one column per contract key    | yes (all keys) | `patient_labels(key=<col>, value=raw)` | er/her2/grade/ki67…; raw, mapped at train time          |
| any other patient-level column | no             | `patient_labels`                       | ingested too; clinical-as-input is not built (see §2.4) |

- **Missing / excluded:** an empty/NaN cell, or (categorical) a raw value not in the contract map keys, excludes that patient from **that endpoint only** (target `-1`). Partial labels are normal — this is why every endpoint gets its own splits, over its own valid patients.
- Continuous columns are plain numeric; `type: continuous` drives the float parse + stratification bin.

## 1.3 `slides.csv` — one row per WSI (patient→slide 1:many)

| column       | required | maps to                 | notes                                                               |
| ------------ | -------- | ----------------------- | ------------------------------------------------------------------- |
| `patient_id` | yes      | `image.patient_id` (FK) | must exist in patients.csv                                          |
| `image_path` | yes      | `image.filename`        | relative to `storage.image_dir` (a URL for a remote reader); unique |
| `reader`     | no       | `image.reader`          | per-slide override of `storage.default_reader`                      |
| `staining`   | no       | `image.staining`        | per-slide override of `storage.default_staining`                    |

- **Auto-probed via dlup (not in the sheet):** `image.mpp`, `image.height`, `image.width`. The `probe` knob sets the depth: `none` (don't open slides, geometry NULL — test sheet/split logic without WSIs), `metadata` (open the header only, real geometry, default), `full` (build the tiling dataset — parse mask + grid, require non-empty; catches broken masks / empty grids).
- **The image reader is a choice, not inferred from the extension.** A per-slide `reader` column overrides; otherwise `storage.default_reader`, one of `FASTSLIDE`, `FIMAGE`, `OPENSLIDE`, `SLIDESCORE`, `TIFFFILE`. FASTSLIDE is the default for local slides. It reads `.ndpi .bif .czi .dcm .ets .isyntax .mrxs .ome.tif/.tiff .ome.zarr .qptiff .svs .tif .tiff .vsi .zarr`, so a cohort rarely needs OPENSLIDE. The OpenSlide binaries are therefore an extra rather than a dependency: a cohort that names that reader installs `dlux[openslide]`.
- **Remote slides.** A cohort whose reader is `SLIDESCORE` streams its slides from a SlideScore study. Its `image_path` values are the slide URLs, passed to dlup verbatim. The entry points that read slides (`build_db`, `generate_masks`, `extract_features`, `extract_tiles`) first export the study's API token, read from the file at `paths.cohorts.<cohort>.slidescore_api_key_path`, and refuse a remote cohort without that path (`dlux.data.slidescore`). The token file lives outside the repository. The viewer (`serve_slides`) opens local files with fastslide, so it shows local cohorts only.
- **Masks are read via `imageio`** (`reader="IMAGEIO"`), assumed to be TIFFs.

## 1.4 `masks.csv` — optional, one row per masked slide

| column       | required | maps to         | notes                                        |
| ------------ | -------- | --------------- | -------------------------------------------- |
| `image_path` | yes      | join key        | must exist in slides.csv; unique             |
| `mask_path`  | yes      | `mask.filename` | relative to `storage.mask_dir`; assumed TIFF |

- **Absent file = unmasked cohort.** A slide with no row samples uniformly; partial coverage is fine.
- **One producer per cohort.** `generate_masks` writes it for masks we compute; the cohort builder writes it for masks that ship with the data.
- **Paths are recorded, not derived**, so masks under any naming convention are usable as-is. Masks from `generate_masks` land at `<mask_dir>/<image_path>.mask.tiff`, mirroring the slide's relative path (so two slides sharing a basename cannot collide).

## 1.5 Contract → sheet binding and splits

Contract keys are exactly the target label columns required in `patients.csv` (all asserted present). Non-contract patient columns are ingested as labels but are not targets. Split generation is per contract field, over that field's valid-value patients, stratified on the binarized class (categorical) or bin (continuous):

| strategy      | role          | emits                              | shape                                                       |
| ------------- | ------------- | ---------------------------------- | ----------------------------------------------------------- |
| `nested_cv`   | `development` | `{field}_cv_o{k}_i{j}`             | `n_outer × n_inner` nested CV; TEST_k shared across inner j |
| `all_test`    | `validation`  | `all_test_{field}` (one per field) | every valid-value patient → TEST; scored, not folded        |
| `all_predict` | `predict`     | `all_predict` (single)             | every patient → PREDICT; unlabeled inference, no metrics    |

- Roles map via `ROLE_STRATEGY`: `development → nested_cv`, `validation → all_test`, `predict → all_predict`.
- A `validation` cohort shares no patient with the study's `development` cohort. `evaluate_external` scores it with every development fold, and each development patient is in the fit or validate split of most folds, so it refuses a cohort with a shared patient code. Two cohorts of the same patients (for example two scans of one slide set) are each the development cohort of their own study, with one shared split, and a project scores one with the other's models fold by fold.
- `all_test` is filtered **per field at build time** (one `all_test_{field}` each), exactly like the internal per-endpoint strategies. `all_predict` is a **single** split — it is unlabeled, so there is no field to filter on.
- **No auto "all slides" split is emitted.** Feature extraction is split-agnostic (cache keyed by manifest × tiling × encoder), so `extract_features` enumerates the whole manifest directly.

## 1.6 Worked example (abbreviated)

`patients.csv`:

```csv
patient_id,er,her2,grade,ki67
P001,positive,negative,3,42.0
P002,negative,positive,1,8.5
P003,,positive,2,
```

`slides.csv`:

```csv
patient_id,image_path
P001,P001/slide_a.svs
P001,P001/slide_b.svs
P002,P002/slide_a.svs
```

`masks.csv` (optional; P002/slide_a has no mask and simply has no row):

```csv
image_path,mask_path
P001/slide_a.svs,P001/slide_a.tiff
P001/slide_b.svs,P001/slide_b.tiff
```

P003 is excluded from the `er` and `ki67` splits (empty), kept for `her2` / `grade`.

## 1.7 Validation build-db enforces

1. Required columns present (`patient_id` in both sheets; `image_path` in slides).
2. Every column the contract reads exists in `patients.csv`: each field's `source.column`, its `time_column` (survival) and its `stratify.column`.
3. `patient_id` unique in `patients.csv`; referential integrity: every `slides.patient_id` ∈ `patients.csv`.
4. `image_path` unique in `slides.csv`. With `probe=metadata` or `probe=full`, each slide is opened. A slide that does not open is skipped and counted in build-db's summary.
5. `masks.csv` (if present) has a unique `image_path`, each present in `slides.csv`.
6. A categorical raw value not in the contract map → **warn + exclude** (not fatal).
7. Reserved column names are not used as labels. Reserved **slide** columns: `{patient_id, image_path, reader, staining}`. A contract key may not be `{patient_id, image_path, mask_path, reader}`.

## 1.8 ahcore schema mapping

`patients.patient_id → patient.patient_code` · `patients.<col> → patient_labels` · `slides.patient_id → image.patient_id` · `slides.image_path → image.filename` · `slides.reader → image.reader` · `slides.staining → image.staining` · `masks.mask_path → mask.filename` · probed → `image.mpp/height/width` · `splits.* → split_definitions + split`.

---

# Part 2 — why it looks like this

## 2.1 Sheets sit below the contract line, and are type-agnostic

A label column carries the endpoint's **rawest available form** — no binarisation, no study logic. The cohort `contract` supplies all semantics, so one sheet shape carries every objective and a target's meaning lives in exactly one place. Harmonisation (mapping a cohort's raw encoding onto the study target) is the per-cohort preprocessing script's job, above build-db's floor.

## 2.2 `masks.csv` is a separate sheet

A mask is **derived** per-slide data: computed by `generate_masks` or shipped with the cohort. Held in `slides.csv`, acquiring masks would force a rewrite of the sheet describing the slides — and the workflow re-derives every other row as a side effect. Split out, the dependency points one way: `slides.csv` never changes because masks appeared. New masks mirror the slide's relative path (`<image_path>.mask.tiff`), unique per cohort, so two slides sharing a basename cannot collide.

## 2.3 The reader is a choice, not an extension inference

ahcore's image path is `ImageBackend[reader]` and accepts any member. Inferring the backend from the file extension bakes a guess into ingestion; a per-slide `reader` column (falling back to `storage.default_reader`) keeps it an explicit, overridable decision. The override is how one cohort mixes readers, for example a few slides in a format FASTSLIDE does not read.

## 2.4 Ingest all patient columns; clinical-as-input is deferred

build-db ingests **every** patient column as a label, not just the contract keys, so a clinical covariate is present in the DB without a rebuild. Consuming those columns as **model inputs** is not built (it will want a column-selector, not a channel path); today the non-target columns are carried but not consumed. Ingesting them now keeps that door open at zero cost.
