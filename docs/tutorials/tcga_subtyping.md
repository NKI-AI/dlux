# Tutorial: TCGA subtyping (BRCA vs COAD)

This tutorial runs the full dlux pipeline on public data with a public model. You train an H&E whole-slide image classifier that distinguishes breast cancer (BRCA) from colon cancer (COAD). You develop on slides from one center with nested cross-validation. You then validate on a second center the model never saw.

The example is deliberately small. It pins about 30 slides per center, so the whole pipeline runs on a modest machine. To keep things simple, we stick to a binary classification task.

## What dlux does

This tutorial walks the most common dlux use case. You extract tile features once with a frozen foundation model, cache them, and train a multiple-instance learning model over the cache. The aggregation model is a configuration choice, with attention-based MIL as the default. dlux runs nested cross-validation and pools the out-of-fold predictions into one score.

dlux ships no data of its own. A project brings its own cohorts and reuses every pipeline stage. This example is such a project. It lives under `examples/` and supplies the cohorts, the study, and the run recipe. dlux supplies the pipeline.

## The pipeline

The stages run in order. Each one writes an artifact the next stage reads.

| stage               | what it produces                                                              |
| ------------------- | ----------------------------------------------------------------------------- |
| `generate_masks`    | a tissue mask per slide                                                       |
| `build_db`          | a per-cohort database: slide geometry, masks, and the cross-validation splits |
| `extract_features`  | Phikon tile features, cached per cohort                                       |
| `train`             | one model per cross-validation fold                                           |
| `aggregate`         | the pooled out-of-fold score on the development center                        |
| `evaluate_external` | the score on the second center                                                |

## The catalog: cohorts, studies, experiments

A project supplies its work as three kinds of YAML under `catalog/`, and the pipeline stages read them. The example's YAMLs are under `examples/catalog/`. You write these three files to run dlux on your own data.

**A cohort is one dataset plus its label contract.** `examples/catalog/cohort/tcga_brca_coad_christiana.yaml` says where the slides and masks are (`storage`) and lists its endpoints (`contract`). The line `cancer_type: {type: binary, map: {BRCA: 0, COAD: 1}}` says the `cancer_type` target is binary and how the raw labels map to classes. `build_db` reads the contract to learn each target's type, and no later stage repeats it.

**A study ties cohorts together for its experiments.** `examples/catalog/study/tcga_subtyping.yaml` gives each cohort a role (`tcga_brca_coad_christiana: development`, `tcga_brca_coad_mskcc: validation`), lists the targets to model (`cancer_type`), and sets the number of cross-validation folds (`n_outer: 3`, `n_inner: 2`). The role decides the split: a development cohort gets nested cross-validation, a validation cohort is held out and scored once as external. `build_db` reads the study to assign roles and draw the folds.

**An experiment is the training recipe.** `examples/catalog/experiment/tcga_subtyping/baseline.yaml` picks the task and model (`override /task: wsi_classification`, attention-MIL by default) and the training settings (`batch_size`, `max_epochs`), and carries its own `experiment_name`. `train` reads the experiment. One study can hold several experiments, one per model or setting you want to compare.

So the cohort says what the data and labels are, the study says how to split and validate, and the experiment says how to train. `build_db` needs the cohort and study. `train` needs all three. Running dlux on your own data means writing these three files. The [endpoint recipes](endpoint_recipes.md) do this for other endpoint types.

## Data: pointers, not pixels

This repository ships no slides. For each center it ships a pinned manifest that lists the exact TCGA files, plus `make_manifest.py`, the script that generated it from the public GDC API. You download the slides yourself with the GDC client. TCGA diagnostic slide images are open-access. The slides for both centers total about 32 GB (Christiana 9 GB, MSKCC 22 GB).

## Before you start

Install dlux from a clone of this repo:

```bash
git clone https://github.com/NKI-AI/dlux.git
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install .
```

`pip install .` pulls in `ahcore` and the other dependencies. Run every command below from this clone.

You also need:

- `pip install "ahcore[tracing]" "transformers==4.48.*"` for the one-time encoder bake. Pin `transformers` to the 4.48 line the pack baker is tested against. A newer one changes the traced model output and breaks the bake. (`==4.48.*` restricts to the 4.48 patch series; `~=4.48` would not, since it allows any 4.x.)
- About 33 GB of free disk: 32 GB for the slides, under 1 GB for the pipeline outputs.
- The [GDC Data Transfer Tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool) (`gdc-client`) on your `PATH`. It is a single static binary. The Linux download is a zip inside a zip:

  ```bash
  curl -sL -o gdc.zip "https://gdc.cancer.gov/system/files/public/file/gdc-client_2.3_Ubuntu_x64-py3.8-ubuntu-20.04.zip"
  unzip gdc.zip && unzip gdc-client_2.3_Ubuntu_x64.zip   # the inner archive holds the binary
  chmod +x gdc-client && mv gdc-client ~/bin/            # or anywhere on your PATH
  ```

The version in the URL may have moved on. macOS and Windows builds are on the same page.

## Step 1. Point dlux at the example

Set three env vars, then create the local paths file.

```bash
export DLUX_PROJECT=$PWD/examples                 # the example's catalog + config
export DLUX_EXAMPLE_DATA=/ABS/PATH/TO/tcga_data   # where you will download the slides
export DLUX_ROOT=/ABS/PATH/TO/your_dlux_outputs   # all dlux outputs: databases, caches, runs, results, models
```

```bash
mkdir -p examples/config/local/paths
cp examples/local_paths.example.yaml examples/config/local/paths/default.yaml
```

That paths file just reads `DLUX_ROOT`, so there is nothing to edit. The tutorial uses `$DLUX_ROOT` for the outputs root from here on, and the model store lives at `$DLUX_ROOT/models`.

## Step 2. Bake the encoder

The example uses [Phikon](https://huggingface.co/owkin/phikon), a public pathology foundation model. It needs no HuggingFace login. The bake produces a self-describing `.pack` of about 346 MB.

```bash
python -m ahcore.tools.fomo_pack --models owkin-phikon --output-dir $DLUX_ROOT/models
```

Phikon is released under a non-commercial license.

## Step 3. Get the slides

The manifests under `examples/scripts/tcga_subtyping/manifests/` pin the exact slides, so everyone downloads the same set. Fetch each center into its image directory.

```bash
gdc-client download -n 4 -m examples/scripts/tcga_subtyping/manifests/christiana.txt -d $DLUX_EXAMPLE_DATA/christiana/images
gdc-client download -n 4 -m examples/scripts/tcga_subtyping/manifests/mskcc.txt      -d $DLUX_EXAMPLE_DATA/mskcc/images
```

The download is resumable. Re-run the same command to continue after an interruption.

To pin a different or larger set, see the [manifests README](../../examples/scripts/tcga_subtyping/manifests/README.md).

## Step 4. Build the sheets

`build_sheets.py` turns the downloaded files into the two rigid sheets that `build_db` reads. Run it once per center.

```bash
python examples/scripts/tcga_subtyping/build_sheets.py \
  --gdc-dir  $DLUX_EXAMPLE_DATA/christiana/images \
  --metadata examples/scripts/tcga_subtyping/manifests/christiana_metadata.tsv \
  --out-dir  $DLUX_ROOT/cohorts/tcga_brca_coad_christiana/sheets

python examples/scripts/tcga_subtyping/build_sheets.py \
  --gdc-dir  $DLUX_EXAMPLE_DATA/mskcc/images \
  --metadata examples/scripts/tcga_subtyping/manifests/mskcc_metadata.tsv \
  --out-dir  $DLUX_ROOT/cohorts/tcga_brca_coad_mskcc/sheets
```

## Step 5. Generate masks

`generate_masks` segments tissue with a public model, downloaded on first use.

```bash
python bin/generate_masks.py cohort=tcga_brca_coad_christiana output_dir=$DLUX_EXAMPLE_DATA/christiana/masks
python bin/generate_masks.py cohort=tcga_brca_coad_mskcc      output_dir=$DLUX_EXAMPLE_DATA/mskcc/masks
```

A GPU is faster. The generator picks up the GPU backend automatically when one is available.

By default the run writes only the masks. Add `mask_generator.thumbnail=true` to also drop a low-resolution QC overlay (`<slide>.thumbnail.png`) next to each mask.

To inspect the tissue masks over the slides interactively, launch the built-in viewer (needs the viewer extra, `pip install ".[viewer]"`):

```bash
python bin/serve_slides.py cohort=tcga_brca_coad_christiana
```

Open <http://127.0.0.1:8000>. It reads the cohort's `slides.csv` and overlays each slide's mask, so it works right after this step, before `build_db`. On a remote host, reach the port over an SSH tunnel.

## Step 6. Build the database

`build_db` opens each slide for its geometry, records the masks, and draws the nested-CV split. The study assigns each cohort its role. Christiana develops. MSKCC validates.

```bash
python bin/build_db.py study=tcga_subtyping cohort=tcga_brca_coad_christiana
python bin/build_db.py study=tcga_subtyping cohort=tcga_brca_coad_mskcc
```

`build_db` also writes a dataset summary to `$DLUX_ROOT/studies/tcga_subtyping/db/<cohort>_analysis/summary.md` — slide and patient counts, label balance, the nested-CV split composition, and figures. Open it to sanity-check a cohort before training.

## Step 7. Extract features

Cache Phikon tile features for both cohorts. `feature_extractor=phikon_tile` is required, because the built-in default points at a different encoder.

```bash
python bin/extract_features.py study=tcga_subtyping cohort=tcga_brca_coad_christiana feature_extractor=phikon_tile tiling=uni2_2mpp
python bin/extract_features.py study=tcga_subtyping cohort=tcga_brca_coad_mskcc      feature_extractor=phikon_tile tiling=uni2_2mpp
```

A GPU is strongly preferred here. Otherwise the extractor uses torch to select the best available accelerator.

It also writes a feature-cache summary next to the cache — `summary.md` plus a tiles-per-slide plot (`tiles.png`) — and prints the path.

## Step 8. Train

Train the development cohort. The grid is 3 outer folds by 2 inner folds, so six folds in total, indices 0 to 5. Re-run the loop to fill any gaps. `train` skips a fold that is already done.

```bash
for fold in 0 1 2 3 4 5; do
  python bin/train.py \
    study=tcga_subtyping cohort=tcga_brca_coad_christiana \
    experiment=tcga_subtyping/baseline feature_extractor=phikon_tile \
    task.target.field=cancer_type fold=$fold
done
```

Training logs metrics to an MLflow store under `$DLUX_ROOT/tracking/mlflow`. The first run creates that SQLite database — the wall of `alembic` migration lines you see once is MLflow building its schema, not an error. To watch the training curves, point the MLflow UI at the store (run it from the venv so the MLflow version matches the one that wrote the database):

```bash
mlflow ui --backend-store-uri "sqlite:///$DLUX_ROOT/tracking/mlflow/mlflow.db"
```

Open the URL it prints (default <http://127.0.0.1:5000>).

## Step 9. Aggregate

Pool the out-of-fold predictions on the development center into one score.

```bash
python bin/aggregate.py study=tcga_subtyping cohort=tcga_brca_coad_christiana experiment_name=baseline
```

## Step 10. External validation

Score the trained models on MSKCC, the center they never trained on.

```bash
python bin/evaluate_external.py study=tcga_subtyping cohort=tcga_brca_coad_mskcc experiment_name=baseline
```

## Read the results

Outputs land under `$DLUX_ROOT/studies/tcga_subtyping/`.

- `runs/baseline/tcga_brca_coad_christiana/` holds one directory per fold, each with its checkpoint, predictions, and metadata.
- `results/baseline/tcga_brca_coad_christiana/` holds the aggregate report: pooled metrics, a per-fold table, and a score-distribution plot with one dot per patient.
- The external step writes its own report, the same shape, under `results/baseline/tcga_brca_coad_mskcc/`, scored on MSKCC.

Reopen the slide server on a trained run to inspect attention and predictions per slide (same viewer extra as the mask step):

```bash
python bin/serve_slides.py cohort=tcga_brca_coad_christiana \
  run.study=tcga_subtyping run.experiment=baseline run.field=cancer_type
```

Open <http://127.0.0.1:8000>. Each slide shows the model's attention over its tiles and its prediction, with a correct/wrong badge.

## Scaling up to a serious run

The pinned set is 15 slides per class per center. That's enough to watch the pipeline run end to end, but not enough for a result you would trust. For a real subtyping run, regenerate the manifests with more slides, or with every available slide, as the [manifests README](../../examples/scripts/tcga_subtyping/manifests/README.md) describes. Then re-run the pipeline from Step 3, since the new manifests mean a fresh download. With more patients per fold you also raise the split grid (`n_outer`, `n_inner`) in the study and the batch size in the experiment, as that README notes. The stages themselves do not change. The download and mask-generation/feature extraction just take longer.

## Notes

**Naming the experiment: recipe vs results.** `train` loads the experiment recipe, a config bundle at `catalog/experiment/<study>/<name>.yaml`. Hydra selects it by that study-scoped path, so `train` takes `experiment=tcga_subtyping/baseline`, and the recipe sets its own `experiment_name`. `aggregate` and `evaluate_external` load no recipe. They read the trained arm's saved outputs under the study, keyed by the bare name, so they take `experiment_name=baseline`. It is the same `baseline` arm either way, named as a config to load or as results to read.

**License and attribution.** Phikon is non-commercial. Results here are based on data generated by the TCGA Research Network (<https://www.cancer.gov/tcga>). Confirm the current GDC data-use terms before use.
