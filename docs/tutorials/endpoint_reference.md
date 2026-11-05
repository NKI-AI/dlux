# Endpoint reference: what dlux can predict

The [tutorial](tcga_subtyping.md) and [endpoint_recipes.md](endpoint_recipes.md) work binary, continuous, multiclass, and survival endpoints end to end. dlux handles two more, sketched here without a full recipe: what the contract looks like, what data you need, and what differs.

## Per-gene bulk-RNA expression (regression_vector)

Predict a panel of gene-expression values from H&E, one regression output per gene. The cohort declares the endpoint with no map and no column:

```yaml
contract:
  expression: { type: expression }
```

The target values come from a bulk RNA-seq matrix (`<cohort>/rnaseq/matrix.parquet` under `paths.cohorts_dir`, patients × genes), not from a sheet column. The `wsi_bulk_rna` task predicts the gene vector. You choose which genes with `task.target.gene_panel=<name>`: `full` (every gene in the matrix, the default), or a fixed gene list under `rnaseq/panels/<name>.csv`, built once over the whole cohort so it is the same across folds. The task log1p-transforms the counts and z-scores each gene on the fold's fit split. The headline metric is the mean per-gene Pearson correlation across the panel (`gene_pearson_mean`). This endpoint takes the most setup, since it needs the RNA matrix aligned to the slides by patient.

## Multimodal: H&E + RNA fusion

Any endpoint above can be trained on H&E alone, on bulk RNA alone, or on the two fused. RNA is not a separate pipeline: the experiment names which modalities the model uses, and the study can require them so every run compares on the same patients:

```yaml
require_modalities: [bulk_rna] # only patients with an RNA row enter the folds
```

Glioma survival is one place to try fusion, since H&E and RNA carry different signal. Fusion does not always win, and RNA alone can be hard to beat. The point of the study is to compare H&E, RNA, and fusion on the same patients and see. Fusion reuses the same cohort contract and study as a single-modality run. What changes is the experiment: it adds a `bulk_rna` input to the task and selects the `concat_fusion` model, whose branch names match the input names:

```yaml
# @package _global_
defaults:
  - override /task: wsi_survival
  - override /lit_module: concat_fusion
experiment_name: fusion
task:
  inputs:
    tile_features: { modality: tile_features }
    bulk_rna: { modality: bulk_rna, gene_panel: hvg2000 }
```

The `rna_only` model is the RNA-alone comparator. Keep `tile_features` declared in that arm too, so the `min_tiles` filter drops the same patients in every arm.
