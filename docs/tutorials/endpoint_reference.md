# Endpoint reference: what dlux can predict

The [tutorial](tcga_subtyping.md) and [endpoint_recipes.md](endpoint_recipes.md) work binary, continuous, multiclass, and survival endpoints end to end. dlux handles two more, sketched here without a full recipe: what the contract looks like, what data you need, and what differs.

## Per-gene bulk-RNA expression (regression_vector)

Predict a panel of gene-expression values from H&E, one regression output per gene. The cohort declares the endpoint with no map and no column:

```yaml
contract:
  expression: { type: expression }
```

The target values come from a bulk RNA-seq matrix (`matrix.parquet`, patients × genes) placed beside the cohort's sheets, not from a sheet column. The `wsi_bulk_rna` task fits the panel, and you choose which genes with `task.gene_panel=<name>`, a fixed gene list built once over the whole cohort so it is the same across folds. Metrics are per-gene R² pooled across the panel. This endpoint takes the most setup, since it needs the RNA matrix aligned to the slides by patient.

## Multimodal: H&E + RNA fusion

Any endpoint above can be trained on H&E alone, on bulk RNA alone, or on the two fused. RNA is not a separate pipeline: the experiment names which modalities the model uses, and the study can require them so every run compares on the same patients:

```yaml
require_modalities: [bulk_rna] # only patients with an RNA row enter the folds
```

Glioma survival is one place to try fusion, since H&E and RNA carry different signal. Fusion does not always win: in our runs RNA alone is often hard to beat, and adding H&E does not meaningfully improve on it. The point of the study is to compare H&E, RNA, and fusion on the same patients and see. Fusion reuses the same cohort contract and study as a single-modality run. What changes is the experiment's model and the modality requirement.
