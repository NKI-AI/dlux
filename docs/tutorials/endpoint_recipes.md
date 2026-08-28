# Endpoint recipes: regression, multiclass, survival

The tutorial trains a binary classifier. dlux also does continuous, multiclass, and survival endpoints, and the only thing that changes is the three catalog files from the tutorial's "The catalog" section: the target's `contract` type, the `study` that models it, and the `experiment` recipe. This guide adds three more endpoints on public TCGA data. It assumes you have done the [tutorial](tcga_subtyping.md) and only calls out what differs.

You reuse the tutorial's data steps. Each endpoint is a target column on a sheet, the same `patients.csv` / `slides.csv` the tutorial's `build_sheets.py` produces. For each one below we point at where the labels live and name the column dlux needs. You fetch the slides from GDC as in the tutorial and build the sheet the same way.

## Continuous: leukocyte fraction (regression)

Predict a slide's leukocyte fraction, the immune-infiltration score, as a number in [0, 1]. It is visible in H&E. This uses TCGA-COAD with all source sites pooled, for sample size.

**Data.** Fetch the TCGA-COAD diagnostic slides from GDC. The labels are in Kather's public clinical table `merged_TCGA_TUM_clini_table_v1.xlsx` ([jnkather/MSIfromHE](https://github.com/jnkather/MSIfromHE/blob/master/cliniData/merged_TCGA_TUM_clini_table_v1.xlsx)), column `LeukocyteFraction`. Build a sheet with a `leukocyte_fraction` float column joined to each patient by TCGA barcode.

**Cohort contract.** Declare the target continuous, and balance folds by its quartiles:

```yaml
contract:
  leukocyte_fraction: { type: continuous, stratify: { method: quantile, k: 4 } }
```

**Study.** One development cohort, nested cross-validation:

```yaml
name: tcga_coad_regression
cohorts:
  tcga_coad_molecular: development
targets: [leukocyte_fraction]
splits: { n_outer: 5, n_inner: 5, per_label: true, random_state: 42 }
```

**Experiment.** Swap the classification task for regression, and z-score the target per fold so the loss scale is stable:

```yaml
defaults:
  - override /task: wsi_regression
experiment_name: baseline
task:
  target_normalize: zscore
```

Run the same stages as the tutorial. `aggregate` reports R² and Pearson correlation in place of AUROC.

## Multiclass: glioma WHO grade

Predict WHO grade (II, III, IV) for glioma, on TCGA-GBM and TCGA-LGG pooled.

**Data.** Fetch the TCGA-GBM and TCGA-LGG diagnostic slides from GDC. The grade labels are in the cBioPortal study [lgggbm_tcga_pub](https://www.cbioportal.org/study/summary?id=lgggbm_tcga_pub) (Ceccarelli 2016), joined by TCGA barcode. Build a sheet with a `grade` column holding `G2` / `G3` / `G4`.

**Cohort contract.** A multiclass target maps each class to an index:

```yaml
contract:
  grade: { type: multiclass, map: { G2: 0, G3: 1, G4: 2 } }
```

**Study.** The tutorial's shape, with `grade` as the target:

```yaml
name: tcga_gbmlgg_grade
cohorts:
  tcga_gbmlgg: development
targets: [grade]
splits: { n_outer: 5, n_inner: 5, per_label: true, random_state: 42 }
```

**Experiment.** The default `wsi_classification` task reads the class count from the contract, so the recipe is the tutorial's baseline unchanged. `aggregate` reports multiclass accuracy and, because grade is ordered, quadratic-weighted kappa.

## Survival: glioma overall survival

Predict overall survival, a time-to-event endpoint, on the same TCGA-GBMLGG cohort. It needs no extra download, since the labels come from the same cBioPortal study as grade.

**Data.** That study carries `OS_MONTHS` and `OS_STATUS` per patient. Build a sheet with an event column (1 = died, 0 = censored) and a follow-up-time column.

**Cohort contract.** A survival target names its event and time columns:

```yaml
contract:
  os: { type: survival, event: os_event, time: os_time }
```

**Study.** Target `os`. Folds are stratified by event so each split has comparable event counts:

```yaml
name: tcga_gbmlgg_survival
cohorts:
  tcga_gbmlgg: development
targets: [os]
splits: { n_outer: 5, n_inner: 5, per_label: true, random_state: 42 }
```

**Experiment.** Select the survival task, which fits a discrete-time hazard model:

```yaml
defaults:
  - override /task: wsi_survival
experiment_name: baseline
```

`aggregate` reports Harrell's concordance index (C-index).

To fuse H&E with bulk RNA on this cohort, rather than train on H&E alone, see [endpoint_reference.md](endpoint_reference.md).
