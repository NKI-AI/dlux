# Robust experiment comparison: real gains vs. split luck

You changed one thing in a recipe and the score moved. Is that a real difference, or did you get a lucky split? A nested-CV run does not give one clean number per recipe. It gives one metric per outer fold. That is a handful of numbers with a spread. Compare two recipes and the gap between their means can be smaller than that fold-to-fold spread.

dlux answers this in two steps. The two steps are the same idea at two levels of rigour. The first step is `compare`. It reads the per-fold results you already have from a normal nested-CV run and shows the paired difference between two recipes across the folds. It is cheap and reuses what you trained. It tells you whether a difference is worth a closer look, but it cannot tell you whether the difference is real. The outer folds share their training data, so they are too few and too correlated to test. The second step is `compare_resplits`. It produces the same kind of spread, but more robustly. In place of the few correlated folds it draws many fresh, independent random splits. It also measures how much the score wobbles from training noise alone. With that floor in hand it can report whether a recipe's gap over the baseline clears it. This tutorial runs both, in that order.

It builds directly on the [subtyping tutorial](tcga_subtyping.md). You reuse that study, that cohort, and the tile features you already cached there. Do the tutorial first.

**What this does and does not claim.** A result here is conditional on this one cohort and these patients. It is not a statement about how the recipes generalize to a new dataset. It gives you a handle on one thing. For this data, is a recipe's difference from the baseline larger than the run-to-run noise, or is it within it. This is a real and useful question. It is the one people usually answer by eye and get wrong.

## The three recipes

You compare three recipes, called arms, against a reference.

- **baseline.** The subtyping tutorial's ABMIL recipe, and the reference every other arm is measured against.
- **meanmil.** The baseline with mean pooling instead of attention. Only the model's aggregation differs, so the arm isolates what attention contributes. This subtyping task is near ceiling, so we expect mean pooling to land within the noise floor. Attention is not needed here, and the arm shows you what no real difference looks like.
- **lr_starved.** The baseline with the learning rate starved to `1e-7`. The head barely moves from its random start, so the arm trains to near chance. We expect it to drop well below the baseline and clear the floor. It is the tutorial's example of a real difference.

You see both outcomes side by side. One arm shows a difference that stays within the noise. The other shows a difference the noise cannot explain.

## Step 1. Write the arm recipes

Each arm is its own experiment recipe, a file under `catalog/experiment/tcga_subtyping/`. Give every arm you compare its own file. The file records what the arm was, and the comparison reads the arms back by name, which keeps the whole comparison reproducible.

`baseline.yaml` already exists from the tutorial. Add two more beside it.

`catalog/experiment/tcga_subtyping/meanmil.yaml`:

```yaml
# @package _global_
defaults:
  - override /task: wsi_classification
  - override /lit_module: mean_mil
experiment_name: meanmil
task_name: wsi_classification
datamodule:
  batch_size: 4
trainer:
  max_epochs: 25
```

`catalog/experiment/tcga_subtyping/lr_starved.yaml`:

```yaml
# @package _global_
defaults:
  - override /task: wsi_classification
experiment_name: lr_starved
task_name: wsi_classification
datamodule:
  batch_size: 4
trainer:
  max_epochs: 25
lit_module:
  optimizer:
    lr: 1.0e-7
```

Each differs from the baseline in exactly one place. `meanmil` swaps the model. `lr_starved` lowers one field. Everything else is held equal, so the comparison is about the one change.

## Step 2. Name the comparison

One file names the arms and the reference. Both comparison stages below read it, so the question is stated once.

`catalog/comparison/tcga_subtyping/robustness.yaml`:

```yaml
# @package _global_
comparison_name: robustness
arms: [baseline, meanmil, lr_starved]
reference: baseline
```

## Step 3. Train and aggregate each arm

`compare` reads what `aggregate` wrote, so each arm first needs a normal nested-CV run and its per-fold scores. Those are the training and aggregate steps from the subtyping tutorial. `baseline` you already have. Do the same for the two new arms.

```bash
export DLUX_PROJECT=$PWD/examples

for exp in meanmil lr_starved; do
  for fold in 0 1 2 3 4 5; do
    python bin/train.py \
      study=tcga_subtyping cohort=tcga_brca_coad_christiana \
      experiment=tcga_subtyping/$exp feature_extractor=phikon_tile \
      task.target.field=cancer_type fold=$fold
  done
  python bin/aggregate.py study=tcga_subtyping cohort=tcga_brca_coad_christiana experiment_name=$exp
done
```

Each arm now has its per-fold scores and its pooled out-of-fold score under `results/<arm>/tcga_brca_coad_christiana/`, the same output the tutorial produced for `baseline`.

## Step 4. Compare, the quick first look

`compare` reads those aggregated results and, for each arm, shows the per-fold difference against the baseline.

```bash
python bin/compare.py study=tcga_subtyping cohort=tcga_brca_coad_christiana \
  comparison=tcga_subtyping/robustness
```

It writes each arm's pooled metric, the paired per-outer-fold delta against the baseline, and each arm's patient coverage to `studies/tcga_subtyping/comparisons/robustness/`. Open `summary.md` there to read it. It also renders `metric.png` (each arm's per-fold metric) and `deltas.png` (the per-fold deltas against the baseline), with the raw numbers in `per_fold.csv`. No model is retrained. It only reads.

The two inner folds are ensembled into one prediction per patient, so the delta is one number per outer fold. In this 3-outer 2-inner nested-cv setup, that produces three numbers. These numbers are merely suggestive of the existence of a performance difference. Three outer folds that share training data are too few and too correlated to support a significance test, so `compare` runs none. A delta that looks large here is a reason to run the resplit sweep. That is what the rest of the tutorial does.

## Step 5. Run the resplit sweep

A resplit trial trains one arm on one random split, drawn in memory from a seed. The comparison needs many trials. For every arm it needs several seeds, and three replicates per seed. Replicate 0 is the arm's measurement. Replicates 1 and 2 are the same arm on the same split, with only the training randomness changed. Their difference is pure training noise. That is the floor every effect is measured against.

The split is drawn from the seed alone, so every arm at the same seed trains on a byte-identical partition. That makes the comparison paired. An easy split lifts all arms together, and a hard one drops them together, so the arm-minus-baseline difference cancels the split's luck out. The trials reuse the tile features you cached in the tutorial, so nothing is re-extracted.

Each trial is one invocation, keyed by `(experiment, seed, rep)`. A trial whose result already exists is skipped, so you can stop and resume, and add seeds later for free. Run the sweep as a loop.

```bash
export DLUX_PROJECT=$PWD/examples
SEEDS=10

for exp in baseline meanmil lr_starved; do
  for seed in $(seq 0 $((SEEDS - 1))); do
    for rep in 0 1 2; do
      python bin/train_resplit.py \
        study=tcga_subtyping cohort=tcga_brca_coad_christiana \
        experiment=tcga_subtyping/$exp feature_extractor=phikon_tile \
        task.target.field=cancer_type \
        resplit_name=robustness seed=$seed rep=$rep
    done
  done
done
```

`feature_extractor=phikon_tile` is passed for the same reason the tutorial passes it. The built-in default points at a different encoder, and the trials must read the Phikon cache you already built. That is 3 arms by 10 seeds by 3 replicates, so 90 short trainings. On the 15-per-class tutorial data each one is quick, and the whole sweep runs on one machine. The trials are fully independent, so on a cluster you submit them as array jobs instead of a loop, one task per `(seed, rep)`. More seeds sharpen the floor. Ten is enough to see the shape. You can rerun the loop with a larger `SEEDS` later, and the trials already done are skipped.

## Step 6. Compare against the noise floor

`compare_resplits` reads the rows the sweep wrote and produces the answer.

```bash
python bin/compare_resplits.py \
  study=tcga_subtyping cohort=tcga_brca_coad_christiana \
  comparison=tcga_subtyping/robustness resplit_name=robustness \
  task.target.field=cancer_type
```

For each arm the report gives:

- **The effect.** The median arm-minus-baseline difference across seeds, and its spread.
- **The noise floor.** The 95th percentile of the within-arm replicate differences. This is the wobble training randomness alone produces.
- **The tail fractions.** How often the arm beats or trails the baseline by more than the floor, counted in both directions.
- **The per-arm metric.** Each arm's own score at replicate 0, and the number of patients it scored.

## Read the results

The report lands under `studies/tcga_subtyping/resplits/robustness/tcga_brca_coad_christiana/results/robustness/cancer_type/`. Open `summary.md`; it embeds three figures. `clouds.png` is the one to read first: each arm's delta cloud against the noise floor. `metric.png` is each arm's absolute AUROC, and `margin_convergence.png` shows how the floor settles as seeds accumulate. The per-seed numbers behind them are in `per_seed.csv`.

Read it as an answer to one question. Does the arm's effect stand outside its noise floor? For **meanmil** on this task we expect it does not. The effect sits inside the floor. Because the task is near ceiling, that often shows up as deltas of exactly zero across every seed rather than a small wobble, since most splits score a perfect AUROC for either arm. Any difference here stays within the noise, which shows attention is not needed. It stops short of the stronger claim that meanmil matches the baseline exactly. For **lr_starved** we expect a clear drop that clears the floor in most splits, a difference the noise cannot explain away.

At ten seeds on 15-per-class data these numbers only illustrate the workflow. A result you would report needs a full-cohort cache (see the subtyping tutorial's "Scaling up") and a few hundred seeds. Only the seed count and the amount of data change. The recipes, the comparison file, and the commands stay the same.

One caveat stands, the same one as at the top. This tells you whether a difference exceeds the run-to-run noise on this cohort. It does not tell you the difference holds on a different one. That needs a second cohort, which is what external validation in the main tutorial is for.
