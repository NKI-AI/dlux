# dlux — the cohort contract

> **What a contract field is and what it admits.** Part 1 is what the code does today, and nothing else. Part 2 is why it looks that way.

Source of truth: `dlux/config/cohort.py`. The contract is the **only** place a prediction target is declared, and every facet it carries is consumed while BUILDING the database. A training-time choice about a target (loss weighting, target standardisation) is not a contract facet — it belongs to the task config.

---

# Part 1 — the spec

## 1.1 Schema

```python
class Objective(str, Enum):
    binary            # 1 logit,  BCE-with-logits,          float{0,1}
    multiclass        # K logits, softmax CE,               long index
    regression        # 1 output, MSE,                      float
    regression_vector # N outputs (gene panel), MSE,        float
    survival          # n_bins hazard logits, discrete-time NLL, Harrell C

class Transform:                     # raw -> modeled label; EXACTLY ONE kind when present
    categorical: dict[str, int]      # raw str -> class idx; keys are an inclusion whitelist
    binarize:    BinarizeSpec        # continuous -> {0,1}; needs a build-db cut (1.5)
    discretize:  DiscretizeSpec      # continuous -> {0..k-1}, k>=3; build-db cut edges (1.5); multiclass
    numeric:     "log1p"                # elementwise; z-scoring is task.target_normalize

class BinarizeSpec:
    method: median | quantile | threshold
    value:  float | None             # threshold only

class Source:                        # WHERE the raw label lives
    modality:    "patient_field" | "bulk_rna" = "patient_field"
    column:      str | None          # defaults to the contract KEY; must be None for bulk_rna
    time_column: str | None          # survival only: follow-up time (column names the event)
    level:       "patient" | "image" = "patient"   # "image" is reserved and raises
    transform:   Transform | None

class Stratify:                      # how CV folds are balanced
    column: str | None               # None -> derive from the modeled target
    method: median | quantile | threshold | None
    k:      int | None               # quantile only, >= 2
    edges:  list[float] | None       # threshold only

class ContractField:                 # THREE facets
    source:    Source
    objective: Objective
    stratify:  Stratify = Stratify()

    @property num_outputs -> int     # 1 | K; RAISES for regression_vector and survival
    @property family      -> str
```

`num_outputs` raises rather than guessing for the two objectives whose head width is not a contract fact: `regression_vector` is the active gene-panel size and `survival` is `task.n_bins`, both resolved at train time.

## 1.2 Shorthand

`{type: ...}` expands to facet form in a `mode="before"` validator. Both forms are accepted; setting `type` **and** `objective` is an error. The contract KEY is not visible here, so `source.column` is left None and injected by the `Cohort` validator.

| shorthand                                              | expands to                                                            |
| ------------------------------------------------------ | --------------------------------------------------------------------- |
| `{type: binary, map: M}`                               | `objective: binary`, `source.transform.categorical: M`                |
| `{type: multiclass, map: M}`                           | `objective: multiclass`, `source.transform.categorical: M`            |
| `{type: continuous}`                                   | `objective: regression`, `stratify.method: median`                    |
| `{type: continuous, target_transform: T, stratify: S}` | `objective: regression`, `source.transform.numeric: T`, `stratify: S` |
| `{type: survival, event: E, time: T}`                  | `objective: survival`, `source: {column: E, time_column: T}`          |
| `{type: expression}`                                   | `objective: regression_vector`, `source.modality: bulk_rna`           |

An unknown `type` raises.

## 1.3 Validation

Per-objective, enforced in `ContractField._check`:

| objective           | transform                                                         | source                               | stratify                                                     |
| ------------------- | ----------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------ |
| `binary`            | `categorical` (exactly 2 classes) or `binarize`; never `numeric`  | column                               | on the class label; `method` forbidden                       |
| `multiclass`        | `categorical` (>= 3 classes) or `discretize` (continuous -> k>=3) | column                               | on the class label; `method` forbidden                       |
| `regression`        | `numeric` or none                                                 | column                               | defaults to `method: median` on the target                   |
| `regression_vector` | none — applied at train time                                      | `modality: bulk_rna`, no column      | **must be empty** (no axis to stratify a high-dim target on) |
| `survival`          | none — event read 0/1, time raw float                             | column (event) **and** `time_column` | free                                                         |

Also: `Transform` must set exactly one kind; `time_column` is survival-only; `modality: bulk_rna` is `regression_vector`-only; `level: "image"` raises (no image-label ingestion exists).

At `Cohort` level: the contract key is injected as `source.column` when absent, and reserved-name clashes are rejected.

## 1.4 Missing values

Dropped from loss and metrics via an objective-aware sentinel, because `-1` is a legitimate regression value. Splits already restrict folds to valid-value patients, so this is a train-time safety net — but it must be objective-correct.

| objective                               | sentinel      | what counts as missing                                         |
| --------------------------------------- | ------------- | -------------------------------------------------------------- |
| binary, multiclass                      | `-1` (long)   | empty cell, or a raw value outside the `categorical` whitelist |
| regression, regression_vector, survival | `NaN` (float) | empty/NaN cell                                                 |

## 1.5 Target statistics

`build_field_transform` is the one place a raw value becomes a modeled label. Five paths:

| transform                | needs a statistic?      | resolved where                                         |
| ------------------------ | ----------------------- | ------------------------------------------------------ |
| `categorical`            | no                      | —                                                      |
| `numeric: log1p`         | no                      | elementwise                                            |
| regression, no transform | no                      | identity, `NaN` for unparseable                        |
| `binarize`               | **yes** — the cut       | build-db, once, over the cohort's valid-value patients |
| `discretize`             | **yes** — the k-1 edges | build-db, once, over the cohort's valid-value patients |

The binarize/discretize cut is written by `build_db` to `studies/<study>/db/<cohort>/contract_stats.json` and read back at task-build time. It must be **fold-invariant**: a cut recomputed per fold would move the ground truth between folds and make pooled OOF incoherent. So a binarize target whose threshold was never recorded **raises** rather than falling back to a per-fold median.

Mild leakage from a cohort-level marginal cut is accepted and standard. A fold-dependent ground truth is not.

The cut is resolved over **that cohort's** valid patients and stored in **that cohort's** stats file. Unlike a `zscore` (μ, σ) or per-gene stats, which restore the development run's recorded state on an external cohort, a `binarize` cut is **not** reused across cohorts. So a binary-from-continuous target scored on a separate-DB external cohort is re-cut at the _external_ median, not at the development cut. Use a `threshold` cut for a binarize target that an external cohort will score: its value is the same in every cohort.

Not to be confused with `task.target_normalize: zscore`, which is live and unrelated: it standardises a regression target on the **fold's fit split** and de-standardises before metrics, so the reported target is unchanged. A training aid, not a redefinition — which is why it is a task knob.

## 1.6 Worked examples

Binary, shorthand and canonical:

```yaml
er: {type: binary, map: {positive: 1, negative: 0}}

er:
  source: {column: er, transform: {categorical: {positive: 1, negative: 0}}}
  objective: binary
```

Multiclass, and an endpoint whose name differs from its column:

```yaml
grade: {type: multiclass, map: {G2: 0, G3: 1, G4: 2}}

grade:
  source: {column: who_grade, transform: {categorical: {G2: 0, G3: 1, G4: 2}}}
  objective: multiclass
```

Regression with an explicit stratify column:

```yaml
leukocyte_fraction:
  source: { column: leukocyte_fraction }
  objective: regression
  stratify: { column: leukocyte_fraction, method: quantile, k: 4 }
```

Continuous column, binary target — the cut is resolved once at build-db:

```yaml
leukocyte_high:
  source:
    { column: leukocyte_fraction, transform: { binarize: { method: median } } }
  objective: binary
```

Survival and expression:

```yaml
os: { type: survival, event: os_event, time: os_days }

expression: { type: expression } # target is the cohort's RNA matrix; unstratified
```

---

# Part 2 — why it looks like this

## 2.1 The facet split

A single `type` cannot drive head, loss, output arity, stratification and target dtype at once, because real endpoints decouple these: a target can be sourced from a _continuous_ column, modelled as _binary_ via a median cut, and stratified on a _third_ column. Splitting `source` / `objective` / `stratify` lets each vary independently, with `type` kept as authoring sugar.

The contract schema is the public surface that cohort YAML depends on. Stability is promised for the YAML, not for the Python APIs behind it.

## 2.2 Same fact vs rhyming behaviour

The rule that keeps this from becoming a god-framework: **consolidate a fact stated in several places; keep implementations that merely rhyme apart.** Output arity and transform selection are facts, single-sourced in `ContractField.num_outputs` and `build_field_transform`. Torchmetrics-on-GPU versus sklearn-on-NPZ, or classification versus regression loss, are different implementations that rhyme — they stay separate.

Concretely: tasks stay concrete config-selected classes, and no objective-generic task dispatches training machinery at runtime.

## 2.3 Why loss weighting is not a facet

Every facet is read while building the database. Loss weighting is read only at train time, by `tasks/build.py`. That makes it a training decision rather than a statement about the endpoint, and two arms of one study must be able to differ on it while sharing a cohort. So it lives at `task.loss.weight`, beside `target_normalize`, which is the same kind of knob. A contract field that carries `loss:` is rejected.

The loss _function_ is still derived from `objective` and lives in each task's `compute_loss`. That is a derivation, not a choice.

## 2.4 Deferred and non-goal

- **Segmentation** is a **non-goal**, not a deferral. dlux predicts at patient level; it may consume compartment masks as an input modality.
- **`level: "image"`** is modelled and rejected. Admitting it needs image-label ingestion and a per-image reader. It exists so that granularity is a data property, not a new concept.
- **Shared binning.** Target `binarize` and fold `stratify` are the same median/quantile/threshold math and should share one `bin_continuous` when binarize is wired. Keep the specs distinct: `BinarizeSpec.value` is one cut resolved once, `Stratify.edges` is k−1 cuts computed per split. That is a real difference in cardinality and provenance, not a naming smell.

## 2.5 Why z-scoring is not a contract transform

`numeric` offers `log1p` only. A z-score changes no claim about the endpoint: R² and Pearson are invariant to an affine transform of the target, so z-scoring would change only the scale-dependent metrics, and would report RMSE in z-units. `task.target_normalize: zscore` gets the optimisation benefit on the fold's fit split and maps predictions back before any metric, so what is reported stays in the target's units.

## 2.6 Labels are keyed on the sheet column

The DB stores labels under the sheet column (`source.column`), not under the endpoint name. Every task reads its label by `source.column`. The two are the same for most endpoints, because `source.column` defaults to the contract key, and differ exactly where the facet split matters: `leukocyte_high` sourcing `leukocyte_fraction`.
