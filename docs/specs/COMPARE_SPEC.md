# `compare` and `compare_resplits`: arm-vs-arm comparison with a measured noise floor

> **Comparing two training arms and saying whether the difference clears the noise floor.** Part 1 is what the two tools do today, and nothing else. Part 2 is why they look that way.

Source of truth: `dlux/eval/compare.py`, `dlux/eval/resplit.py`, `bin/{compare,train_resplit,compare_resplits}.py`. `compare` (stage 1) reads an existing nested-CV sweep for a cheap first look; `compare_resplits` (stage 2) reads a `train_resplit` sweep — R seed-drawn splits with a measured null — and is the answer. Both stages are built. They share one `comparison=<study>/<name>` config: the same arms and reference asked at two levels of rigour.

---

# Part 1 — the tools

## 1.1 The two stages

- **Stage 1 — `compare`.** Reads the `n_outer × n_inner` nested-CV runs that already exist, computes the paired Δ per outer fold, and shows the cloud. No test: the 5 folds share training data so their deltas are not independent, and a signed-rank on 5 pairs bottoms out at p = 1/32 anyway. It answers "is this worth a closer look", nothing more. Never re-runs a model.
- **Stage 2 — `compare_resplits`.** Reads a `train_resplit` sweep (R seed-drawn splits, three replicates per seed) and produces the two clouds, the margin, and the fractions. This is the number to report.

## 1.2 Config, inputs, and the metric per objective

The `comparison` config names the `arms` and the `reference` (shared with `compare`). Both stages read ONLY persisted per-patient artifacts; the objective declares its headline scalar, the same dispatch `eval/` uses, so `compare` and `aggregate` cannot disagree about an arm (each `METRICS` entry delegates to the function `aggregate` already reports with).

**The NPZ is the record, the CSVs are for people.** `aggregate` writes `per_patient_<field>.npz` for every endpoint: the patient codes, the outer fold that tested each patient, the ensembled prediction and the label, plus the endpoint type. Every reader of aggregate results reads that file and nothing else. The CSVs `aggregate` writes beside it exist only for people to read, and no code reads them back. An arm without the NPZ of an endpoint is **refused**.

| objective         | metric                  | chance | prediction, label in the NPZ        |
| ----------------- | ----------------------- | ------ | ----------------------------------- |
| binary            | AUROC                   | 0.5    | positive-class probability, class   |
| multiclass        | macro one-vs-rest AUROC | 0.5    | (K,) class probabilities, class     |
| regression        | R²                      | 0.0    | value, value                        |
| survival          | Harrell's C             | 0.5    | risk, (time, event)                 |
| regression_vector | mean per-gene Pearson   | 0.0    | (G,) genes, (G,) genes (float16/32) |

Arms that predict different gene panels have different label vectors by design, so the label-agreement refusal (§1.3) does not apply to expression.

The first metric listed per objective in `compare._METRICS` is the default. A comparison config selects another of the objective's metrics with `metric:` (for example `metric: qwk` for an ordinal endpoint). Only higher-is-better metrics are offered, so a positive paired delta always reads "the arm is better".

## 1.3 Joining, intersection, coverage, refusals

Join arms on `(field, patient_code)`. Then:

- **Intersect by default, and report each arm's coverage** above the metric table. A patient one arm scored and another did not is a **coverage** fact about the arm's reach, reported — not an error.
- **Refuse, loudly, on:** a shared patient with **disagreeing labels** across arms (one arm is reading a different endpoint or sheet); an **empty intersection**; a **field missing** from an arm.

Coverage and discrimination are different axes and are reported as two numbers, never collapsed:

```
os  ΔC = +0.009 on the 199 shared patients    coverage: arm_a 199/205, arm_b 205/205
```

## 1.4 Stage 2 execution — `train_resplit` then `compare_resplits`

- **One invocation = one run**, keyed by `(experiment, seed, rep)`; the sweep is a job array, no orchestration inside the bin. The results row _is_ the resumability sentinel, so adding seeds later is more array tasks with nothing recomputed.
- **Every arm gets reps 0/1/2** — `3R` tasks per arm. `rep 0` is the arm's measurement; `rep 1`/`rep 2` are its within-arm null pair.
- **The split is drawn in memory from `seed`**, deterministically, never written to the DB. Population is the union of FIT/VALIDATE/TEST of one existing nested-CV split version — nested CV partitions the study population, so any single version's union is the whole admitted set, and it inherits the study's `require_modalities` gate for free. Drawn at **patient** level (slide-level leaks) and stratified the same way `splits.py` does.
- **Seeds:** split seed = `seed`; training seed = `f(seed, rep)`, chosen so no `(seed, rep)` pair collapses — without it `rep 1` reproduces `rep 0` and the null cloud is all zeros.
- **Features are reused as-is at any R** — the cache namespace `(manifest, model, model_hash, grid)` has no split component, so nothing about resplit touches extraction.

Persisted per trial (the standard run dir is wrong at 800–4000 runs):

| artifact                                                                                | kept?                                                                                                                                                   |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| results row `(experiment, seed, rep, metric, n_test, n_pos, split_hash, git_sha, wall)` | **yes** — doubles as the resumability sentinel                                                                                                          |
| per-patient NPZ (~5 KB)                                                                 | **yes** — re-running is expensive, so keep the cheap thing that makes any metric recomputable (else wanting AP instead of AUROC costs R more trainings) |
| `best.ckpt`, `hparams.yaml`, run dir                                                    | **no** — an ephemeral split cannot be externally evaluated, so a resplit checkpoint has no downstream use                                               |

`split_hash` hashes the patient→category assignment: the draw is deterministic from the seed _while the splitter is stable_, and the hash makes drift detectable rather than silent.

## 1.5 The statistics `compare_resplits` computes

Arms are compared **against a reference, not all-vs-all** (N arms give N(N−1)/2 mostly-unasked pairs). Per seed:

```
delta_real[arm] = m(arm, seed, 0) - m(reference, seed, 0)       # config effect + training noise
delta_null      = { m(a, seed, 1) - m(a, seed, 2) for every arm a }  # training noise only
delta           = P95(|delta_null|)                              # the margin: the measured noise floor
```

Because the `delta_real` clouds all share `m(reference, s, 0)`, they are correlated across arms: each arm-vs-reference statement stands alone, but ranking test arms against each other off these numbers does not.

## 1.6 What a report prints

Per endpoint:

- **Both clouds on one axis** (`delta_real` vs `delta_null`) — the answer at a glance.
- **Effect size:** median `delta_real` and its spread.
- **Fraction of splits with `|delta_real| > delta`**, both tails ("in 87% of splits B beat A by more than the floor").
- **Empirical percentile** of median `delta_real` within `delta_null` — a one-sided empirical p whose null was _measured_, not assumed.
- **Per-arm absolute metric** at rep 0, each arm's **coverage**, and the **commit provenance** of the trials (flagged if an arm's rows span several git shas).

And, unconditionally, the conditioning statement — not decoration, it is the difference between a number and a misread number:

> Pooled OOF mixes the outer-fold models, so it estimates the PROCEDURE's performance, not any single model's. Conditional on this cohort. Not a statement about the population.

---

# Part 2 — why it looks like this

## 2.1 The rulers that don't work

We reach for a measured floor because every naive ruler is the wrong one:

- **fold-SEM (`std/√5`) is not an error bar on the effect.** The 5 outer folds share training data, so they are positively correlated and √5 overstates the precision. There is no unbiased estimator of the variance of k-fold CV (Bengio & Grandvalet 2004). An effect that reads "+2.4 SEM" can come out at p≈0.06 under a patient-level bootstrap.
- **A single partition hides partition luck.** Reseeding the fold assignment alone can swing a metric by more than the effect under study. Apparent improvements then evaporate when tested against a measured noise floor.

## 2.2 The paired difference, not the marginal overlap

Plotting the two arms' metric distributions and eyeballing overlap is misleading for a paired comparison. Both arms are scored on the same patients and folds, so their metrics move together: an easy partition lifts both, a hard one drops both. `AUC_A` ∈ [0.60, 0.70] and `AUC_B` ∈ [0.61, 0.71] overlap almost completely, yet B may beat A in _every_ replication by a consistent +0.01. Marginal overlap is not evidence of equivalence — so the object is the distribution of the **paired difference**.

## 2.3 A margin, not zero

A plain signed-rank against zero at R=1000 tests "is the median difference exactly zero", which goes to p→0 for any systematic difference however tiny: a +0.002 effect comes out overwhelmingly "significant", guaranteed and uninformative. The fix is a minimally-relevant effect size `delta` — ask whether the difference exceeds something worth caring about. With the whole distribution in hand a test is not even needed: report the median and the fraction of splits above `delta`, both assumption-free. Report both tails; a one-sided claim is legitimate only if the direction and `delta` were fixed before seeing results.

## 2.4 The margin is the measured floor

`delta` need not be an opinion: take `delta = P95(|delta_null|)`, the 95th percentile of what training randomness alone produces. The claim becomes "the effect exceeds what training noise alone produces 95% of the time", calibrated to the measured floor rather than picked. A floor is specific to `(endpoint, cohort, n)` — never transport it between studies.

## 2.5 Every arm carries its own null, and each needs three reps

`delta_real` carries noise variance `var_A + var_B`. A null measured on the reference alone carries `2·var_A` — equal only if the arms are equally noisy, which nothing guarantees. A noisier test arm makes the floor too low and the comparison over-declares effects. So **the null pools every arm's within-arm `rep1 − rep2` pair**:

```
delta_null = { A1-A2 } ∪ { B1-B2 } ∪ …
```

Its variance is `(2·var_A + 2·var_B)/2 = var_A + var_B` — exactly what `delta_real` carries — and the pooled cloud is larger, sharpening the margin for free with no rescaling. Within-arm differences are the only null material that exists: any A-vs-B difference contains the effect by construction. That forces **three reps per arm** — `delta_real` and `delta_null` must share no run, because the headline statistic is evaluated _per split_, so a shared run would inflate both together on the same split.

**Identical randomness structure.** `delta_real` and `delta_null` must carry training noise the same way. Do not match the torch seed between arms to "reduce noise": `delta_real` would then be quieter than `delta_null` and the comparison biased toward declaring an effect (and it fails anyway — different configs consume the RNG differently). Draw the torch seed independently in every arm, real and null alike.

## 2.6 Intersect by default, coverage as a result

Two failure modes look alike but are not the same thing:

- **The legitimate case.** A 0.5 mpp arm scored 205 patients, a 2 mpp arm 199. `min_tiles` is a _train-time_ gate, applied nowhere near split generation, so both arms have all 205 in their splits and the coarse arm simply **failed to predict 6** (at 2mpp those slides yield fewer tiles than the gate admits). That is one population where an arm has worse reach — refusing to compare would treat a _result_ as an error.
- **The bug case.** Same grid and cache, yet one arm has 2 patients the other lacks, because the study DB was rebuilt between the two runs and the earlier arm is stale. Re-running it brings both to the same set.

The danger is **silence, not intersection**. An opt-in intersection flag would be set once and left on, at which point the bug-case signal disappears forever. Making intersection unconditional but its _consequence_ unmissable — a coverage table the report leads with — keeps the information where it cannot be switched off. The 0.5 mpp arm reaching 6 more patients is a real advantage but not a better-ranking one; one number either hides it or confounds it, two numbers say both.

## 2.7 An objective is a table entry

The join, the pairing, the refusals and the report are the same for every objective. What differs is which table holds the per-patient values and which metric scores them, so an objective is an entry in `compare._METRICS` and `read_arm`, not a branch through the tool. Every metric delegates to the function `aggregate` reports with. The check that the path is right is that each arm reproduces its recorded `aggregate` metric through this independent path, and that a control trained with a vanishing learning rate lands at its objective's chance level.

## 2.8 Controls that make the harness self-validating

A harness that reports "no difference" everywhere is indistinguishable from a broken one, so it needs something it is known to detect.

| arm pair                   | expectation                                      | role             |
| -------------------------- | ------------------------------------------------ | ---------------- |
| **lr ±1%**                 | `delta_real` indistinguishable from `delta_null` | negative control |
| **a known large effect**   | far outside the null cloud                       | positive control |
| the comparison under study | unknown                                          | the question     |

A 1% knob beats a byte-identical copy: a copy only shows the plumbing injects no spurious difference, whereas if 1% lr moves the metric as much as the headline effect, the comparison is fragile regardless of any test. A positive control must be large on the endpoint at hand: an arm swap known to cost a large margin on that cohort, never an endpoint at ceiling (no room to move) or the comparison under study. Where no such arm is known, a `max_epochs: 2` arm is a cohort-agnostic large control.

## 2.9 Null flavours

The **`seed` null** — identical config and split, different torch seed — is the floor any claim must clear. A **`nuisance` null** — a knob changed ~10% that should be immaterial (lr, weight decay) — is broader and more conservative but conflates the floor with knob-sensitivity; run it as a separate, separately-labelled null, because the conflation is itself a result. An mpp ±10% null is parked as conceptually muddy: mpp is not a nuisance knob when the thing under comparison _is_ the grid.

## 2.10 Level 2 (formal testing) is deferred

Level 1 reports the observed spread and lets the reader judge; the empirical percentile is the only inferential number, and its null was measured, so the runs behind it can be pointed at. Formal testing is deferred, and only if we can state exactly what a reported p conditions on:

- **DeLong (1988)** — paired ΔAUC with a closed-form CI and p, standard in clinical venues. The strongest candidate, deferred because it conditions on one partition and the fitted models, so it answers a narrower question than readers take it for — and v1's measured null covers the same ground without that hazard. If ever added it is ~40 lines, easy to get subtly wrong, and must agree with a paired bootstrap on synthetic data before any p it prints is trusted.
- **5×2cv / Nadeau–Bengio** — asks "are the two _procedures_ equal on this dataset", a different question and cost class; the seed null is a cheap partial probe of the same thing.
- **Generalisation.** No test on one cohort licenses "A is better" — that needs a second cohort.

## 2.11 Verification

v1 computes metrics, paired differences and percentiles — arithmetic, not inference — so verification is cheap: metrics against a reference implementation on the real per-patient tables; degenerate inputs (one class, identical predictors → Δ exactly 0, ties); a negative control on real data (the same arm under two names → Δ = 0, percentile 50); and the null arms themselves, which are the harness's self-check — a seed-only null with a Δ spread far from zero is a finding about the procedure, not a bug in the tool.

## 2.12 Still open

- What R does the null cloud need before its shape stops moving? Add seeds, watch the percentile settle; R=1000 is an upper bound, not a requirement.
- Where does `delta` come from by default — the null-calibrated 95th percentile, or always explicit?
- Should `compare` read a validation cohort's `per_patient_<field>.npz` once enough experiments have one? Same shape, and generalisation is the only claim a second cohort can support.
