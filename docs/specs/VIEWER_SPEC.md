# dlux — the cohort slide viewer (`serve_slides`)

> **A first-class entry point for eyeballing every slide in a cohort in the browser, with tissue-mask and attention overlays.** Part 1 is what the viewer does today, and nothing else. Part 2 is why.

Source of truth: `dlux/viewer/app.py` + `entries.py`, `bin/serve_slides.py`, and — for the attention overlay — `eval/attention.py` (capture) + `viewer/run_records.py` (load). Templated on the fastslide example viewer, not vendored from it. Slide tiles are always JPEG (quality from config); mask and attention tiles need alpha.

---

# Part 1 — the viewer

## 1.1 What it serves, and from what

`serve_slides cohort=<name>` serves a cohort's slides as XYZ tile pyramids with a slide picker, a toggleable tissue-mask overlay, and — when a `run` is named (§1.4) — an attention overlay. Purpose is **data QC**: catch bad masks and visually-obvious slide damage. Not in scope: annotation, mask editing, multi-user state, auth.

Input is the cohort's **sheets**, not a DB: `slides.csv` (`patient_id, image_path`) and the optional `masks.csv` (`image_path, mask_path`) beside it, plus the cohort YAML (`storage.image_dir` / `mask_dir`). `cohort=<name>` is the only required input. With a `run` named (§1.4), the viewer also reads the study DB's split table when the DB exists, to mark the slides the study admits for that endpoint. The slide list is **injected** as a list of entries (`id | image_path | mask_path | patient_id`), not discovered by a directory walk, so `open_slide(id)` is a dict lookup over a fixed key set — no path-traversal guard needed. The **slide id is `image_path` relative to `image_dir`** (the sheets' own key), and `only=[…]` serves a subset (e.g. the twelve slides a corruption report flagged). The viewer opens local slide files with fastslide, so a cohort whose slides are remote (SlideScore) is refused at startup.

## 1.2 Endpoints

| route                                         | description                                                                                                  |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `GET /`                                       | slide picker (patient_id, name, mask present y/n)                                                            |
| `GET /api/slides`                             | JSON: cohort name, `image_dir`/`mask_dir`, slide entries                                                     |
| `GET /viewer?slide=<id>`                      | OpenLayers viewer                                                                                            |
| `GET /info?slide=<id>`                        | slide + per-image metadata; mask geometry; attention presence; neighbours (`prev`/`next`/`position`/`total`) |
| `GET /tiles/{image}/{z}/{x}/{y}.{ext}?slide=` | slide tile                                                                                                   |
| `GET /mask/{z}/{x}/{y}.png?slide=`            | mask tile (404 when the cohort has no mask)                                                                  |
| `GET /attention.json?slide=`                  | per-tile attention values for the named run (404 when none)                                                  |

Readers are `functools.lru_cache`d with the `_OpenSlide` dataclass idiom (keeps a reader alive for in-flight requests even when evicted). Neighbours ride along on `/info` so stepping through a cohort of thousands of slides never pulls the whole list per slide; prev/next drive a full page load, keeping map state clean.

## 1.3 The mask overlay

The mask is **resampled server-side onto the slide's tile grid**: `/mask/{z}/{x}/{y}.png` takes the _slide's_ z/x/y, maps that tile's slide-level-0 extent into the mask frame, reads the finest mask level that covers it, and nearest-neighbour resizes to the tile size. The client reuses the slide's `TileGrid` verbatim, so alignment is true by construction. Mask pixel values are `{0, 1}` (a label image), pushed through a LUT (blue tissue, matching `masks.py`) before display; tiles are PNG with alpha, and the browser owns the blend via layer opacity (default 0.35). Resampling is NEAREST **both** server-side (`PIL.Image.NEAREST`) and client-side (`interpolate: false`), or a binary mask gets soft, wrong edges.

**Masks are optional at every level** — four cases, none of which raises, all "no mask layer, slides still work":

| case                                                     | `has_mask` | `mask_declared` | UI                                            |
| -------------------------------------------------------- | ---------- | --------------- | --------------------------------------------- |
| cohort has no `mask_dir`                                 | false      | false           | controls hidden                               |
| cohort has no `masks.csv`                                | false      | false           | controls hidden                               |
| slide has no row in `masks.csv`, or an empty `mask_path` | false      | false           | controls hidden                               |
| declared but absent on disk                              | true       | **true**        | "declared in the sheet but not found on disk" |

The last row is a real QC signal (the sheet promised a mask that isn't there), distinct from plain absence. `/mask/…` returns 404 in all four.

## 1.4 The attention overlay — the `run` group

Naming a trained model via the `run` group (`run.study`, `run.experiment`, `run.field`, `run.inner`) enables inspecting its attention and predictions:

- **Availability is implicit, choice is explicit.** The group's presence is the switch; a partial group (`run.experiment` without `run.field`) is an error, never a silent no-op. Startup logs coverage ("baseline/os inner=0 — 331/1654 slides have attention").
- **`inner` picks one replicate; `inner: null` averages every replicate present.** The **outer fold is derived per slide** (each slide sits in exactly one outer test split), so `inner=0` means "the i0 model of each slide's own outer fold" — whole-cohort coverage, one model per slide.
- **Predictions are the fold's own, never the aggregate ensemble.** Pairing one replicate's attention with the ensemble's probability would describe two different models. The sidebar shows the **head's raw output** (sigmoid for binary; the full softmax vector for multiclass, argmax highlighted; the predicted value for regression; the predicted survival curve for survival), with a correct/wrong badge and **class names inverted from the contract map** (`{"1":0,"2":0,"3":1}` → `{0:"grade 1, 2", 1:"grade 3"}`).
- **Expression shows attention only.** An expression prediction is a vector over thousands of genes, which the sidebar cannot show in a readable form, so `/info` sends no prediction for a `regression_vector` run.
- **Attention data** is the per-slide `attention.npz` an inspected run persisted (§2.5), read eagerly at startup; the server is immutable after. Slides with no attention degrade exactly like a missing mask.
- **Controls:** opacity; a top-k retention toggle (keep the k highest logits, renormalise over them, mirroring GatedAttention); and a colour scale by weight or by rank (per-slide percentile). With `inner: null`, the sidebar also shows a cross-model rank-agreement number and a cumulative-attention- vs-tile-rank plot.
- **Layer defaults:** with a run named, attention starts ON and the mask OFF; without one, the mask keeps its default.
- **In-study slides.** The study's splits say which slides it admits for the endpoint (one split version's fit, validate and test together is the whole admitted set). Each slide is marked in or out of the study, so a slide with no attention reads as "not in the study" or "missing prediction", and prev/next can step through the in-study slides only. Without the DB the marks are absent and a warning says so.

## 1.5 Build wiring

```
dlux/viewer/            # app factory, slide-entry resolution, static assets
bin/serve_slides.py
```

It needs the `viewer` extra (`fastapi`, `fastslide`, `uvicorn`), and ships the static `index.html` and `browser.html`. Runs on a CPU node; the user tunnels to the port.

```bash
serve_slides cohort=tcga_brca_coad_christiana [host=… port=… only=[…]] \
  [run.study=… run.experiment=… run.field=… run.inner=0]
```

---

# Part 2 — why it looks like this

## 2.1 Sheets, not the DB

A cohort exists independently of a study; the DB does not (`studies/<study>/db/<cohort>/manifest.db`), so a DB-backed viewer would force you to name a study just to look at a cohort's slides. Sheets already pair `image_path` with `mask_path` — exactly the viewer's input, which a directory walk cannot reconstruct. And the ordering is right: mask QC happens _before_ the DB exists (generate masks → eyeball → build DB → extract), so requiring a DB would invert the workflow at the point a bad mask is still cheap to fix. The DB is read only when a run is named, and only for split membership (§1.4). `min_tiles` exclusions and labels are not shown.

## 2.2 The mask lives on the slide's grid

The mask and slide pyramids share no grid — on a real pair the mask downsamples 23.93× against the slide, with different zoom ranges and no shared level boundaries. Giving each layer its own `TileGrid` and letting the client overlay pre-scaled resolutions does not work: even with the server geometry correct to 0.03%, the **browser misregisters the overlay at coarse zoom** (the mask sits off the tissue at full zoom-out, then snaps in on zooming), because it reconciles two pyramids with no shared level boundaries below either grid's coarsest level. Serving the mask on the slide's own grid makes alignment true by construction and deletes the whole class of bug, at the cost of one cheap region read per tile. **Do not optimise this back into two grids** — the two-grid version looks cheaper and is the bug. (Measured, served-vs-native: IoU 0.997, centroid offset < 1 px in a 3250×2159 frame.)

## 2.3 Two `XYZPyramid` gotchas, and a NaN

- **`get_tile` returns cropped edge tiles; the server must pad them.** The last column/row come back smaller than `tile_size`, and OpenLayers stretches an undersized image to fill its slot — warping the slide away from the mask at coarse zoom (a single valid pixel smeared across a whole tile, up to 256×). `_slide_tile_bytes` pads with white at the origin. The upstream fastslide example has the same defect.
- **`read_region(location, level, size)` takes LEVEL-native coordinates, not level-0.** Not the OpenSlide convention, and nothing in the signature says so. Latent wherever masks are coarser than every slide level (level 0 always selected, frames coincide); it misplaces the mask on any cohort with a finer mask. Only a cohort whose mask reads select a non-zero mask level exercises this path.
- **`pd.isna` when reading `mask_path`.** An absent value arrives as NaN, which is **truthy**, so a plain falsy check silently yields a `"nan"` filename.

## 2.4 Corrupt slides render blank — a real Stage-1 win

fastslide returns a blank tile for a corrupt read rather than raising (this _is_ the silent-corruption mechanism). In the viewer that becomes visible: corrupt regions render as blank white areas. Stress-tested on slides 55–73% unreadable — 120 full-res requests, 0 errors, no `DATA_LOSS` chatter leaking to the log — so the viewer surfaces by eye a class of defect that otherwise passes through a run silently.

## 2.5 The attention artifact persists RAW LOGITS

Attention is captured on the prediction pass itself (`collect_predictions`, no second forward). `dlux.training.write_predictions` writes it with `eval/attention.py::write_attention_npz`, and `bin/train.py` registers `attention.npz` in the `artifacts` map, so it is copied with the run behind the same `metadata.json` sentinel as everything else. `predict` writes the first fold's attention as `attention_<field>.npz` beside its predictions. It stores the model's **raw logits**, not softmax weights, per `(outer k, inner j)` model:

- **Softmax weights are confounded by slide size** (they sum to 1 over N tiles, so mean weight is forced to 1/N and a 3000-tile slide is incomparable with a 300-tile one). A logit is a per-tile function of that tile's embedding with `f` shared across slides, so similar tiles get similar logits regardless of slide. Softmax/temperature/top-k become display choices derived in the UI.
- **The logit scale is arbitrary** (softmax is shift- and scale-invariant), so only relative comparisons are meaningful and the colormap must calibrate on the observed distribution, never a fixed range.
- **Never average raw logits across models** — each inner model's `f_m` has an independent offset/scale, so a raw mean is dominated by whichever sits highest. Any ensemble view must normalise per model first (softmax, z-score, or rank). The artifact does not enforce this; it is an easy mistake to make later.
- **Persist per model, never aggregate at write time** (~15 KB/slide; ~25 MB for a cohort of ~1,700 slides). Averaging on write is lossy and destroys the inter-model-disagreement diagnostic; averaging on read is free.

## 2.6 What makes the inner models comparable

Tile sampling is **fit-only** (`num_tiles: 250` on the `fit:` stage; inference uses the full bag), so at test time every inner model sees the identical, complete tile bag for a slide — which is the only reason their attention vectors are element-wise comparable. Attention exists only for a model with a tile branch: an RNA-only model has none, so the UI must not imply otherwise. The model declares its own attention module as `attention_encoder` (ahcore's ABMIL classifier and fusion model; `None` for a model without attention), so dlux reads that property rather than searching the module tree.

## 2.7 Show the head's raw output, named from the contract

Two displays mislead and are not used: "78% confident" for a negative binary prediction (that is `1 − p`, a number the model never emitted) and "P(grade 3) = 0.224" under a predicted grade 1/2 (naming a class that is not the prediction). The rule: show the sigmoid / full softmax vector, report confidence for the **predicted** class, and name classes by inverting the contract's categorical map. The full vector also makes near-misses legible (`G3 0.467 / G4 0.457` is a torn model, not a plain error), which matters where a cohort's own config flags a pair as the hard part.

## 2.8 What is not built yet

The attention overlay is built end to end: persistence (`attention.npz`), the render, the opacity / top-k-retention / colour-scale (by-weight or by-rank) controls, per-model-normalised averaging (`inner: null`), a cross-model rank-agreement number and a cumulative-attention-vs-rank plot. Top- and bottom-k attended tiles need no callback of their own: with logits and coords on disk, top-k is an argsort plus a crop. What remains is minor:

- A **temperature** control (the top-k retention plus the by-weight/by-rank scale arguably already cover the need).
- **Cohort-wide colour calibration** — the scale is per-slide today (the sensible default for a heavy-tailed distribution); a cohort-fixed range is not built.

Any further ensemble aggregation stays subject to §2.5's never-average-raw-logits constraint, which `run_records._weights` already enforces: it takes a softmax over each model's logits per slide before `_combine` averages them.

## 2.9 Open questions

- **Picker ergonomics at thousands of slides** — plain list, or search/filter + paging? Sheets give `patient_id` for grouping, and the existing `<stem>.thumbnail.png` QC thumbnails could back a contact-sheet view where masks exist.
- **Multi-image slides** (VSI navigator + region images) — the template handles them via an image switcher; verify against a real multi-image slide if a cohort has one.
- **Serving over the cluster** — document the port-forward workflow once it settles.
