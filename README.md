# dlux

**A standalone, application-agnostic pipeline for whole-slide-image and multimodal classification.**

## Overview

dlux predicts patient-level endpoints from whole-slide images. Its most common use, and the path the tutorial walks, extracts tile features once with a frozen foundation model, caches them, and trains a multiple-instance learning model over the cache.

- **Objectives.** Binary, multiclass, regression, bulk gene-expression prediction, and survival.
- **Multimodal.** Fuse H&E with bulk RNA-seq and other declared modalities, or run a single modality.
- **Aggregators.** Attention-based MIL is the default. A mean-pooling model is also provided.
- **Rigorous by default.** Nested cross-validation, out-of-fold pooling, and a two-stage arm-vs-arm comparison with a measured noise floor.
- **Bring your own data.** A project supplies its cohorts, studies, and data paths. dlux ships no data.

## How it works

You describe your data once, in three small YAML files: a cohort (a dataset and what its labels mean), a study (which cohorts play which role, and how to split them), and an experiment (the model and its training recipe). The pipeline reads those and runs itself.

It masks the tissue, extracts and caches tile features from a frozen foundation model, then trains a multiple-instance model over that cache. Development runs under nested cross-validation, so every patient is scored by a model that never trained on them, and the out-of-fold predictions pool into one score. A cohort you mark for validation stays held out and is scored once as external evaluation.

dlux also gives you tools to compare training recipes against each other. One works at the nested cross-validation level, reusing the folds you already trained, for a quick read. A deeper one retrains each recipe over many fresh random splits, which separates a robust gain from the data-dependent performance noise that comes with a finite dataset.

## Install

dlux is a clone-and-run pipeline. Install it from a clone of this repo.

```bash
git clone https://github.com/NKI-AI/dlux.git
cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install .
```

`pip install .` pulls in all dependencies from PyPI. You then run the pipeline stages from the clone with `python bin/<stage>.py`. dlux needs Python 3.11 or newer.

## Documentation

Full documentation lives in [`docs/`](docs/README.md). Start there. It opens with a worked example that runs the whole pipeline end to end on public TCGA data with the public Phikon model, then covers the other endpoint types and the recipe-comparison workflow.

## Acknowledgements

Built on [ahcore](https://github.com/NKI-AI/ahcore) and [dlup](https://github.com/NKI-AI/dlup).

## License

Apache License 2.0. See [LICENSE](LICENSE).
