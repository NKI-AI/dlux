# dlux documentation

dlux predicts patient-level endpoints from whole-slide images. The [root README](../README.md) gives the overview and the install steps. This is the guide to running it.

## Where to start

Read **[tutorials/tcga_subtyping.md](tutorials/tcga_subtyping.md)** first. It runs the whole pipeline end to end on public TCGA data with the public Phikon model.

## Going further

- **[tutorials/endpoint_recipes.md](tutorials/endpoint_recipes.md)** adds continuous, multiclass, and survival endpoints on public TCGA data, written as deltas from the tutorial.
- **[tutorials/robust_experiment_comparison.md](tutorials/robust_experiment_comparison.md)** compares two training recipes over many random splits, to tell a real difference from split-to-split noise.
- **[tutorials/endpoint_reference.md](tutorials/endpoint_reference.md)** maps what dlux can predict, including the endpoints it does not give a full recipe for: per-gene bulk-RNA expression, and multimodal H&E + RNA fusion.

## Design specs

[specs/README.md](specs/README.md) lists the specs: precise descriptions of each subsystem (the cohort contract, the input sheets, predefined splits, consumer projects, arm comparison, the slide viewer) and why each looks the way it does. Read them before you change that code.

## Planned

- A general guide to the dlux setup and the full feature set.
