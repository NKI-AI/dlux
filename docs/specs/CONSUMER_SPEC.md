# dlux — consumer projects

> **How a project consumes dlux.** This is the contract the code implements today.

Source of truth: the `hydra.searchpath` + `DLUX_PROJECT` env in each bin's primary config (e.g. `config/train.yaml`). A project owns its cohorts, studies, experiments and any code of its own, and reuses every dlux pipeline stage without copying pipeline config. dlux ships **no cohorts and no data paths**.

---

## 1.1 Layout

```
<project>/
├── catalog/              # WORLD-bound — mirrors dlux/catalog/
│   ├── cohort/  study/  experiment/<study>/  comparison/<study>/
├── config/               # CODE-bound — mirrors dlux/config/
│   ├── task/  lit_module/          # only if the project adds its own, paired with src/
│   └── project/paths/default.yaml  # this project's cohort INPUT roots  (`# @package paths`)
├── src/<project>/        # custom Task/model; code that consumes dlux OUTPUTS
├── bin/                  # entry points for project-specific stages ONLY
├── scripts/              # source data -> rigid sheets (below the contract line)
├── notes/  pyproject.toml  README.md   # pyproject.toml only when the project ships code (src/ or bin/)
```

`catalog/` and `config/` keep dlux's own meaning, so there is one vocabulary to learn. A project that adds no code of its own has an empty `config/` bar `project/paths`.

## 1.2 How dlux finds it

One environment variable, two searchpath roots:

```yaml
# every dlux entry config
hydra:
  searchpath:
    - file://${dlux.project:}/catalog
    - file://${dlux.project:}/config
```

`DLUX_PROJECT` is required. `dlux.project` (`dlux/config/resolvers.py`) takes a relative value from the current directory (the workspace root under `bazel run`). It fails with the value and its resolved path if the variable is unset or the directory lacks `catalog/` or `config/`. dlux ships no catalog, so there is no default. Every bin calls `register_resolvers()` before `@hydra.main`, because the searchpath needs the resolver.

**Code, by the same knob.** The searchpath above finds a project's _config_. Its _code_ — the custom Task/model classes that `_target_` entries name — lives at `$DLUX_PROJECT/src`, and dlux appends that to `sys.path` at startup (`dlux/__init__.py`). So the one `DLUX_PROJECT` variable supplies both halves of a project: **config through the Hydra searchpath, code through `sys.path`.** Those classes then resolve wherever dlux runs, an installed venv or a clone: `_target_: my_project.tasks.MyTask` imports by the env var alone. This is the ONLY mechanism; dlux never depends on a project.

**Project drivers.** A project that ships stages of its own puts their drivers in `bin/`. Each gets its own primary config in the project's `config/`, with `@hydra.main(config_path="../config", config_name="<driver>")`. A driver that trains extends dlux's training config by listing `train_base` (the body of `train.yaml`) in its defaults and declaring its own keys, so a misspelled key fails instead of being added with `+`. It cannot list `train` itself: Hydra accepts `hydra.searchpath` only in the primary config. So the driver's config sets its own searchpath, `file://${dlux.repo_path:config}` for dlux's config groups and `file://${dlux.project:}/catalog` for the project's catalog. The project's `pyproject.toml` installs its code into the venv beside dlux. The dependency runs one way: a project driver depends on dlux, never the reverse.

**Appended, never prepended.** An _installed_ copy of the project (e.g. pip-installed, non-editable, into the venv a sweep runs on) sits earlier on `sys.path` and is found first, so the append takes effect only when the project is not installed. It therefore cannot shadow the frozen snapshot a reproducible sweep depends on: the hook is only the fallback for running in place from a clone. A relative `DLUX_PROJECT` is resolved the same way as for the searchpath.

## 1.3 The paths contract

Three groups merge into one `paths` object, each with one owner:

| group                        | owner       | holds                                                    |
| ---------------------------- | ----------- | -------------------------------------------------------- |
| `paths/default.yaml`         | dlux        | output roots, `???` — the public shape                   |
| `local/paths/default.yaml`   | the SITE    | real output roots (`root_dir`, `models_dir`)             |
| `project/paths/default.yaml` | the PROJECT | cohort INPUT roots (`cohorts.<name>.image_dir/mask_dir`) |

Loaded in that order, so a project adds without fighting anything:

```yaml
defaults:
  - optional local/paths: default
  - optional project/paths: default # resolved from the project's config/ root
```

Both `optional`: dlux ships no `project/` dir, so dlux's own runs are unaffected. **Nothing is passed on the CLI.** A project's file needs `# @package paths` — without it the values land in the wrong package and are silently ignored.

**dlux's `paths` group carries no `cohorts:` map at all.** A cohort's storage is declared by whoever owns the cohort.

## 1.4 What a project may contribute

| case                                                                                          | outcome                                   |
| --------------------------------------------------------------------------------------------- | ----------------------------------------- |
| a **new name** in any dlux group (`lit_module/my_model.yaml`, `task/my_task.yaml`, `study/…`) | project wins — select it normally         |
| a dlux/site **config file** of the same name (a second `local/paths/default.yaml`)            | **dlux/site wins — you can't replace it** |

So a project extends by adding new names, not by replacing dlux's own files — which is why the project contributes through the SEPARATE `project/paths` file rather than a second `local/paths/default`. It can still set **values** inside `paths`: `project/paths` loads after `local/paths` in the defaults list, so keys it sets merge over and win. Most projects only add their `cohorts:` map and share the site's `root_dir`; a project that wants its own output root sets `root_dir` in `project/paths` (it takes effect).

## 1.5 Invocation

Standard stages need no project code:

```bash
export DLUX_PROJECT=<project>
python bin/build_db.py study=tcga_subtyping cohort=tcga_brca_coad_christiana
python bin/train.py study=tcga_subtyping cohort=tcga_brca_coad_christiana \
  experiment=tcga_subtyping/baseline task.target.field=cancer_type fold=0
```

A sweep is one `train` per fold, `fold=0` to `fold=<n_outer × n_inner − 1>`, for example as a job array. `DLUX_PROJECT` may be relative or absolute. A wrong value fails at compose time, naming the value and its resolved path.

Project-specific stages are the project's own drivers (§1.2, "Project drivers"), reading persisted dlux artifacts under `studies/<study>/results/`.

## 1.6 Checking a project composes

```bash
DLUX_PROJECT=<project> <dlux bin> ... --cfg job --resolve
```

`--resolve` matters: without it a missing cohort path stays an unresolved interpolation and compose succeeds. With it you get `InterpolationKeyError: Interpolation key 'paths.cohorts.<name>.image_dir' not found`, which names the cohort and the key.
