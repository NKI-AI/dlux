# Copyright 2026 Joren Brunekreef. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Train one fold of an endpoint on a study's cohort.

    train study=tcga_subtyping cohort=tcga_brca_coad_christiana task.target.field=cancer_type \
          fold=0 experiment=tcga_subtyping/baseline

`fold` is the flat nested-CV index, resolved against the study's n_outer x n_inner grid.
`split_version=cancer_type_cv_o0_i0` names a stored split directly, and is the only way to reach the
non-CV families. Pass exactly one of the two.

The task's sample unit decides the data: a slide task reads the feature cache from `extract_features`
(same cohort + tiling + feature_extractor => same cache), a tile task reads one tile per item. The run
fits on the fold, then tests the best-by-`validate/loss` checkpoint, and persists {best.ckpt, predictions,
metadata} into the canonical runs dir for cross-fold aggregation.

Hydra run dirs are disposable. What is kept is copied into
`${paths.studies_dir}/<study>/runs/<experiment_name>/<cohort>/<split_version>/`.

`main` is a thin spine: resolve -> build -> mlflow -> fit/test -> write -> persist. The steps are in
`dlux.training`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import hydra
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf

from ahcore.utils.io import print_config

from dlux.config.resolvers import register_resolvers
from dlux.eval.persistence import persist_run_artifacts
from dlux.modalities.state import STATE_FILENAME, write_stream_state
from dlux.training import (
    build_setup,
    configure_mlflow,
    fit_and_test,
    load_study_and_cohort,
    resolve_run_context,
    run_metadata,
    scored_test_coverage,
    write_predictions,
)

warnings.filterwarnings("ignore", message=r".*does not have many workers.*")

register_resolvers()


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> None:
    # Before anything builds a model or a loader: weight init, shuffling and augmentation all draw from
    # these generators, and workers=True gives each dataloader worker its own derived seed.
    seed_everything(int(cfg.seed), workers=True)
    print_config(
        cfg,
        fields=(
            "study",
            "cohort",
            "tiling",
            "feature_extractor",
            "task",
            "lit_module",
            "augmentations",
            "callbacks",
            "paths",
            "datamodule",
            "trainer",
            "experiment_name",
            "split_version",
            "force",
        ),
    )
    # -- 1. work out which run this is ----------------------------------------
    ctx = resolve_run_context(cfg)
    if ctx is None:  # already-persisted fold -> clean skip
        return

    # -- 2. the run itself ----------------------------------------------------
    # All the actual work is these four calls: assemble the model and this fold's data, open the
    # mlflow run that will receive the metrics, fit and test the best checkpoint, then make a final
    # pass to collect the per-slide predictions. Steps 1 and 3 are naming and bookkeeping around them.
    #
    # The build has to come first: it resolves the model widths into cfg, and mlflow snapshots cfg as
    # the run's config.yaml artifact — in this order, that snapshot is the config the run really used.
    study, cohort = load_study_and_cohort(cfg, "train")
    setup = build_setup(cfg, study, cohort, ctx.target, split_version=ctx.split_version)
    mlflow_logger = configure_mlflow(cfg, ctx, setup)
    final_epoch, best_ckpt_path, test_metrics = fit_and_test(
        cfg,
        setup.task,
        setup.lit_module,
        setup.datamodule,
        run_logger=mlflow_logger,
        random_baseline=ctx.random_baseline,
        output_dir=ctx.output_dir,
    )
    tiling = OmegaConf.to_container(cfg.tiling, resolve=True)
    assert isinstance(tiling, dict)
    preds = write_predictions(ctx, setup, tiling, cfg.trainer)
    # Only now do both test numbers exist: the streamed per-slide one from trainer.test, and the
    # per-patient one that needed the collected predictions. Logged together, without a step, so each
    # renders as a single bar rather than a point pinned to the final training global_step.
    test_metrics.update(preds.patient_metrics)
    for key, value in test_metrics.items():
        mlflow_logger.experiment.log_metric(mlflow_logger.run_id, key, value)

    # -- 3. persist to the canonical run dir (metadata.json written last) ------
    # The hydra dir is disposable, so kept artifacts are copied out here. metadata.json is
    # written last and acts as the sentinel: a fold without it counts as unfinished, both for the
    # already-persisted check in step 1 and for the cross-fold aggregate.
    metadata = run_metadata(
        cfg,
        ctx,
        setup,
        final_epoch=final_epoch,
        best_ckpt_path=best_ckpt_path,
        test_metrics=test_metrics,
        test_coverage=scored_test_coverage(ctx, setup, preds.scored_patients),
    )
    OmegaConf.save(cfg, ctx.output_dir / "hparams.yaml")
    artifacts: dict[str, str | Path] = {
        "best.ckpt": best_ckpt_path,
        "hparams.yaml": ctx.output_dir / "hparams.yaml",
        "test_predictions.npz": preds.npz_path,
    }
    if preds.writes_csv:  # only require the CSV where it is actually produced (scalar endpoints)
        artifacts["test_predictions.csv"] = preds.csv_path
    if preds.attention_path is not None:  # only models with an attention branch produce one
        artifacts["attention.npz"] = preds.attention_path
    # Fit-derived stream state (bulk RNA's per-gene μ/σ): a cohort with no fit split cannot recompute it,
    # so the run records it. Only streams with such state produce a file, hence the None check. It must
    # be registered here, because persist_run_artifacts copies this map and nothing else.
    state_path = write_stream_state(setup.task, ctx.output_dir)
    if state_path is not None:
        artifacts[STATE_FILENAME] = state_path
    persist_run_artifacts(run_dir=ctx.run_dir, artifacts=artifacts, metadata=metadata, force=ctx.force)


if __name__ == "__main__":
    main()
