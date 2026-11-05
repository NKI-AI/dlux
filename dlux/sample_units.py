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
"""What a flow does per sample unit: the datamodule it builds, the callbacks the unit adds, and how a model's
outputs become per-slide predictions.

A task declares its unit (``task.sample_unit``, see ``ahcore.data.sample_units``). Every flow calls these
functions where the unit matters, and does everything else the same way for every unit.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import lightning.pytorch as pl
from lightning.pytorch import Callback
from lightning.pytorch.trainer import Trainer
from omegaconf import DictConfig

from ahcore.data.interfaces import Stage
from ahcore.data.mm_dataset import MultiModalDataModule
from ahcore.data.tile_dataset import TileDataModule
from ahcore.manifest import DataDescription, DataManager
from ahcore.transforms.tile import TileTransformFactory
from ahcore.utils.io import get_logger

from dlux.config.cohort import Objective, SplitCategory
from dlux.eval.predictions import collect_predictions
from dlux.eval.tile_pooling import SlidePooledValAUROC, TilePooler
from dlux.modalities import TileFeatures
from dlux.tasks.fit_state import FitState

logger = get_logger(__name__)


def _unit(task: Any) -> str:
    unit = task.sample_unit
    if unit not in ("slide", "tile"):
        raise ValueError(f"No flow support for sample_unit {unit!r} ({type(task).__name__}).")
    return str(unit)


def build_datamodule(
    task: Any, data_description: DataDescription, datamodule_cfg: DictConfig, **selection: Any
) -> pl.LightningDataModule:
    """The datamodule that serves ``task``'s items. ``selection`` passes through to the slide datamodule, for
    example a resplit's ``patient_codes_by_stage``."""
    if _unit(task) == "slide":
        return MultiModalDataModule(
            data_description=data_description,
            task=task,
            tile_transform=TileTransformFactory.for_mil_classification(),
            batch_size=datamodule_cfg.batch_size,
            eval_batch_size=datamodule_cfg.eval_batch_size,
            num_workers=datamodule_cfg.num_workers,
            min_tiles=datamodule_cfg.min_tiles,
            **selection,
        )
    if selection:
        raise ValueError(f"A tile datamodule takes no selection, got {sorted(selection)}.")
    return TileDataModule(
        data_description=data_description,
        task=task,
        tile_transform=TileTransformFactory.for_tile_classification(),
        batch_size=datamodule_cfg.batch_size,
        validate_batch_size=datamodule_cfg.eval_batch_size,
        num_workers=datamodule_cfg.num_workers,
        # Pinned host memory lets the large float32 tile batch copy to the GPU while the previous step computes.
        pin_memory=True,
    )


def unit_callbacks(task: Any) -> list[Callback]:
    """The training callbacks ``task``'s unit adds to every run."""
    return [] if _unit(task) == "slide" else [SlidePooledValAUROC()]


def encoder_identity(task: Any, lit_module_cfg: DictConfig, lit_module_choice: str) -> tuple[str, str]:
    """``(model_name, model_hash)`` of what turns a slide's pixels into features, for the run's provenance. A slide
    task reads cached features, so it is their extractor. A tile task trains its own model, so it is that model
    (``lit_module_choice``, the chosen ``lit_module`` config) and its initial weights."""
    if _unit(task) == "slide":
        tiles = next((stream for stream in task.inputs.values() if isinstance(stream, TileFeatures)), None)
        return (tiles.model_name, tiles.model_hash) if tiles else ("", "")
    weights = lit_module_cfg.model.get("weights")
    return lit_module_choice, str(weights) if weights else "scratch"


def slide_predictions(
    task: Any,
    lit_module: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    stage: Stage,
    *,
    trainer_cfg: DictConfig,
    endpoint_type: Objective | str,
    target_key: str,
    fit_state: FitState,
    require_labels: bool = True,
) -> Dict[str, Any]:
    """Per-slide predictions of ``lit_module`` on the ``stage`` split, in the dict ``collect_predictions``
    returns. ``fit_state`` is what the model's fold fixed besides its weights: it maps the predictions back to
    label units and places survival predictions on the fold's time bins.

    A slide task runs each slide through the model where the model is. A tile task runs every tile through
    ``trainer.test`` on the accelerator of ``trainer_cfg``, and scores each slide as the mean of its tiles'
    probabilities. ``trainer_cfg.limit_test_batches`` bounds that pass for a smoke run, and then the slide means
    are partial.
    """
    if stage not in ("test", "predict"):
        raise ValueError(f"Predictions are collected on the test or predict split, not {stage!r}.")
    if _unit(task) == "slide":
        datamodule.setup(stage)
        loader = datamodule.test_dataloader() if stage == "test" else datamodule.predict_dataloader()
        return collect_predictions(
            lit_module,  # type: ignore[arg-type]
            loader,
            endpoint_type=endpoint_type,
            num_classes=task.num_classes,
            target_key=target_key,
            target_mean=fit_state.pred_mean,
            target_std=fit_state.pred_std,
            time_edges=fit_state.time_edges,
            require_labels=require_labels,
        )

    if stage != "test" or not require_labels or Objective(endpoint_type) != Objective.binary:
        raise ValueError(
            f"Tile predictions are pooled on the labelled test split of a binary endpoint, not stage {stage!r}, "
            f"require_labels={require_labels}, endpoint {endpoint_type!r}."
        )
    data_description = task.data_description
    with DataManager(data_description.manifest_database_uri) as data_manager:
        patient_of = {
            str(image.filename): str(patient.patient_code)
            for patient in data_manager.get_records_by_split(
                manifest_name=data_description.manifest_name,
                split_version=data_description.split_version,
                split_category=SplitCategory.TEST,
            )
            for image in patient.images
        }
    pooler = TilePooler()
    # The pooler sees only its own rank's tiles, so the test runs on one device.
    trainer = Trainer(
        accelerator=trainer_cfg.accelerator,
        devices=1,
        limit_test_batches=trainer_cfg.get("limit_test_batches"),
        logger=False,
        enable_progress_bar=False,
        callbacks=[pooler],
    )
    started = time.perf_counter()
    trainer.test(lit_module, datamodule)
    predictions = pooler.predictions(patient_of, target_key)
    logger.info(
        "Pooled the test tiles of %d slide(s) in %.0fs.", len(predictions["slide_ids"]), time.perf_counter() - started
    )
    return predictions
