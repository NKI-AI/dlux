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
"""Tests for dlux.sample_units and dlux.eval.tile_pooling: each sample unit's datamodule, callbacks and
provenance, the guards on slide predictions, and tile predictions pooled to slides."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from ahcore.data.interfaces import TileRef
from ahcore.data.mm_dataset import MultiModalDataModule
from ahcore.data.tile_dataset import TileDataModule

from dlux.eval.tile_pooling import SlidePooledValAUROC, TilePooler
from dlux.modalities import TileFeatures
from dlux.sample_units import build_datamodule, encoder_identity, slide_predictions, unit_callbacks
from dlux.tasks.fit_state import FitState

_DATAMODULE = OmegaConf.create({"batch_size": 8, "eval_batch_size": 32, "num_workers": 0, "min_tiles": 1})
_TRAINER = OmegaConf.create({"accelerator": "cpu", "devices": 1})


def _task(unit: str, inputs: dict[str, Any] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        sample_unit=unit, inputs=inputs or {}, num_classes=1, data_description=None, collate_policies=lambda: {}
    )


def test_a_slide_task_gets_the_slide_datamodule() -> None:
    datamodule = build_datamodule(_task("slide"), None, _DATAMODULE)  # type: ignore[arg-type]

    assert isinstance(datamodule, MultiModalDataModule)


def test_a_tile_task_gets_the_tile_datamodule_with_the_eval_batch_size() -> None:
    datamodule = build_datamodule(_task("tile"), None, _DATAMODULE)  # type: ignore[arg-type]

    assert isinstance(datamodule, TileDataModule)
    assert datamodule._validate_batch_size == 32
    assert datamodule._pin_memory is True


def test_a_tile_datamodule_takes_no_selection() -> None:
    with pytest.raises(ValueError, match="takes no selection"):
        build_datamodule(_task("tile"), None, _DATAMODULE, patient_codes_by_stage={})  # type: ignore[arg-type]


def test_an_unknown_unit_is_rejected() -> None:
    with pytest.raises(ValueError, match="No flow support for sample_unit 'patient'"):
        build_datamodule(_task("patient"), None, _DATAMODULE)  # type: ignore[arg-type]


def test_only_the_tile_unit_adds_the_slide_pooled_validation_auroc() -> None:
    assert unit_callbacks(_task("slide")) == []
    assert [type(cb) for cb in unit_callbacks(_task("tile"))] == [SlidePooledValAUROC]


class _Features(TileFeatures):
    def __init__(self, model_name: str, model_hash: str) -> None:
        self.model_name, self.model_hash = model_name, model_hash


def test_the_encoder_identity_per_unit() -> None:
    features = _Features("uni2", "399c238ebd69")
    lit_module = OmegaConf.create({"model": {"weights": "imagenet"}})
    scratch = OmegaConf.create({"model": {"weights": None}})

    assert encoder_identity(_task("slide", {"tiles_20x": features}), lit_module, "ab_mil") == (
        "uni2",
        "399c238ebd69",
    )
    assert encoder_identity(_task("slide"), lit_module, "rna_only") == ("", "")
    assert encoder_identity(_task("tile"), lit_module, "tile_cnn") == ("tile_cnn", "imagenet")
    assert encoder_identity(_task("tile"), scratch, "tile_cnn") == ("tile_cnn", "scratch")


def test_predictions_come_from_the_test_or_predict_split_only() -> None:
    with pytest.raises(ValueError, match="test or predict split, not 'fit'"):
        slide_predictions(
            _task("slide"),
            cast(Any, None),
            cast(Any, None),
            cast(Any, "fit"),
            trainer_cfg=_TRAINER,
            endpoint_type="binary",
            target_key="k",
            fit_state=FitState(),
        )


def test_tile_predictions_need_the_labelled_test_split_of_a_binary_endpoint() -> None:
    for stage, endpoint_type in (("predict", "binary"), ("test", "multiclass")):
        with pytest.raises(ValueError, match="pooled on the labelled test split of a binary endpoint"):
            slide_predictions(
                _task("tile"),
                cast(Any, None),
                cast(Any, None),
                cast(Any, stage),
                trainer_cfg=_TRAINER,
                endpoint_type=endpoint_type,
                target_key="k",
                fit_state=FitState(),
            )


def _outputs(logits: list[float], labels: list[int]) -> dict[str, Any]:
    return {"predictions": {"logits": torch.tensor(logits)}, "targets": {"label": torch.tensor(labels)}}


def _batch(slides: list[str]) -> dict[str, Any]:
    return {"tile_ref": [TileRef(slide_id=s, manifest_name="m", tile_index=i) for i, s in enumerate(slides)]}


def test_the_test_tiles_pool_to_slide_means() -> None:
    pooler = TilePooler()
    pooler.on_test_epoch_start(None, None)  # type: ignore[arg-type]
    pooler.on_test_batch_end(None, None, _outputs([0.0, 2.0, -1.0], [1, 1, 0]), _batch(["b", "b", "a"]), 0)  # type: ignore[arg-type]
    pooler.on_test_batch_end(None, None, _outputs([4.0], [1]), _batch(["b"]), 1)  # type: ignore[arg-type]

    predictions = pooler.predictions({"a": "p1", "b": "p2"}, "patient.recurrence")

    assert predictions["slide_ids"] == ["a", "b"]
    assert predictions["patient_codes"] == ["p1", "p2"]
    probs_b = torch.sigmoid(torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64)).mean().item()
    np.testing.assert_allclose(predictions["probs"], [torch.sigmoid(torch.tensor(-1.0)).item(), probs_b])
    np.testing.assert_allclose(predictions["logits"], [-1.0, np.log(probs_b / (1 - probs_b))])
    assert predictions["labels"].tolist() == [0, 1]
    assert (predictions["endpoint_type"], predictions["target_key"]) == ("binary", "patient.recurrence")


def test_excluded_tiles_are_left_out_and_a_confident_slide_keeps_a_finite_logit() -> None:
    pooler = TilePooler()
    pooler.on_test_epoch_start(None, None)  # type: ignore[arg-type]
    pooler.on_test_batch_end(None, None, _outputs([60.0, -5.0, 3.0], [1, -1, -1]), _batch(["a", "a", "b"]), 0)  # type: ignore[arg-type]

    predictions = pooler.predictions({"a": "p1", "b": "p2"}, "k")

    assert predictions["slide_ids"] == ["a"]
    np.testing.assert_allclose(predictions["logits"], [60.0])


def test_a_slide_with_tiles_of_two_labels_is_refused() -> None:
    pooler = TilePooler()
    pooler.on_test_epoch_start(None, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Slide a has tiles labelled 1 and 0"):
        pooler.on_test_batch_end(None, None, _outputs([0.0, 0.0], [1, 0]), _batch(["a", "a"]), 0)  # type: ignore[arg-type]


def test_a_pooled_slide_without_a_patient_is_refused() -> None:
    pooler = TilePooler()
    pooler.on_test_epoch_start(None, None)  # type: ignore[arg-type]
    pooler.on_test_batch_end(None, None, _outputs([0.0], [1]), _batch(["a"]), 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="have no patient in the test split"):
        pooler.predictions({}, "k")


class _Module:
    def __init__(self) -> None:
        self.logged: dict[str, float] = {}

    def log(self, name: str, value: float, prog_bar: bool) -> None:
        self.logged[name] = value


def _validation_auroc(logits: list[float], labels: list[int], slides: list[str]) -> float:
    callback, module = SlidePooledValAUROC(), _Module()
    callback.on_validation_epoch_start(None, module)  # type: ignore[arg-type]
    callback.on_validation_batch_end(None, module, _outputs(logits, labels), _batch(slides), 0)  # type: ignore[arg-type]
    callback.on_validation_epoch_end(None, module)  # type: ignore[arg-type]
    return module.logged["validate/auroc_slide"]


def test_the_validation_auroc_is_over_slide_means_and_skips_excluded_tiles() -> None:
    # Counted, the excluded tile would lift slide a's mean to 9, above both positives, and the AUROC to 0.
    auroc = _validation_auroc([-2.0, -1.0, 3.0, 5.0, 30.0], [0, 0, 1, 1, -1], ["a", "a", "b", "c", "a"])

    assert auroc == 1.0


def test_the_validation_auroc_is_nan_when_undefined() -> None:
    assert np.isnan(_validation_auroc([1.0, 2.0], [1, 1], ["a", "b"]))
    assert np.isnan(_validation_auroc([1.0], [1], ["a"]))
    assert np.isnan(_validation_auroc([float("nan"), 1.0], [0, 1], ["a", "b"]))
