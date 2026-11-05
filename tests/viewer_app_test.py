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
"""Tests for the viewer app on a run of each endpoint type: every route answers with strict JSON, and
the prediction panel gets what its endpoint type can show."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import torch
from fastapi.testclient import TestClient

from dlux.eval.attention import write_attention_npz
from dlux.eval.predictions import write_predictions_npz
from dlux.tasks.survival import survival_risk
from dlux.viewer import create_app
from dlux.viewer.entries import SlideEntry
from dlux.viewer.run_records import load_run_records

FIELD = "target"
MPP, TILE = 0.5, 256
# One slide per outer fold. The second slide has a single tile, so its cross-replicate rank agreement is
# undefined (NaN) and has to reach the browser as null.
SLIDES = {"o0": ("pA/slide_a.tif", 3), "o1": ("pB/slide_b.tif", 1)}
GENES = 50


def _write_slide(path: Path) -> None:
    image = np.full((512, 768, 3), 200, np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolution = (1e4 / MPP, 1e4 / MPP)
    with tifffile.TiffWriter(path) as writer:
        writer.write(
            image, tile=(256, 256), subifds=1, resolution=resolution, resolutionunit="CENTIMETER",
            photometric="rgb", compression="zlib", metadata=None,
        )  # fmt: skip
        writer.write(image[::2, ::2], tile=(256, 256), subfiletype=1, photometric="rgb", compression="zlib")


def _outputs(objective: str, rng: np.random.Generator) -> dict:
    """Head outputs and a label for one slide, in the shapes the training run writes."""
    if objective == "binary":
        logits = rng.normal(size=(1, 1))
        return {"logits": logits, "probs": torch.sigmoid(torch.as_tensor(logits[:, 0])).numpy(), "labels": [1]}
    if objective == "multiclass":
        logits = rng.normal(size=(1, 3))
        return {"logits": logits, "probs": torch.softmax(torch.as_tensor(logits), -1).numpy(), "labels": [2]}
    if objective == "regression":
        logits = rng.normal(size=(1, 1))
        return {"logits": logits, "probs": logits[:, 0], "labels": rng.normal(size=1)}
    if objective == "regression_vector":
        logits = rng.normal(size=(1, GENES))
        return {"logits": logits, "probs": logits, "labels": rng.normal(size=(1, GENES))}
    logits = rng.normal(size=(1, 4))
    return {
        "logits": logits,
        "probs": survival_risk(torch.as_tensor(logits)).numpy(),
        "labels": [[300.0, 1.0]],
        "time_edges": [100.0, 200.0, 400.0],
    }


def _write_run(runs_dir: Path, objective: str) -> None:
    """Two outer folds of two inner replicates, each a complete fold folder as training leaves it."""
    rng = np.random.default_rng(0)
    for outer, (slide_id, n_tiles) in SLIDES.items():
        for inner in (0, 1):
            fold = runs_dir / f"{FIELD}_cv_{outer}_i{inner}"
            coords = np.stack([np.arange(n_tiles) * TILE, np.zeros(n_tiles)], axis=1)
            records = np.concatenate([coords, rng.normal(size=(n_tiles, 1))], axis=1)
            write_attention_npz([(slide_id, records)], fold / "attention.npz", tiling={"mpp": MPP, "tile_size": TILE})
            outputs = _outputs(objective, rng)
            write_predictions_npz(
                {
                    "slide_ids": [slide_id],
                    "patient_codes": [slide_id.split("/")[0]],
                    "logits": np.asarray(outputs["logits"], dtype=np.float64),
                    "probs": np.asarray(outputs["probs"], dtype=np.float64),
                    "labels": np.asarray(outputs["labels"]),
                    "endpoint_type": objective,
                    "num_classes": np.asarray(outputs["logits"]).shape[1],
                    "target_key": f"patient.{FIELD}",
                    "time_edges": outputs.get("time_edges"),
                },
                fold / "test_predictions.npz",
            )
            (fold / "metadata.json").write_text("{}")


def _strict_json(response) -> dict:
    """The body as the browser's ``JSON.parse`` reads it, which has no NaN or Infinity."""
    assert response.status_code == 200, response.text

    def refuse(token: str) -> None:
        raise ValueError(f"bare {token} in the response")

    return json.loads(response.text, parse_constant=refuse)


@pytest.fixture(scope="module")
def image_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("slides")
    for slide_id, _ in SLIDES.values():
        _write_slide(root / slide_id)
    return root


@pytest.mark.parametrize("objective", ["binary", "multiclass", "regression", "regression_vector", "survival"])
def test_every_route_answers_for_each_endpoint_type(tmp_path, image_dir, objective):
    _write_run(tmp_path, objective)
    run = load_run_records(tmp_path, FIELD, inner=None)
    assert run is not None
    entries = [
        SlideEntry(slide_id=slide_id, image_path=image_dir / slide_id, mask_path=None, patient_id=slide_id[:2])
        for slide_id, _ in SLIDES.values()
    ]
    client = TestClient(
        create_app(entries, cohort_name="c", image_dir=image_dir, run=run, run_label="exp", run_classes={0: "x"})
    )

    listing = _strict_json(client.get("/api/slides"))
    assert [s["has_prediction"] for s in listing["slides"]] == [True, True]

    for slide_id, n_tiles in SLIDES.values():
        info = _strict_json(client.get("/info", params={"slide": slide_id}))
        assert info["attention"]["n_tiles"] == n_tiles
        prediction = info["prediction"]
        if objective == "regression_vector":
            # A gene vector has no readable panel, so the slide shows its attention only.
            assert prediction is None
        else:
            assert prediction["endpoint_type"] == objective
            assert (prediction["regression"] is not None) == (objective == "regression")
            assert (prediction["survival"] is not None) == (objective == "survival")

        attention = _strict_json(client.get("/attention.json", params={"slide": slide_id}))
        assert len(attention["logit"]) == n_tiles

        tile = client.get("/tiles/0/1/0/0.jpg", params={"slide": slide_id})
        assert tile.status_code == 200 and tile.headers["content-type"] == "image/jpeg"

    single_tile = _strict_json(client.get("/info", params={"slide": SLIDES["o1"][0]}))
    assert single_tile["attention"]["consistency"] is None
