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
"""FastAPI app serving a cohort's slides as XYZ tile pyramids.

Each zoom level is one native pyramid level, so tiles are direct native reads and the server never
resamples; OpenLayers scales between levels client-side.
"""

from __future__ import annotations

import functools
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import fastslide
import numpy as np
import PIL.Image
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from fastslide.xyz_pyramid import XYZPyramid

from dlux.eval.run_records import RunRecords
from dlux.viewer.entries import SlideEntry

_STATIC = Path(__file__).resolve().parent / "static"

_MEDIA_TYPES = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}

# Tissue colour for the mask layer. Tiles are emitted fully opaque on a transparent background; the
# client sets layer opacity, so the blend is adjustable without re-fetching tiles.
_MASK_RGB = (0, 0, 255)

# Slide background, used to pad partial edge tiles.
_PAD_RGB = (255, 255, 255)


def _finite(value: float | None) -> float | None:
    """None for a non-finite value. ``json.dumps`` writes NaN as the bare token ``NaN``, which is not
    JSON and makes the browser's ``JSON.parse`` throw, the whole sidebar fails on one absent number,
    so undefined statistics have to leave as null."""
    return None if value is None or not np.isfinite(value) else float(value)


def _slide_tile_bytes(pyramid: XYZPyramid, z: int, x: int, y: int, tile_size: int, fmt: str, quality: int) -> bytes:
    """Encode one slide tile, padding partial edge tiles to the full tile size.

    ``XYZPyramid.get_tile`` crops at each level's right and bottom edge, and XYZ clients stretch an
    undersized tile to fill the full slot. Valid data starts at the tile's top-left, so it pastes at
    the origin.
    """
    tile = pyramid.get_tile(z, x, y)
    if tile.size != (tile_size, tile_size):
        canvas = PIL.Image.new("RGB", (tile_size, tile_size), _PAD_RGB)
        canvas.paste(tile.convert("RGB"), (0, 0))
        tile = canvas
    buffer = io.BytesIO()
    if fmt.lower() in ("jpg", "jpeg"):
        tile.convert("RGB").save(buffer, format="JPEG", quality=quality)
    else:
        tile.save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass(frozen=True)
class _OpenSlide:
    """An opened slide: its reader plus one pyramid per navigable image.

    Holding an instance keeps the reader alive, so the weak handle inside each pyramid stays valid for
    as long as a caller references it. Cache eviction is therefore safe mid-request. The entry is torn
    down only once the last in-flight request drops its reference.
    """

    reader: fastslide.FastSlide
    pyramids: tuple[XYZPyramid, ...]


def _mask_tile_png(mask: XYZPyramid, slide: XYZPyramid, z: int, x: int, y: int, tile_size: int) -> bytes:
    """Render the mask over the region covered by slide tile ``(z, x, y)``, as an RGBA PNG.

    The mask has its own mpp, level count and zoom range, so it is resampled onto the slide's tile
    grid and both layers share one grid. Resampling is nearest-neighbour to keep the label image
    binary.
    """
    slide_res = slide.resolutions[z]  # slide level-0 px per tile px at this zoom
    span = tile_size * slide_res  # slide level-0 px covered by the tile, per axis
    slide_w, slide_h = slide.level0_dimensions
    mask_w, mask_h = mask.level0_dimensions
    scale_x, scale_y = slide_w / mask_w, slide_h / mask_h

    # The tile's extent in the mask's own level-0 frame.
    mask_x, mask_y = x * span / scale_x, y * span / scale_y
    span_x, span_y = span / scale_x, span / scale_y

    source = mask.slide
    downsamples = list(source.level_downsamples)
    # Finest level that is still coarse enough to cover the span without oversampling.
    target = max(span_x, span_y) / tile_size
    level = 0
    for index, downsample in enumerate(downsamples):
        if downsample <= target:
            level = index
    downsample = downsamples[level]

    want_w, want_h = max(1, round(span_x / downsample)), max(1, round(span_y / downsample))
    level_w, level_h = source.level_dimensions[level]
    origin_x, origin_y = int(mask_x / downsample), int(mask_y / downsample)

    canvas = np.zeros((want_h, want_w), dtype=np.uint8)
    read_w, read_h = min(want_w, level_w - origin_x), min(want_h, level_h - origin_y)
    if read_w > 0 and read_h > 0:  # tiles past the mask's edge stay fully transparent
        # read_region takes level-native coordinates, not level-0 ones.
        region = source.read_region((origin_x, origin_y), level, (read_w, read_h))
        values = np.asarray(region.to_interleaved().numpy())
        canvas[:read_h, :read_w] = values[..., 0] if values.ndim == 3 else values

    tissue = np.array(PIL.Image.fromarray(canvas).resize((tile_size, tile_size), PIL.Image.NEAREST))
    rgba = np.zeros((tile_size, tile_size, 4), dtype=np.uint8)
    for channel, level_value in enumerate(_MASK_RGB):
        rgba[..., channel] = np.where(tissue > 0, level_value, 0)
    rgba[..., 3] = np.where(tissue > 0, 255, 0)
    buffer = io.BytesIO()
    PIL.Image.fromarray(rgba, "RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def create_app(
    entries: Sequence[SlideEntry],
    *,
    cohort_name: str,
    image_dir: Path,
    mask_dir: Path | None = None,
    run: RunRecords | None = None,
    run_label: str = "",
    run_classes: dict[int, str] | None = None,
    in_study: set[str] | None = None,
    tile_size: int = 256,
    jpeg_quality: int = 90,
):
    """Builds the FastAPI app serving ``entries``.

    Args:
        entries: the slides to serve, in the order the viewer steps through them; ``slide_id`` is the
            key the endpoints take as ``?slide=``.
        cohort_name: shown in the picker.
        image_dir: the cohort's slide root; shown in the picker so the relative ids have a referent.
        mask_dir: the cohort's mask root, if it has one.
        run: one model's attention and predictions; None disables both.
        run_label: shown in the sidebar to identify which model produced them.
        run_classes: class index -> human name, from the endpoint's contract map.
        in_study: the slides the study's splits admit for this endpoint. Distinct from having a
            prediction: a slide in here without one is a gap (a fold that never ran, or a slide dropped
            at dataset construction), while a slide outside it is absent by design. None hides the
            distinction, which is what serving a cohort with no run does.
        tile_size: tile edge length in pixels.
        jpeg_quality: quality used when encoding JPEG tiles.

    Returns:
        A configured :class:`fastapi.FastAPI` instance.
    """
    by_id = {entry.slide_id: entry for entry in entries}
    order = [entry.slide_id for entry in entries]

    # Bounded warm-reader cache. lru_cache is thread-safe, which matters because FastAPI runs these
    # sync endpoints in a threadpool. maxsize bounds how many readers stay warm, not correctness.
    @functools.lru_cache(maxsize=16)
    def open_slide(slide_id: str) -> _OpenSlide:
        entry = by_id.get(slide_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"no slide {slide_id!r} in cohort {cohort_name!r}")
        if not entry.image_path.exists():
            raise HTTPException(status_code=404, detail=f"slide file missing on disk: {entry.image_path}")
        reader = fastslide.FastSlide.from_file_path(str(entry.image_path))
        images = reader.images
        pyramids = tuple(XYZPyramid(images[i], tile_size=tile_size) for i in range(len(images)))
        return _OpenSlide(reader=reader, pyramids=pyramids)

    # Masks are optional at every level: the cohort may declare no mask_dir, the sheet may have no
    # mask_path column or an empty value, and a declared mask may not exist on disk yet. All of those
    # resolve to None here and the viewer simply omits the layer.
    @functools.lru_cache(maxsize=16)
    def open_mask(slide_id: str) -> _OpenSlide | None:
        entry = by_id.get(slide_id)
        if entry is None or entry.mask_path is None or not entry.mask_path.exists():
            return None
        reader = fastslide.FastSlide.from_file_path(str(entry.mask_path))
        images = reader.images
        return _OpenSlide(reader=reader, pyramids=(XYZPyramid(images[images.primary_index], tile_size=tile_size),))

    @functools.lru_cache(maxsize=32)
    def slide_attention(slide_id: str) -> dict | None:
        """The slide's tile positions, raw logits and tile footprint, ready to send to the client."""
        records = run.attention.get(slide_id) if run else None
        if records is None:
            return None
        opened = open_slide(slide_id)
        pyramid = opened.pyramids[opened.reader.images.primary_index]
        slide_mpp = (pyramid.info().get("mpp") or [None])[0]
        if not slide_mpp:
            return None
        return {
            "x": records[:, 0].astype(float).round(1).tolist(),
            "y": records[:, 1].astype(float).round(1).tolist(),
            "logit": records[:, 2].astype(float).round(4).tolist(),
            "footprint": run.tile_size * (run.mpp / float(slide_mpp)),
            "n_tiles": int(records.shape[0]),
        }

    def _mask_info(slide_id: str, opened: _OpenSlide) -> dict | None:
        """Mask native size and downsample relative to the slide, or None if there is no mask.

        Descriptive only: the mask layer uses the slide's tile grid, so no geometry is needed to
        place it.
        """
        mask = open_mask(slide_id)
        if mask is None:
            return None
        pyramid = mask.pyramids[0]
        slide_w, slide_h = opened.pyramids[opened.reader.images.primary_index].level0_dimensions
        mask_w, mask_h = pyramid.level0_dimensions
        return {
            "dimensions": [mask_w, mask_h],
            "mpp": pyramid.info().get("mpp"),
            "level_count": pyramid.info().get("level_count"),
            "scale": [slide_w / mask_w, slide_h / mask_h],
        }

    app = FastAPI(title=f"dlux slide viewer — {cohort_name}")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC / "browser.html", media_type="text/html")

    @app.get("/viewer")
    def viewer() -> FileResponse:
        return FileResponse(_STATIC / "index.html", media_type="text/html")

    @app.get("/api/slides")
    def api_slides() -> JSONResponse:
        return JSONResponse(
            {
                "cohort": cohort_name,
                "image_dir": str(image_dir),
                "mask_dir": str(mask_dir) if mask_dir else None,
                # Null unless a run is being inspected, which is what hides the column and the filter.
                "run": run_label if run is not None else None,
                "slides": [
                    {
                        "slide_id": e.slide_id,
                        "name": e.name,
                        "patient_id": e.patient_id,
                        "has_mask": e.mask_path is not None,
                        "has_prediction": run is not None and e.slide_id in run.attention,
                        # None when unknown (no study consulted), so the browser can tell "not in the
                        # study" apart from "nobody asked".
                        "in_study": None if in_study is None else e.slide_id in in_study,
                    }
                    for e in entries
                ],
            }
        )

    # Stepping order per scope. "in_study" walks only the slides the study admits, so prev/next and the
    # position counter agree with whatever the browser list was filtered to; the scope arrives in the
    # URL rather than being held server-side, so two windows can browse different scopes.
    scoped_order = {"all": order}
    if in_study is not None:
        scoped_order["in_study"] = [slide_id for slide_id in order if slide_id in in_study]

    def _walk(slide: str, scope: str) -> tuple[list[str], int]:
        """The order to step through and the slide's index in it, falling back to the full cohort when
        the requested scope does not contain this slide (a link to an excluded slide still opens)."""
        walk = scoped_order.get(scope, order)
        if slide not in walk:
            walk = order
        return walk, walk.index(slide)

    @app.get("/info")
    def info(slide: str = Query(...), scope: str = Query("all")) -> JSONResponse:
        opened = open_slide(slide)
        images = []
        for i, pyramid in enumerate(opened.pyramids):
            entry = pyramid.info()
            entry["index"] = i
            entry["name"] = pyramid.slide.name
            images.append(entry)
        # Position + neighbours, so stepping through the cohort needs no extra request.
        walk, position = _walk(slide, scope)
        entry = by_id[slide]
        return JSONResponse(
            {
                "slide_id": slide,
                "name": entry.name,
                "patient_id": entry.patient_id,
                "format": opened.reader.format,
                "primary_index": opened.reader.images.primary_index,
                "images": images,
                "mask": _mask_info(slide, opened),
                "mask_declared": entry.mask_path is not None,
                "attention": (
                    {
                        "n_tiles": found["n_tiles"],
                        "label": run_label,
                        "aggregated": run.aggregated,
                        "inners": run.inners_by_outer.get(run.outer_of.get(slide, ""), []),
                        "consistency": _finite(run.consistency.get(slide)),
                    }
                    if (found := slide_attention(slide)) is not None
                    else None
                ),
                "attention_available": run is not None,
                # This fold's own prediction, not the cross-replicate ensemble the aggregate reports.
                "prediction": (
                    {
                        "prob": found.prob,
                        "label": found.label,
                        "endpoint_type": found.endpoint_type,
                        "fold": found.fold,
                        "classes": {str(k): v for k, v in (run_classes or {}).items()},
                        "regression": (
                            {
                                "lo": run.regression.lo,
                                "hi": run.regression.hi,
                                "median_abs_error": run.regression.median_abs_error,
                                "pearson_r": run.regression.pearson_r,
                                "label_sd": run.regression.label_sd,
                                "n_slides": run.regression.n_slides,
                                "beats_fraction": run.error_percentile.get(slide),
                            }
                            if run.regression is not None
                            else None
                        ),
                        # The predicted survival curve on this fold's own bin edges, plus the cohort
                        # placement the unitless risk score needs to be readable.
                        "survival": (
                            {
                                "edges": found.survival.edges,
                                "surv": found.survival.surv,
                                "hazards": found.survival.hazards,
                                "tail": found.survival.tail,
                                "riskier_than": run.risk_percentile.get(slide),
                                "n_slides": run.survival.n_slides if run.survival else None,
                                "n_events": run.survival.n_events if run.survival else None,
                                "median_follow_up": run.survival.median_follow_up if run.survival else None,
                                "n_patients": run.survival.n_patients if run.survival else None,
                                # The cohort KM, so one patient's curve can be compared against it.
                                "km_times": run.survival.km_times if run.survival else [],
                                "km_surv": run.survival.km_surv if run.survival else [],
                            }
                            if found.survival is not None
                            else None
                        ),
                    }
                    if run and (found := run.predictions.get(slide)) is not None
                    else None
                ),
                "position": position,
                "total": len(walk),
                "prev_slide_id": walk[position - 1] if position > 0 else None,
                "next_slide_id": walk[position + 1] if position + 1 < len(walk) else None,
            }
        )

    @app.get("/tiles/{image}/{z}/{x}/{y}.{ext}")
    def tile(image: int, z: int, x: int, y: int, ext: str, slide: str = Query(...)) -> Response:
        media_type = _MEDIA_TYPES.get(ext.lower())
        if media_type is None:
            raise HTTPException(status_code=404, detail=f"unsupported extension: {ext}")
        opened = open_slide(slide)
        if not 0 <= image < len(opened.pyramids):
            raise HTTPException(status_code=404, detail=f"no image {image}")
        try:
            data = _slide_tile_bytes(opened.pyramids[image], z, x, y, tile_size, ext, jpeg_quality)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(content=data, media_type=media_type)

    # z/x/y are the slide's tile coordinates. The mask is resampled onto the slide's grid so both
    # layers share one grid, one origin and one zoom range.
    @app.get("/mask/{z}/{x}/{y}.png")
    def mask_tile(z: int, x: int, y: int, slide: str = Query(...)) -> Response:
        if slide not in by_id:
            raise HTTPException(status_code=404, detail=f"no slide {slide!r} in cohort {cohort_name!r}")
        mask = open_mask(slide)
        if mask is None:
            raise HTTPException(status_code=404, detail=f"no mask for slide {slide!r}")
        opened = open_slide(slide)
        slide_pyramid = opened.pyramids[opened.reader.images.primary_index]
        if not slide_pyramid.min_zoom <= z <= slide_pyramid.max_zoom:
            raise HTTPException(status_code=404, detail=f"zoom {z} out of range for this slide")
        try:
            data = _mask_tile_png(mask.pyramids[0], slide_pyramid, z, x, y, tile_size)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(content=data, media_type="image/png")

    @app.get("/attention.json")
    def attention_values(slide: str = Query(...)) -> JSONResponse:
        """Every tile's position and raw logit for one slide, so the client owns thresholding and colour."""
        if slide not in by_id:
            raise HTTPException(status_code=404, detail=f"no slide {slide!r} in cohort {cohort_name!r}")
        found = slide_attention(slide)
        if found is None:
            raise HTTPException(status_code=404, detail=f"no attention for slide {slide!r}")
        return JSONResponse(found)

    return app
