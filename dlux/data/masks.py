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
"""Native tissue/background mask generation from a TorchScript segmenter.

The model is the public HuggingFace ``NKI-AI/tissue-bg-all-stains`` ``.pack`` (a zip of a TorchScript
``model.pt`` + an ``model_config.xml`` giving mpp / tile size / normalisation). We run it tile-by-tile
via dlup and stitch to a pyramidal ``<stem>.mask.tiff``. It runs on dlux's already-bundled WSI machinery
(ahcore zarr I/O + dlup tiling/TIFF), so the public dlux release carries no extra vendored library,
only the model (fetched from HF) and this glue we own.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ahcore.readers import StitchingMode, ZarrFileImageReader
from ahcore.utils.io import get_logger
from ahcore.writers import DataFormat, ZarrFileImageWriter
from dlup import ImageConfig, SlideDataset, SlideImage, TilingConfig
from dlup.tiling import Grid
from dlup.writers import TiffCompression, TifffileImageWriter

logger = get_logger(__name__)


@contextlib.contextmanager
def _suppress_c_stderr(enabled: bool = True):
    """Redirect the C-level stderr (fd 2) to /dev/null. fastslide logs corrupt-tile reads (DATA_LOSS)
    straight to fd 2 from C++, which a Python ``try/except`` cannot silence. The message is emitted
    before the exception is raised. Python-level errors still propagate as exceptions; only the C++
    stderr chatter is dropped. Restores fd 2 on exit. No-op when ``enabled`` is False (for debugging)."""
    if not enabled:
        yield
        return
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


@dataclass(frozen=True)
class SegConfig:
    """The bits of the ``.pack``'s ``model_config.xml`` inference needs: the resolution to tile at, the
    tile geometry, and the per-channel input normalisation."""

    mpp: float
    tile_size: tuple[int, int]
    tile_overlap: tuple[int, int]
    mean: list[float]
    std: list[float]


def parse_seg_config(xml_content: str) -> SegConfig:
    """Parse the ``.pack``'s ``model_config.xml`` (root ``AifoModelConfiguration``) into a SegConfig."""
    root = ET.fromstring(xml_content)
    if root.tag != "AifoModelConfiguration":
        raise ValueError(f"unexpected config root '{root.tag}', expected 'AifoModelConfiguration'")

    def _text(element: ET.Element | None, tag: str) -> str:
        child = element.find(tag) if element is not None else None
        if child is None or child.text is None:
            raise ValueError(f"model_config.xml missing <{tag}>")
        return child.text.strip()

    def _wh(tag: str) -> tuple[int, int]:
        elem = root.find(tag)
        return int(_text(elem, "Width")), int(_text(elem, "Height"))

    def _channels(parent: ET.Element, tag: str) -> list[float]:
        elem = parent.find(tag)
        if elem is None:
            raise ValueError(f"model_config.xml missing Normalization/{tag}")
        return [float(c.text) for c in elem if c.tag.startswith("Channel") and c.text is not None]

    norm = root.find("Normalization")
    if norm is None:
        raise ValueError("model_config.xml missing <Normalization>")
    return SegConfig(
        mpp=float(_text(root, "Mpp")),
        tile_size=_wh("TileSize"),
        tile_overlap=_wh("TileOverlap"),
        mean=_channels(norm, "Mean"),
        std=_channels(norm, "Std"),
    )


def load_pack(pack_path: Path | str, device: torch.device) -> tuple[torch.jit.ScriptModule, SegConfig]:
    """Load a ``.pack`` (zip of TorchScript ``model.pt`` + ``model_config.xml``) -> (eval model, SegConfig)."""
    with zipfile.ZipFile(pack_path, "r") as zf:
        names = set(zf.namelist())
        for required in ("model.pt", "model_config.xml"):
            if required not in names:
                raise KeyError(f"{required} not in {pack_path} (has: {sorted(names)})")
        model = torch.jit.load(io.BytesIO(zf.read("model.pt")), map_location=device)
        model.eval()
        config = parse_seg_config(zf.read("model_config.xml").decode("utf-8"))
    return model, config


def _write_overlay(
    slide_image: SlideImage, mask_file: Path, thumbnail_file: Path, *, max_dim: int = 4096, quiet: bool = True
) -> None:
    """Write a low-res QC overlay PNG: the tissue mask (blue, alpha) composited over the slide thumbnail.
    Needs only the mask and a low-res slide read (no model), so it also backfills thumbnails. The
    low-res read can itself hit corrupt tiles, so it too runs under the C-stderr suppression."""
    import imageio.v3 as iio
    import PIL.Image

    full_size = slide_image.size
    scaling = max_dim / max(full_size)
    scaled_size = (int(full_size[0] * scaling), int(full_size[1] * scaling))
    with _suppress_c_stderr(quiet):
        image = slide_image.read_region((0, 0), scaling, scaled_size).to_numpy()[:, :, :3]

    mask = iio.imread(mask_file)
    seg = mask[..., 0] if mask.ndim == 3 else mask
    lut = np.array([[0, 0, 0], [0, 0, 255]], dtype=np.uint8)  # background -> black (transparent), tissue -> blue
    colored = lut[np.clip(seg, 0, len(lut) - 1)]

    base = PIL.Image.fromarray(image).convert("RGBA")
    overlay = np.array(PIL.Image.fromarray(colored).convert("RGBA").resize(base.size))
    overlay[:, :, 3] = np.where(np.any(overlay[:, :, :3] != 0, axis=2), 90, 0).astype(np.uint8)  # 90 = overlay alpha
    Path(thumbnail_file).parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.alpha_composite(base, PIL.Image.fromarray(overlay, "RGBA")).convert("RGB").save(thumbnail_file)


def write_overlay(
    image_file: Path, mask_file: Path, thumbnail_file: Path, *, reader: str = "FASTSLIDE", quiet: bool = True
) -> None:
    """Backfill a QC thumbnail from an already-written ``.mask.tiff``, opens the slide standalone (no
    model, no inference), so it runs over a directory of masks that lack thumbnails."""
    _write_overlay(SlideImage.from_file_path(Path(image_file), backend=reader), mask_file, thumbnail_file, quiet=quiet)


@torch.no_grad()
def segment_slide(
    model: torch.jit.ScriptModule,
    config: SegConfig,
    image_file: Path,
    output_file: Path,
    *,
    device: torch.device,
    reader: str = "FASTSLIDE",
    thumbnail_file: Path | None = None,
    quiet_reader: bool = True,
) -> int:
    """Tile ``image_file`` at the model's mpp, run the segmenter per tile (argmax over classes), and
    stitch the label map into a pyramidal ``output_file`` (``.mask.tiff``, ZSTD). Overlapping tiles are
    resolved through an intermediate zarr (crop stitching). If ``thumbnail_file`` is given, also write a
    low-res QC overlay PNG. Returns the number of unreadable
    (corrupt) tiles that were marked background."""
    dataset = SlideDataset.from_standard_tiling(
        Path(image_file),
        image_config=ImageConfig(backend=reader, apply_color_profile=False),
        tiling_config=TilingConfig(
            mpp=config.mpp,
            tile_size=config.tile_size,
            tile_overlap=config.tile_overlap,
            random_sample_in_grid=False,
            tile_mode="overflow",
            grid_order="C",
            limit_bounds=True,
        ),
    )
    slide_image = dataset.slide_image
    scaled_view = slide_image.get_view_at_scaling(slide_image.get_scaling(config.mpp))
    mean = torch.tensor(config.mean, device=device).view(1, len(config.mean), 1, 1)
    std = torch.tensor(config.std, device=device).view(1, len(config.std), 1, 1)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        zarr_path = Path(tmpdir) / "seg.zarr"
        zarr_writer = ZarrFileImageWriter(
            zarr_path,
            size=scaled_view.size,
            mpp=config.mpp,
            tile_size=config.tile_size,
            tile_overlap=config.tile_overlap,
            num_samples=len(dataset),
            data_format=DataFormat.IMAGE,
        )

        # A few tiles in some TCGA .svs files have corrupt/missing data (DATA_LOSS, usually edge tiles).
        # Treat an unreadable tile as background rather than failing the whole slide.
        background = np.zeros((1, 1, config.tile_size[1], config.tile_size[0]), dtype="uint8")
        failed = 0

        def _tiles():
            nonlocal failed
            for sample in dataset:
                coordinates = np.asarray(sample.coordinates)[np.newaxis, ...]
                try:
                    image = sample.image.to_numpy()[:, :, :3]
                except Exception:  # noqa: BLE001, corrupt tile data: mark background, keep the slide going
                    failed += 1
                    yield coordinates, background
                    continue
                data = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                data = (data - mean) / std
                segmentation = model(data).cpu()[0].argmax(dim=0).numpy().astype("uint8")
                yield coordinates, segmentation[np.newaxis, np.newaxis, ...]

        with _suppress_c_stderr(quiet_reader):
            zarr_writer.consume(_tiles())
        if failed:
            logger.warning(
                "%s: %d/%d tiles unreadable (corrupt slide data) -> marked background",
                Path(image_file).name,
                failed,
                len(dataset),
            )

        zarr_reader = ZarrFileImageReader(zarr_path, stitching_mode=StitchingMode.CROP)
        grid = Grid.from_tiling(
            (0, 0), zarr_reader.size, tile_size=config.tile_size, tile_overlap=(0, 0), mode="overflow", order="C"
        )

        def _regions():
            for coordinates in grid:
                yield zarr_reader.read_region(coordinates, 0, config.tile_size).astype(np.uint8)

        TifffileImageWriter(
            Path(output_file),
            size=zarr_reader.size,
            mpp=zarr_reader.mpp,
            tile_size=config.tile_size,
            pyramid=True,
            compression=TiffCompression.ZSTD,
        ).from_tiles_iterator(_regions())

    if thumbnail_file is not None:
        _write_overlay(slide_image, output_file, thumbnail_file, quiet=quiet_reader)
    return failed
