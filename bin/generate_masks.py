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
"""Generate tissue/background masks for a cohort's slides.

    generate_masks cohort=tcga_brca_coad_christiana output_dir=/path/to/masks

Reads the cohort's ``sheets/slides.csv`` (masks are produced before build_db), runs the HuggingFace
tissue segmenter (``NKI-AI/tissue-bg-all-stains``, auto-downloaded + cached) over each slide, and writes
``<output_dir>/<mask name>.mask.tiff``. A local slide's mask mirrors its relative path, which is unique
per cohort, so two slides of the same basename cannot map to one file; a remote slide's URL is collapsed
to a flat ``slidescore_<id>`` key.

Each mask is recorded in ``sheets/masks.csv`` (``image_path,mask_path``), which this binary owns and
``build_db`` joins on. Point the cohort's ``mask_dir`` at ``output_dir`` to pick them up. A cohort that
ships its own masks writes that sheet itself and never runs this. Resumable: a slide whose recorded
mask exists is skipped, so masks written under any convention are left alone. GPU job.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from ahcore.utils.io import get_logger, print_config
from ahcore.utils.remote import is_remote_reader, slide_cache_key

from dlux.config.cohort import Cohort
from dlux.config.resolvers import register_resolvers
from dlux.data import sheets
from dlux.data.masks import load_pack, segment_slide, write_overlay
from dlux.data.sheets import IMAGE_PATH, read_slides
from dlux.data.slidescore import export_slidescore_key

logger = get_logger(__name__)


def _resolve_pack(cfg: DictConfig) -> Path:
    """Explicit ``pack_path`` if set, else download the .pack from HuggingFace (into HuggingFace's own cache)."""
    if cfg.mask_generator.get("pack_path"):
        return Path(cfg.mask_generator.pack_path)
    from huggingface_hub import hf_hub_download

    logger.info("fetching %s from HF %s", cfg.mask_generator.hf_filename, cfg.mask_generator.hf_repo_id)
    return Path(
        hf_hub_download(
            repo_id=str(cfg.mask_generator.hf_repo_id),
            filename=str(cfg.mask_generator.hf_filename),
        )
    )


register_resolvers()


@hydra.main(version_base=None, config_path="../config", config_name="generate_masks")
def main(cfg: DictConfig) -> None:
    print_config(cfg, fields=("cohort", "mask_generator", "paths", "output_dir", "overwrite"))

    cohort = Cohort(**OmegaConf.to_container(cfg.cohort, resolve=True))  # type: ignore[arg-type]
    # Segmenting a remote cohort streams its tiles over HTTP, so the token must be exported first.
    export_slidescore_key(cohort, cfg)
    image_dir = Path(cohort.storage.image_dir)
    slides_csv = Path(cfg.paths.cohorts_dir) / cohort.name / "sheets" / "slides.csv"
    if not slides_csv.exists():
        sys.exit(f"[generate_masks] slides sheet not found: {slides_csv} (build the cohort sheets first).")
    output_dir = Path(cfg.output_dir)
    overwrite = bool(cfg.get("overwrite", False))

    # Unique image paths (a patient may have several slides; the sheet is per-slide).
    image_paths = list(dict.fromkeys(read_slides(slides_csv)[IMAGE_PATH].astype(str)))

    device_str = cfg.mask_generator.get("device")
    if device_str in (None, "auto"):
        device = torch.accelerator.current_accelerator() or torch.device("cpu")
    else:
        device = torch.device(device_str)
    model, seg_config = load_pack(_resolve_pack(cfg), device)
    logger.info(
        "segmenter loaded (mpp=%s, tile=%s) on %s; %d slides -> %s",
        seg_config.mpp,
        seg_config.tile_size,
        device,
        len(image_paths),
        output_dir,
    )

    reader = cohort.storage.default_reader
    remote = is_remote_reader(reader)
    thumbnails = bool(cfg.mask_generator.thumbnail)
    quiet_reader = bool(cfg.mask_generator.get("quiet_reader", True))

    def slide_source(rel: str) -> str:
        """The slide address dlup opens: a remote rel is already a full URL; a local rel joins image_dir."""
        return rel if remote else str(image_dir / rel)

    # masks.csv records where each slide's mask actually is, so a mask written under any convention
    # (a cohort's own, or this generator's) is recognised and not regenerated.
    masks_csv = slides_csv.with_name(sheets.MASKS_SHEET)
    recorded = sheets.read_masks(masks_csv)

    def mask_path_for(rel: str) -> Path:
        """Where a slide's mask is, or will be. A local rel names its own relative path, unique per
        cohort; a remote rel is a URL, collapsed to the flat ``slidescore_<id>`` key the feature store
        uses, because a URL cannot be a filename."""
        return output_dir / recorded[rel] if rel in recorded else output_dir / f"{slide_cache_key(rel)}.mask.tiff"

    def thumb_path_for(mask_out: Path) -> Path:
        """The QC thumbnail beside a mask, named off the same base — ``<base>.mask.tiff`` pairs with
        ``<base>.thumbnail.png`` whatever convention produced the mask."""
        name = mask_out.name
        base = name[: -len(".mask.tiff")] if name.endswith(".mask.tiff") else mask_out.stem
        return mask_out.with_name(f"{base}.thumbnail.png")

    # Upfront breakdown (slides are processed in sheet order, no shuffle) so a resumed run's work is
    # visible: masks already done are skipped. If a mask exists but its thumbnail doesn't, only the
    # thumbnail is backfilled (no inference).
    have_mask = [rel for rel in image_paths if mask_path_for(rel).exists()]
    need_thumb = sum(1 for rel in have_mask if not thumb_path_for(mask_path_for(rel)).exists()) if thumbnails else 0
    logger.info(
        "%d slides in sheet order | %d already have a mask (%d need a thumbnail backfill) | %d to generate%s",
        len(image_paths),
        len(have_mask),
        need_thumb,
        len(image_paths) - len(have_mask),
        " | overwrite=true: regenerating all" if overwrite else "",
    )

    n_done = n_thumb = n_skipped = 0
    failures: list[str] = []
    unreadable: dict[str, int] = {}  # slide -> #corrupt tiles marked background (this run)

    def _process(rel: str) -> tuple[str, str, object]:
        """One slide: generate its mask, backfill a missing thumbnail, or skip. Returns a tagged result
        and mutates no shared state, so the pool needs no locks — the caller aggregates."""
        mask_out = mask_path_for(rel)
        thumb_out = thumb_path_for(mask_out)
        mask_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            if mask_out.exists() and not overwrite:
                # Mask already present: backfill a missing QC thumbnail (no inference), else skip.
                if thumbnails and not thumb_out.exists():
                    write_overlay(slide_source(rel), mask_out, thumb_out, reader=reader, quiet=quiet_reader)
                    return "thumb", rel, None
                return "skip", rel, None
            n_failed = segment_slide(
                model,
                seg_config,
                slide_source(rel),
                mask_out,
                device=device,
                reader=reader,
                thumbnail_file=thumb_out if thumbnails else None,
                quiet_reader=quiet_reader,
            )
            return "done", rel, n_failed
        except Exception as exc:  # noqa: BLE001 — per-slide failures are logged, not fatal
            return "fail", rel, exc

    # A remote slide's cost is its HTTP fetch (~1s/tile), and one slide's backend cannot parallelize its
    # own reads (a single asyncio loop) — but each slide has its own backend, so running slides
    # concurrently overlaps the fetches. Local slides read from disk fast and default to one worker.
    slide_workers = cfg.mask_generator.get("slide_workers")
    slide_workers = int(slide_workers) if slide_workers else (12 if remote else 1)
    with ThreadPoolExecutor(max_workers=slide_workers) as pool:
        futures = [pool.submit(_process, rel) for rel in image_paths]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="masks", unit="slide"):
            kind, rel, payload = fut.result()
            if kind == "thumb":
                n_thumb += 1
            elif kind == "skip":
                n_skipped += 1
            elif kind == "fail":
                failures.append(rel)
                logger.error("FAILED %s: %s", rel, payload)
            else:
                if payload:
                    unreadable[rel] = payload
                n_done += 1

    logger.info(
        "done: generated=%d thumbnails_backfilled=%d skipped=%d failed=%d -> %s",
        n_done,
        n_thumb,
        n_skipped,
        len(failures),
        output_dir,
    )

    # This generator owns masks.csv: producing a mask records it here; the cohort's slides.csv is never
    # rewritten. Rows already present are kept, so a cohort that ships its own masks keeps its paths and
    # a resumed run only adds what it made.
    written = {
        rel: str(mask_path_for(rel).relative_to(output_dir)) for rel in image_paths if mask_path_for(rel).exists()
    }
    if written:
        rows = pd.DataFrame(sorted(written.items()), columns=[sheets.IMAGE_PATH, sheets.MASK_PATH])
        rows.to_csv(masks_csv, index=False)
        logger.info("%d masks recorded -> %s", len(rows), masks_csv)

    # Audit trail of slides with corrupt/unreadable tiles (marked background) — accumulated across runs
    # (a resumed run only re-segments new slides, so we merge rather than overwrite). Format: "<image>\t<n>".
    audit_path = output_dir / "unreadable_slides.txt"
    if unreadable or audit_path.exists():
        merged: dict[str, str] = {}
        if audit_path.exists():
            for line in audit_path.read_text().splitlines():
                if "\t" in line:
                    slide, count = line.rsplit("\t", 1)
                    merged[slide] = count
        merged.update({slide: str(count) for slide, count in unreadable.items()})
        audit_path.write_text("".join(f"{slide}\t{count}\n" for slide, count in sorted(merged.items())))
        logger.info("%d slides with unreadable tiles (%d this run) -> %s", len(merged), len(unreadable), audit_path)

    if failures:
        logger.error("failed slides:\n%s", "\n".join(failures))
        sys.exit(2)


if __name__ == "__main__":
    main()
