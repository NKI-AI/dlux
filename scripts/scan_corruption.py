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
"""Scan a cohort's slides for unreadable tiles and rank them by corruption.

    scan_corruption --slides-csv <cohort>/sheets/slides.csv --image-dir <images> --out report.csv

A corrupt tile makes fastslide log a DATA_LOSS error to C-stderr and return a BLANK tile WITHOUT
raising — so the slide completes normally, writes its features, and leaves no error marker. Corruption
is therefore silent: a heavily-broken slide is indistinguishable from a healthy one downstream. This
tool re-reads every tile of every slide, counts fastslide's DATA_LOSS lines (captured off fd 2), and
writes a per-slide report (with ``file_id`` for the sheet blacklist) so genuinely-broken slides can be
excluded. Whole-slide tiling (no mask) so every tile is exercised regardless of tissue coverage.
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from dlup import ImageConfig, SlideDataset, TilingConfig
from dlup.utils.backends import ImageBackend

_DATA_LOSS = "Tile data empty or missing"


def scan_slide(image_path: str, mpp: float, tile_size: int) -> tuple[str, int, int, str]:
    """Read every tile of one slide; return (image_path, n_tiles, data_loss_count, error)."""
    cfg = ImageConfig(backend=ImageBackend["FASTSLIDE"], apply_color_profile=False)
    try:
        dataset = SlideDataset.from_standard_tiling(
            path=Path(image_path),
            tiling_config=TilingConfig(mpp=mpp, tile_size=(tile_size, tile_size), tile_overlap=(0, 0)),
            image_config=cfg,
        )
    except Exception as exc:  # a slide that won't even open is itself a defect worth reporting
        return image_path, -1, -1, f"open_failed: {exc}"

    n = len(dataset)
    with tempfile.TemporaryFile(mode="w+") as errf:  # fastslide writes DATA_LOSS to fd 2, not via Python
        saved = os.dup(2)
        os.dup2(errf.fileno(), 2)
        try:
            for i in range(n):
                np.asarray(dataset[i].image)  # force the decode -> triggers the read
        finally:
            os.dup2(saved, 2)
            os.close(saved)
        errf.seek(0)
        data_loss = errf.read().count(_DATA_LOSS)
    return image_path, n, data_loss, ""


def main() -> None:
    ap = argparse.ArgumentParser("scan_corruption")
    ap.add_argument("--slides-csv", type=Path, required=True, help="cohort slides.csv (needs an image_path column)")
    ap.add_argument("--image-dir", type=Path, required=True, help="root the image_path values are relative to")
    ap.add_argument("--out", type=Path, required=True, help="per-slide CSV report (sorted worst-first)")
    ap.add_argument("--mpp", type=float, default=2.0, help="tiling resolution to probe at (match extraction)")
    ap.add_argument("--tile-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=8, help="slides scanned in parallel")
    ap.add_argument("--threshold", type=float, default=5.0, help="print slides with DATA_LOSS %% at/above this")
    args = ap.parse_args()

    slides = pd.read_csv(args.slides_csv, dtype=str)
    if "image_path" not in slides.columns:
        raise SystemExit(f"[scan_corruption] {args.slides_csv} has no 'image_path' column")
    paths = [str(args.image_dir / p) for p in slides["image_path"]]
    print(f"[scan_corruption] scanning {len(paths)} slides at {args.mpp}mpp with {args.workers} workers")

    header = ["file_id", "image_path", "tiles", "data_loss", "pct", "error"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # Write each slide's result as it lands (flushed) so the report is live-peekable and survives a
    # crash; the file is re-written sorted worst-first once the scan completes.
    with open(args.out, "w", newline="") as fh, ProcessPoolExecutor(max_workers=args.workers) as pool:
        writer = csv.writer(fh)
        writer.writerow(header)
        fh.flush()
        futures = {pool.submit(scan_slide, p, args.mpp, args.tile_size): p for p in paths}
        for done, fut in enumerate(as_completed(futures), 1):
            image_path, n, data_loss, err = fut.result()
            pct = (100.0 * data_loss / n) if n > 0 else -1.0
            file_id = Path(image_path).name[:-4] if image_path.endswith(".svs") else Path(image_path).name
            row = (file_id, image_path, n, data_loss, pct, err)
            rows.append(row)
            writer.writerow(row)
            fh.flush()
            if done % 25 == 0:
                print(f"  scanned {done}/{len(paths)}", flush=True)

    rows.sort(key=lambda r: r[4], reverse=True)
    with open(args.out, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    flagged = [r for r in rows if r[4] >= args.threshold or r[2] < 0]
    print(
        f"[scan_corruption] wrote {args.out}: {len(rows)} slides; {len(flagged)} at/above {args.threshold}% DATA_LOSS:"
    )
    for file_id, _, tiles, _, pct, err in flagged:
        print(f"  {pct:5.1f}%  {tiles:6d} tiles  {file_id}  {err}")


if __name__ == "__main__":
    main()
