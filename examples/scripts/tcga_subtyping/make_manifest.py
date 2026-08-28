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
"""Generate the pinned GDC manifest + metadata sidecar for one center of the tcga_subtyping example.

    python make_manifest.py --center christiana --n-per-class 15 --out-dir manifests

Queries the public GDC REST API for TCGA-BRCA + TCGA-COAD diagnostic slide images from one tissue source
site, sorts by file id, and per project keeps the first `--n-per-class` files — or every available file
when `--n-per-class` is omitted (the full-cohort run). It then writes:

  - `<center>.txt`          — the gdc-client manifest (id / filename / md5 / size / state)
  - `<center>_metadata.tsv` — the sidecar build_sheets.py reads (file_uuid / file_name / patient_id /
                              project_id / tissue_source_site)

Both are public text (no pixels) and are COMMITTED as the source of truth: re-running may drift as GDC
updates, so the committed files are what makes the example byte-reproducible. Stdlib only. Needs internet.

The tissue-source-site names and filters were verified against the live GDC API (2026-08-26): both centers
carry ample BRCA and COAD diagnostic slides.
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from pathlib import Path

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
PROJECTS = ["TCGA-BRCA", "TCGA-COAD"]

# center slug -> GDC `cases.tissue_source_site.name` (verified against the live API 2026-08-26:
# Christiana Healthcare has 59 BRCA / 49 COAD diagnostic slides, MSKCC 47 / 37).
CENTER_TO_TSS = {
    "christiana": "Christiana Healthcare",
    "mskcc": "MSKCC",
}


def _query(tss_name: str) -> list[dict]:
    """All TCGA-BRCA/COAD diagnostic slide-image files for one tissue source site."""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": PROJECTS}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
            {"op": "in", "content": {"field": "experimental_strategy", "value": ["Diagnostic Slide"]}},
            {"op": "in", "content": {"field": "cases.tissue_source_site.name", "value": [tss_name]}},
        ],
    }
    body = {
        "filters": filters,
        "fields": ",".join(
            [
                "file_id",
                "file_name",
                "md5sum",
                "file_size",
                "cases.submitter_id",
                "cases.project.project_id",
                "cases.tissue_source_site.name",
            ]
        ),
        "format": "JSON",
        "size": "10000",
    }
    req = urllib.request.Request(
        GDC_FILES_ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (fixed, trusted GDC endpoint)
        payload = json.load(resp)
    return payload["data"]["hits"]


def _flatten(hit: dict) -> dict[str, str]:
    case = hit["cases"][0]
    return {
        "file_uuid": hit["file_id"],
        "file_name": hit["file_name"],
        "md5": hit["md5sum"],
        "size": str(hit["file_size"]),
        "patient_id": case["submitter_id"],
        "project_id": case["project"]["project_id"],
        "tissue_source_site": case["tissue_source_site"]["name"],
    }


def _select(rows: list[dict[str, str]], n_per_class: int | None) -> list[dict[str, str]]:
    """Sort by file_uuid, then per project take the first n_per_class (None = keep all)."""
    kept: list[dict[str, str]] = []
    for project in PROJECTS:
        of_project = sorted((r for r in rows if r["project_id"] == project), key=lambda r: r["file_uuid"])
        if n_per_class is not None and len(of_project) < n_per_class:
            print(f"WARNING: only {len(of_project)} {project} files available (< {n_per_class})")
        kept.extend(of_project if n_per_class is None else of_project[:n_per_class])
    return kept


def generate(center: str, n_per_class: int | None, out_dir: Path) -> None:
    tss_name = CENTER_TO_TSS[center]
    rows = [_flatten(h) for h in _query(tss_name)]
    selected = _select(rows, n_per_class)
    if not selected:
        raise SystemExit(f"no files selected for center {center!r} (tss {tss_name!r}) — verify the query")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{center}.txt", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "filename", "md5", "size", "state"])
        for r in selected:
            w.writerow([r["file_uuid"], r["file_name"], r["md5"], r["size"], "released"])
    with open(out_dir / f"{center}_metadata.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["file_uuid", "file_name", "patient_id", "project_id", "tissue_source_site"])
        for r in selected:
            w.writerow([r["file_uuid"], r["file_name"], r["patient_id"], r["project_id"], r["tissue_source_site"]])

    per = {p: sum(1 for r in selected if r["project_id"] == p) for p in PROJECTS}
    print(f"{center}: wrote {len(selected)} files ({per}) to {out_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the pinned GDC manifest + sidecar for one center.")
    p.add_argument("--center", choices=sorted(CENTER_TO_TSS), required=True)
    p.add_argument(
        "--n-per-class",
        type=int,
        default=None,
        help="files per project (BRCA/COAD) to pin; omit to take every available file",
    )
    p.add_argument("--out-dir", type=Path, default=Path("manifests"))
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate(args.center, args.n_per_class, args.out_dir)
