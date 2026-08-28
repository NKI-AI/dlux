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
"""Build the tcga_subtyping example sheets for ONE center from a completed GDC download.

    python build_sheets.py --gdc-dir <example_data_root>/christiana/images \
        --metadata manifests/christiana_metadata.tsv --out-dir <example_data_root>/christiana/images/../sheets

`gdc-client download -m manifests/<center>.txt -d <gdc-dir>` lays one directory per file UUID under
`<gdc-dir>`, each holding the slide (`<gdc-dir>/<file_uuid>/<file_name>.svs`). This reads that tree plus the
committed metadata sidecar (file UUID -> patient -> project) and emits the rigid `patients.csv` + `slides.csv`
that `dlux build_db` ingests. Stdlib only, offline — no live GDC query. Masks are NOT built here; run
`dlux generate_masks` after.

Image paths in `slides.csv` are relative to the cohort's `image_dir`, which is `<gdc-dir>` itself.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# project_id (GDC) -> the raw cancer_type label the cohort contract maps ({BRCA: 0, COAD: 1}).
_PROJECT_TO_LABEL = {"TCGA-BRCA": "BRCA", "TCGA-COAD": "COAD"}


def _read_metadata(path: Path) -> dict[str, dict[str, str]]:
    """file_uuid -> {file_name, patient_id, cancer_type} from the committed sidecar TSV."""
    rows: dict[str, dict[str, str]] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            project = row["project_id"]
            if project not in _PROJECT_TO_LABEL:
                raise SystemExit(f"unexpected project_id {project!r} (expected one of {sorted(_PROJECT_TO_LABEL)})")
            rows[row["file_uuid"]] = {
                "file_name": row["file_name"],
                "patient_id": row["patient_id"],
                "cancer_type": _PROJECT_TO_LABEL[project],
            }
    if not rows:
        raise SystemExit(f"empty metadata sidecar: {path}")
    return rows


def build(gdc_dir: Path, metadata: Path, out_dir: Path) -> None:
    meta = _read_metadata(metadata)

    patients: dict[str, str] = {}  # patient_id -> cancer_type
    slides: list[tuple[str, str]] = []  # (patient_id, image_path relative to gdc_dir)
    missing = []
    for file_uuid, info in meta.items():
        slide = gdc_dir / file_uuid / info["file_name"]
        if not slide.is_file():
            missing.append(file_uuid)
            continue
        patient, label = info["patient_id"], info["cancer_type"]
        if patients.get(patient, label) != label:
            raise SystemExit(f"patient {patient} has conflicting labels {patients[patient]} / {label}")
        patients[patient] = label
        slides.append((patient, f"{file_uuid}/{info['file_name']}"))

    if missing:
        print(f"WARNING: {len(missing)} manifest file(s) not found under {gdc_dir} (download incomplete?)")
    if not slides:
        raise SystemExit(f"no downloaded slides found under {gdc_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "patients.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "cancer_type"])
        for patient in sorted(patients):
            w.writerow([patient, patients[patient]])
    with open(out_dir / "slides.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "image_path"])
        for patient, image_path in sorted(slides):
            w.writerow([patient, image_path])

    print(f"wrote {len(patients)} patients / {len(slides)} slides to {out_dir}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build tcga_subtyping example sheets from a GDC download.")
    p.add_argument("--gdc-dir", type=Path, required=True, help="dir gdc-client downloaded into (the cohort image_dir)")
    p.add_argument("--metadata", type=Path, required=True, help="committed metadata sidecar TSV for this center")
    p.add_argument(
        "--out-dir", type=Path, required=True, help="where to write patients.csv + slides.csv (a sheets/ dir)"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build(args.gdc_dir, args.metadata, args.out_dir)
