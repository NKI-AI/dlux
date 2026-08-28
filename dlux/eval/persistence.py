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
"""Persistent-output helpers for the training binary, stdlib only (no ahcore/torch).

Hydra run dirs are treated as always disposable: training writes there, and we then
copy the artifacts we care about into a stable, canonical dir keyed by
``(experiment_name, split_version)`` under ``studies/<study>/runs/``. Downstream aggregation reads
only the canonical dirs.

Three safeguards against the timestamp-clash bug (two runs resolving to the same
hydra dir and silently clobbering each other):

1. **Namespacing**, the hydra ``run.dir`` includes ``experiment_name`` + ``split_version``
   (configured in ``config/hydra``), so different folds cannot collide by construction.
2. **Active collision detection**, ``claim_hydra_run_dir`` atomically creates a
   ``run_marker.json`` (O_EXCL) in the hydra dir at startup; a pre-existing marker means
   another run already owns this dir, and we raise loudly instead of clobbering.
3. **Canonical-dir sentinel**, ``metadata.json`` is written last into the canonical dir
   and gates idempotent re-runs (``refuse_if_run_dir_already_persisted``).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

logger = logging.getLogger(__name__)

_MARKER_NAME = "run_marker.json"
_SENTINEL_NAME = "metadata.json"


def persistent_run_dir(
    studies_dir: str | Path, study: str, experiment_name: str, cohort: str, split_version: str
) -> Path:
    """Canonical run dir: ``studies/<study>/runs/<experiment>/<cohort>/<split_version>/``.

    Study groups all its experiments. An experiment (a config variant) is a single deletable
    subtree spanning its cohorts. Cohort is the development cohort the models trained on."""
    parts = {"study": study, "experiment_name": experiment_name, "cohort": cohort, "split_version": split_version}
    for label, value in parts.items():
        if not value:
            raise ValueError(f"{label} is required to compute the persistent run directory.")
    return Path(studies_dir) / study / "runs" / experiment_name / cohort / split_version


def run_already_persisted(run_dir: str | Path) -> bool:
    """True if a completed run already persisted to ``run_dir``.

    ``metadata.json`` is the sentinel (written last by ``persist_run_artifacts``). Callers
    (e.g. the train bin) use this to skip already-done folds cleanly in a resumable sweep,
    rather than raising, a benign "already done" is not an error."""
    return (Path(run_dir) / _SENTINEL_NAME).exists()


def claim_hydra_run_dir(output_dir: str | Path, *, experiment_name: str, split_version: str) -> Dict[str, Any]:
    """Atomically stake a claim on the (disposable) hydra output dir; detect collisions.

    Writes ``run_marker.json`` via an O_EXCL create. If it already exists, another run
    resolved to the same output_dir (a timestamp clash despite namespacing), raise loudly
    rather than let the two runs clobber each other's checkpoints/predictions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    marker_path = output_dir / _MARKER_NAME
    marker = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "experiment_name": experiment_name,
        "split_version": split_version,
        "created": datetime.now(timezone.utc).isoformat(),
    }
    try:
        # "x" == exclusive create: atomic, races are impossible.
        with open(marker_path, "x") as f:
            json.dump(marker, f, indent=2)
    except FileExistsError:
        try:
            other = json.loads(marker_path.read_text())
        except (OSError, json.JSONDecodeError):
            other = {}
        raise RuntimeError(
            f"Hydra run dir collision at {output_dir}: already claimed by "
            f"pid={other.get('pid')} host={other.get('host')} "
            f"experiment={other.get('experiment_name')} split={other.get('split_version')}. "
            f"Two runs resolved to the same output_dir (timestamp clash) — check the "
            f"hydra run.dir namespacing (experiment_name/split_version must be in the path)."
        ) from None
    return marker


def get_git_sha(cwd: str | Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL).decode().strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def persist_run_artifacts(
    *,
    run_dir: str | Path,
    artifacts: Mapping[str, str | Path],
    metadata: Mapping[str, Any],
    force: bool,
) -> None:
    """Copy ``artifacts`` into the canonical ``run_dir`` and write ``metadata.json`` last.

    ``artifacts`` maps destination filename -> source path (in the disposable hydra dir).
    Missing sources are warned about, not fatal (the run dir will be flagged incomplete via
    the log). ``metadata.json`` is written last so its presence means "run fully persisted".
    """
    run_dir = Path(run_dir)
    if force and run_dir.exists():
        logger.info("+force=true: clearing existing %s before persisting fresh artifacts.", run_dir)
        shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    for dest_name, src in artifacts.items():
        src_path = Path(src)
        if src_path.exists():
            shutil.copy2(src_path, run_dir / dest_name)
            logger.info("Persisted %s -> %s", dest_name, run_dir / dest_name)
        else:
            logger.warning("Missing artifact %s at %s; persistent run dir will be incomplete.", dest_name, src_path)

    # Sentinel, written last.
    (run_dir / _SENTINEL_NAME).write_text(json.dumps(dict(metadata), indent=2))
    logger.info("Persisted run metadata -> %s", run_dir / _SENTINEL_NAME)
