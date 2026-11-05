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
"""OmegaConf resolvers shared by the entry points.

Registered by calling :func:`register_resolvers`, which every bin must do before ``@hydra.main`` runs:
every entry config resolves its search path through ``dlux.project``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


def _from_repo_root(path: str) -> Path:
    # `bazel run` executes in the runfiles tree and names the checkout in BUILD_WORKSPACE_DIRECTORY.
    # Every other entry point runs from the repository root.
    root = os.environ.get("BUILD_WORKSPACE_DIRECTORY") or os.getcwd()
    return (Path(root) / path).resolve()


def repo_path(path: str) -> str:
    """``path`` as an absolute path, a relative one taken from the repository root.

    Raises ``FileNotFoundError`` naming the resolved location if nothing is there.
    """
    resolved = _from_repo_root(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{path!r} resolves to {resolved}, which does not exist")
    return str(resolved)


def project_dir() -> str:
    """The consumer project named by ``DLUX_PROJECT``, as an absolute path.

    A relative value is taken from the repository root. The project must hold both search-path roots,
    ``catalog/`` and ``config/``. Raises with the value and its resolved location otherwise.
    """
    value = os.environ.get("DLUX_PROJECT")
    if not value:
        raise RuntimeError(
            "DLUX_PROJECT is not set. Point it at a consumer project holding catalog/ and config/, "
            "e.g. DLUX_PROJECT=examples."
        )
    resolved = _from_repo_root(value)
    missing = [name for name in ("catalog", "config") if not (resolved / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"DLUX_PROJECT={value!r} resolves to {resolved}, which has no {' and no '.join(f'{m}/' for m in missing)}"
        )
    return str(resolved)


def run_key(split_version: Optional[str], field: str, fold: Optional[int]) -> str:
    """The fold's name for the hydra scratch path, resolved at compose time.

    Hydra builds ``hydra.run.dir`` before ``main`` runs, so the split name that a bin derives from
    ``fold`` does not exist yet and cannot be interpolated into the path. ``fold`` and ``field`` are
    known at compose time, so this name is resolvable.

    Only names a disposable directory. Collisions are prevented by ``claim_hydra_run_dir``'s O_EXCL
    marker, not this string.
    """
    if split_version is not None:
        return str(split_version)
    return f"{field}_fold{fold}"


def auto_workers() -> int:
    """Dataloader workers: allocated cores minus one for the main process.

    On Linux ``sched_getaffinity`` reads the cgroup the scheduler placed us in, so it reports the job's
    own allocation rather than the machine's core count. It does not exist on macOS, which falls back to
    the machine core count.
    """
    if hasattr(os, "sched_getaffinity"):
        cores = len(os.sched_getaffinity(0))
    else:
        cores = os.cpu_count() or 1
    return max(1, cores - 1)


def register_resolvers() -> None:
    """Idempotent: bins may be imported more than once in a session."""
    OmegaConf.register_new_resolver("dlux.auto_workers", auto_workers, use_cache=True, replace=True)
    OmegaConf.register_new_resolver("dlux.project", project_dir, replace=True)
    OmegaConf.register_new_resolver("dlux.repo_path", repo_path, replace=True)
    OmegaConf.register_new_resolver("dlux.run_key", run_key, replace=True)
