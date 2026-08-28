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
"""MLflow tracking, hard-baked (no Hydra ``_target_``).

dlux commits to mlflow as its single logger. Like ``tiling``/``dataset``/``paths``, it is a fixed type
constructed in code, not through a polymorphic ``_target_``. Everything is derived: the sqlite backend
and artifact store from ``paths.mlflow_dir``, the experiment from ``experiment_name`` (the sweep), and
the run name from ``split_version`` (the fold).

**Backend = sqlite** (`sqlite:///{mlflow_dir}/mlflow.db`): one indexed DB + a separate artifact dir,
which avoids the many-tiny-files / slow-UI / stale-NFS-handle issues of a file store. sqlite is
single-writer, but logging is not on the critical path (results flow through the persisted NPZ/metadata
and the aggregate stage), so a rare write-lock at worst drops a metric point.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from lightning.pytorch.loggers import MLFlowLogger


def mlflow_tracking_uri(mlflow_dir: str | Path) -> str:
    """sqlite backend-store URI under ``mlflow_dir`` (absolute path -> ``sqlite:////abs/...``)."""
    return f"sqlite:///{Path(mlflow_dir) / 'mlflow.db'}"


def build_mlflow_logger(
    *,
    mlflow_dir: str | Path,
    experiment_name: str,
    run_name: str,
    tags: Optional[Dict[str, str]] = None,
) -> "MLFlowLogger":
    """Construct the dlux MLflow logger: sqlite backend + a sibling artifact dir.

    ``experiment_name`` groups a sweep's folds; ``run_name`` (the split_version) is the per-fold run.
    The mlflow_dir is created if needed. The sqlite schema is auto-initialised by mlflow on first use.
    """
    from lightning.pytorch.loggers import MLFlowLogger

    # mlflow migrates its sqlite backend via alembic on first use, which logs the schema-migration
    # check at INFO on every run ("Context impl SQLiteImpl" / "Will assume non-transactional DDL").
    # It's benign boilerplate, keep it off the console.
    logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)

    mlflow_dir = Path(mlflow_dir)
    artifacts = mlflow_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)  # backend db is created by mlflow on first use

    # Constructing the logger imports mlflow, whose import calls logging.config.dictConfig and resets
    # the root logger to WARNING with a bare handler. That drops Hydra's INFO level and its file handler,
    # so every log record after this point vanishes from the run's <task>.log. Snapshot the root level
    # and handlers and restore them once the logger is built.
    root = logging.getLogger()
    saved_level, saved_handlers = root.level, root.handlers[:]
    try:
        return MLFlowLogger(
            experiment_name=experiment_name,
            run_name=run_name,
            tracking_uri=mlflow_tracking_uri(mlflow_dir),
            artifact_location=artifacts.as_uri(),
            tags=tags,
        )
    finally:
        root.setLevel(saved_level)
        root.handlers[:] = saved_handlers
