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
"""OmegaConf resolvers shared by the training entry points.

Registered by calling :func:`register_resolvers`, which every bin whose config uses one must do
before ``@hydra.main`` runs.
"""

from __future__ import annotations

import os

from omegaconf import OmegaConf


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
