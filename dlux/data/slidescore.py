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
"""Auth for cohorts streamed from SlideScore.

Every entry point that opens a slide (build_db's geometry probe, extract_features, generate_masks)
calls ``export_slidescore_key`` once at startup. For a local cohort it is a no-op; for a remote one it
exports the API token so dlup can authenticate. Each remote cohort is one SlideScore study with its
own token, so the token path is a per-cohort fact read from
``paths.cohorts.<cohort>.slidescore_api_key_path``. The token FILE lives outside the repo; only its
path is configured.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from ahcore.utils.remote import is_remote_reader
from dlup.backends.slidescore_backend import export_api_key

from dlux.config.cohort import Cohort


def export_slidescore_key(cohort: Cohort, cfg: DictConfig) -> None:
    """Export the SlideScore API token when ``cohort`` streams its slides; no-op otherwise.

    Reads the token path from ``paths.cohorts.<cohort>.slidescore_api_key_path``. Raises if a remote
    cohort has no token path configured — a remote cohort cannot open a single slide without it, so
    failing here beats an opaque auth error deep in dlup.
    """
    if not is_remote_reader(cohort.storage.default_reader):
        return
    api_key_path = OmegaConf.select(cfg, f"paths.cohorts.{cohort.name}.slidescore_api_key_path")
    if not api_key_path:
        raise SystemExit(
            f"[slidescore] cohort '{cohort.name}' uses reader {cohort.storage.default_reader} but "
            f"paths.cohorts.{cohort.name}.slidescore_api_key_path is not set — point it at your "
            "SlideScore token file."
        )
    export_api_key(str(api_key_path))
