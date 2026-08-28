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
"""dlux, WSI/multimodal classification pipeline.

Runs before dlup is first imported (any ``dlux.*`` import triggers this), so it is the place to silence
dlup's optional-backend startup warning: dlux is fastslide-only, so the missing OpenSlide backend (we
don't install ``openslide-python`` and don't want it) is expected, and not something to warn about every run."""

import os
import sys
import warnings
from pathlib import Path

__version__ = "0.1.0"

warnings.filterwarnings("ignore", message="OpenSlide is not available", category=UserWarning)

# A consumer project points dlux at itself with DLUX_PROJECT: it supplies that project's config through
# the Hydra searchpath (see config/train.yaml) and its Python code -- the custom Task/model classes that
# `_target_` entries reference -- under $DLUX_PROJECT/src. Put that on the import path so those classes
# resolve, wherever dlux runs (the sweep venv or a clone). Appended, never prepended: an installed copy
# of the project (e.g. frozen into a sweep venv) is found first, so this only takes effect when the
# project is not installed, and it can never shadow the frozen snapshot a sweep depends on. When run in
# place, DLUX_PROJECT must be absolute (the cwd may not be the project root), same as the searchpath requires.
_dlux_project = os.environ.get("DLUX_PROJECT")
if _dlux_project:
    _dlux_project_src = str(Path(_dlux_project) / "src")
    if os.path.isdir(_dlux_project_src) and _dlux_project_src not in sys.path:
        sys.path.append(_dlux_project_src)
