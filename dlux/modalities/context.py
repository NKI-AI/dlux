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
"""What a modality may need to construct itself, gathered once by ``build_task``.

Each modality takes what it needs from this and ignores the rest, so adding a modality never changes
the code that builds them. ``fit_patient_codes`` is a callable rather than a list because resolving it
reads the DB: only a modality with fold-dependent statistics pays for it, and the leakage boundary
stays in ``build.py``, which is the one place allowed to read the ``fit`` split. ``recorded_state`` is
its mirror image, the same state, replayed from a run's record rather than fitted, which is how a
cohort with no fit split (an external one) can be scored at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


@dataclass(frozen=True)
class ModalityContext:
    cohort_name: str
    cohorts_dir: Optional[Path]  # cohort data root (feature cache, RNA matrix, gene panels)
    data_description: Any  # ahcore DataDescription, the tiling grid + manifest identity
    feature_extractor: Any  # the feature-extractor config node; a tile stream derives its identity from it
    fit_patient_codes: Callable[[], Sequence[str]]  # lazy: this fold's fit split, read on demand
    # Recorded fit-derived state to replay instead of fitting, keyed by stream name; returns None when
    # there is nothing recorded for that stream. A callable for the same reason as
    # ``fit_patient_codes``: only a modality with replayable state pays to look. Training leaves the
    # default (fit fresh); scoring a cohort with no fit split supplies the run's record.
    recorded_state: Callable[[str], Optional[dict]] = lambda _key: None
