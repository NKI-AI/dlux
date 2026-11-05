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
"""What a task's fit split fixed besides the model weights.

Every dlux task returns a ``FitState`` from ``fit_state()``. The training steps read it to return
predictions to raw units and to record the fold's fit-derived numbers in ``metadata.json``. A task sets
only the fields its endpoint has, and the defaults are the identity and "none".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class FitState:
    # Predictions are ``raw * pred_std + pred_mean``: a scalar for a z-scored regression head, one value per
    # gene for the expression head, the identity otherwise.
    pred_mean: float | np.ndarray = 0.0
    pred_std: float | np.ndarray = 1.0
    # Survival: the fold's interior discrete-time bin edges.
    time_edges: Optional[np.ndarray] = None
    # Classification: the loss weights, derived from the fold's fit split under `balanced`.
    pos_weight: Optional[float] = None
    class_weights: Optional[Sequence[float]] = None

    def recorded(self) -> dict:
        """The fields a run records in ``metadata.json``. A per-gene scale is recorded in the stream state,
        not here, so ``target_mean`` and ``target_std`` are set for a scalar scale only."""
        scalar = np.ndim(self.pred_mean) == 0
        return {
            "target_mean": float(self.pred_mean) if scalar else None,
            "target_std": float(self.pred_std) if scalar else None,
            "pos_weight": self.pos_weight,
            "class_weights": None if self.class_weights is None else [float(w) for w in self.class_weights],
            "time_edges": None if self.time_edges is None else [float(v) for v in self.time_edges],
        }
