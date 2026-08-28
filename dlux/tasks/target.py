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
"""The single raw -> modeled-target seam (fed to the PatientLabelAdapter).

All target transforms (``categorical`` / ``binarize`` / ``numeric``) are applied through the
*same* mechanism the categorical path already uses (the adapter's field-transform), constructed
here from the contract field (+ build-db-resolved statistics for binarize/numeric). This is where
the objective-aware missing sentinel is also housed, so there is exactly one place a raw value
becomes a modeled label. See ``docs/specs/CONTRACT_SPEC.md`` ("Transform statistics").
"""

from __future__ import annotations

import bisect
import json
import math
from pathlib import Path
from typing import Any, Callable, Optional

from dlux.config.cohort import ContractField, Objective
from dlux.tasks.classification import EXCLUDED, LabelMapping

STATS_SUFFIX = "_contract_stats.json"


def load_target_stats(stats_path: Path, field_name: str) -> Optional[dict]:
    """The build-db-resolved statistic for one endpoint, or None if it needs none.

    Written by ``build_db`` next to the study's DB. Absent file and absent entry are the same answer:
    this endpoint carries no fold-invariant statistic. A transform that needs one says so itself when
    it is handed None."""
    path = Path(stats_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text()).get(field_name)
    except (OSError, json.JSONDecodeError):
        return None


def _to_float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")  # missing / unparseable -> NaN sentinel, masked in the loss + metrics


def _binarizer(threshold: float) -> Callable[[Any], int]:
    """Continuous raw value -> {0, 1} at a cohort-wide cut; unparseable/NaN -> the categorical sentinel.

    ``>=`` puts the cut point itself in the positive class. Arbitrary but fixed: the alternative
    silently moves one patient per tied value between arms."""

    def apply(value: Any) -> int:
        parsed = _to_float_or_nan(value)
        if math.isnan(parsed):
            return EXCLUDED
        return int(parsed >= threshold)

    return apply


def _discretizer(edges: list[float]) -> Callable[[Any], int]:
    """Continuous raw value -> class idx {0..k-1} at cohort-wide interior ``edges``; unparseable/NaN -> the
    categorical sentinel. A value at a cut goes to the higher class, the same tie rule as ``_binarizer``
    (>=) and ``np.digitize`` (which the fold stratification uses), so labels and strata never disagree."""
    cuts = [float(e) for e in edges]

    def apply(value: Any) -> int:
        parsed = _to_float_or_nan(value)
        if math.isnan(parsed):
            return EXCLUDED
        return bisect.bisect_right(cuts, parsed)

    return apply


def build_field_transform(field: ContractField, resolved_stats: Optional[dict] = None) -> Callable[[Any], Any]:
    """Return the raw -> modeled-target callable for ``field``.

    ``resolved_stats`` is this endpoint's entry from the study's ``*_contract_stats.json``, resolved
    once at build-db over the cohort's valid-value patients. Only ``binarize`` needs it: the cut must
    be identical in every fold, or the ground truth itself would move between folds.
    """
    transform = field.source.transform
    if transform is not None and transform.categorical is not None:
        return LabelMapping(transform.categorical)
    if transform is not None and transform.binarize is not None:
        if not resolved_stats or "threshold" not in resolved_stats:
            raise ValueError(
                f"binarize target needs its build-db-resolved threshold, and none was recorded. "
                f"Rebuild the study's DB so {STATS_SUFFIX.lstrip('_')} carries it."
            )
        return _binarizer(float(resolved_stats["threshold"]))
    if transform is not None and transform.discretize is not None:
        if not resolved_stats or "edges" not in resolved_stats:
            raise ValueError(
                "discretize target needs its build-db-resolved edges, and none was recorded. "
                f"Rebuild the study's DB so {STATS_SUFFIX.lstrip('_')} carries it."
            )
        return _discretizer(resolved_stats["edges"])
    if transform is not None and transform.numeric == "log1p":
        return lambda value: math.log1p(_to_float_or_nan(value))  # elementwise: no statistic to resolve
    if field.objective == Objective.regression and transform is None:
        return _to_float_or_nan
    raise NotImplementedError(f"build_field_transform: no transform wired for {field.objective.value} + {transform!r}")
