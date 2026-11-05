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
"""The patient label on every tile: the target stream of ``TileClassificationTask``.

A task builds it directly, as it builds ``PatientField``, so it is not in the modality registry.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

from ahcore.data.adapters.patient_label import PatientLabelAdapter, PatientLabelTileAdapter
from ahcore.data.interfaces import ShapePolicy


class PatientFieldTile:
    """The tile counterpart of ``PatientField``: the same patient label, on every tile of a slide. Its
    adapter is a ``TileAdapter``, so it belongs to a tile datamodule, where each item is a tile."""

    name = "patient_field_tile"
    collate: ShapePolicy = "fixed"

    def __init__(
        self,
        *,
        key: str,
        fields: Sequence[str],
        transforms: Optional[Dict[str, "PatientLabelAdapter.Transform"]] = None,
    ) -> None:
        self.key = key
        self.fields = list(fields)
        self.transforms: Dict[str, "PatientLabelAdapter.Transform"] = dict(transforms or {})

    @property
    def width(self) -> int:
        return len(self.fields)

    def adapter(self) -> PatientLabelTileAdapter:
        return PatientLabelTileAdapter(
            patient_fields=self.fields,
            slide_fields=[],
            patient_field_transforms=self.transforms,
        )
