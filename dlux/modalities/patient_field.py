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
"""Patient fields: named columns from the cohort's patient sheet, one value per patient.

Today this is how every scalar endpoint's label is read (the contract field's column, mapped through
the contract's transform). It is the same stream a clinical covariate would come from, which is why it
is a modality rather than a hardcoded "label adapter".
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

from ahcore.data.adapters.patient_label import PatientLabelAdapter
from ahcore.data.interfaces import ShapePolicy


class PatientField:
    """A named patient-sheet stream: which columns to read, and how to convert each raw value."""

    name = "patient_field"
    collate: ShapePolicy = "fixed"

    def __init__(
        self,
        *,
        key: str,
        fields: Sequence[str],
        transforms: Optional[Dict[str, Callable]] = None,
    ) -> None:
        self.key = key
        self.fields = list(fields)
        self.transforms = dict(transforms or {})

    @property
    def width(self) -> int:
        """The width this stream contributes to the model: one value per named column."""
        return len(self.fields)

    def adapter(self) -> PatientLabelAdapter:
        return PatientLabelAdapter(
            patient_fields=self.fields,
            slide_fields=[],
            patient_field_transforms=self.transforms,
        )
