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
"""Modalities: the kinds of data stream a task can read.

A **modality** is a kind (tile features, bulk RNA, a patient-sheet column). An **instance** of one is a
named stream in a particular task, so two tile-feature instances at different grids are normal. Each
modality owns how its adapter is built, how its samples collate, and any preprocessing the stream needs,
but not whether it is an input or a target. That is the task's decision, which lets one
definition of bulk RNA serve both a fusion input and the expression endpoint's target.

The registry is written out by hand: a new modality is one module plus one entry here.
"""

from __future__ import annotations

from dlux.modalities.bulk_rna import BulkRNA
from dlux.modalities.patient_field import PatientField
from dlux.modalities.tile_features import TileFeatures

MODALITIES = {modality.name: modality for modality in (TileFeatures, BulkRNA, PatientField)}

__all__ = ["MODALITIES", "BulkRNA", "PatientField", "TileFeatures"]
