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
"""Tests for dlux.data.sheet_builder: the metadata a cohort builder accepts."""

import pandas as pd
import pytest

from dlux.data.sheet_builder import one_row_per_patient


def test_repeated_rows_count_once_and_differing_rows_are_refused() -> None:
    meta = pd.DataFrame({"pid": ["a", "a", "b"], "grade": ["2", "2", "3"]})
    assert one_row_per_patient(meta, "pid", "c")["pid"].tolist() == ["a", "b"]

    conflicting = pd.DataFrame({"pid": ["a", "a", "b"], "grade": ["2", "3", "3"]})
    with pytest.raises(ValueError, match=r"\[c\] 1 patient\(s\) have metadata rows that differ, e.g. \['a'\]"):
        one_row_per_patient(conflicting, "pid", "c")
