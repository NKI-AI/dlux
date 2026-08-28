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
"""Tests for the modality registry and the bulk_rna missing-data policy.

The policy matters more than it looks: imputing a patient with no RNA row gives every such patient one
identical constant vector, which changes what an arm measures while leaving its reports looking
complete. So the default must refuse, and opting in must be audible."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from dlux.modalities import MODALITIES, BulkRNA

_MEANS = np.zeros(3, dtype=np.float32)
_STDS = np.ones(3, dtype=np.float32)


def _rna(**kwargs) -> BulkRNA:
    return BulkRNA(key="bulk_rna", matrix_path=Path("/none.parquet"), gene_means=_MEANS, gene_stds=_STDS, **kwargs)


def _batch(n_missing: int, n: int = 4) -> dict:
    """A batch where the first ``n_missing`` patients have no RNA row (an all-NaN vector)."""
    raw = torch.zeros(n, 3)
    raw[:n_missing] = float("nan")
    return {"expression": raw}


def test_registry_is_keyed_by_modality_name():
    assert MODALITIES["bulk_rna"] is BulkRNA
    assert all(name == cls.name for name, cls in MODALITIES.items())


def test_missing_defaults_to_refuse():
    """Not setting a policy must not silently impute — the default is the safe one."""
    with pytest.raises(ValueError, match="require_modalities"):
        _rna().select(_batch(1))


def test_refuse_reports_how_many_and_names_both_remedies():
    with pytest.raises(ValueError, match=r"2 of 4 samples.*require_modalities.*impute_mean"):
        _rna(missing="refuse").select(_batch(2))


def test_impute_mean_is_opt_in_and_produces_finite_values():
    out = _rna(missing="impute_mean").select(_batch(2))
    assert torch.isfinite(out).all()
    assert torch.equal(out[0], out[1])  # every missing patient gets the SAME constant vector


def test_impute_mean_warns_once(caplog):
    rna = _rna(missing="impute_mean")
    with caplog.at_level("WARNING"):
        rna.select(_batch(2))
        rna.select(_batch(2))
    assert sum("imputed to the per-gene mean" in r.getMessage() for r in caplog.records) == 1


def test_a_complete_batch_is_untouched_under_either_policy():
    for policy in ("refuse", "impute_mean"):
        out = _rna(missing=policy).select(_batch(0))
        assert torch.isfinite(out).all()


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="must be one of"):
        _rna(missing="drop")


def test_old_impute_missing_key_points_at_the_new_spelling():
    with pytest.raises(ValueError, match="missing: impute_mean"):
        BulkRNA.from_spec(key="bulk_rna", spec={"modality": "bulk_rna", "impute_missing": True}, ctx=None)
