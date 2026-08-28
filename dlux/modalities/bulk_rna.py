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
"""Bulk RNA-seq: one gene-expression vector per patient, read from the cohort's RSEM matrix.

The same stream serves two roles, an input to a fusion model, or the target of the expression
endpoint, so it lives here and the role is chosen by the caller (see ``standardize``). The
active gene panel sizes the vector. The per-gene statistics come from the fold's fit split only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from ahcore.data.adapters.rna_expression import RNAExpressionAdapter
from ahcore.data.interfaces import ShapePolicy
from ahcore.utils.io import get_logger

from dlux.data import bulk_rna_matrix
from dlux.data.errors import BuildDbError
from dlux.data.gene_panel import resolve_panel
from dlux.modalities.context import ModalityContext

logger = get_logger(__name__)

# What to do about a patient with no RNA row. Refusing is the default because imputing changes what an
# arm is measuring while leaving every report looking complete; imputing stays available for the runs
# that want it.
MISSING_REFUSE = "refuse"
MISSING_IMPUTE_MEAN = "impute_mean"
MISSING_POLICIES = (MISSING_REFUSE, MISSING_IMPUTE_MEAN)


class BulkRNA:
    """A named bulk-RNA stream: the matrix to read, the active panel, and the fit-split per-gene
    statistics used to standardise it."""

    name = "bulk_rna"
    collate: ShapePolicy = "fixed"

    def __init__(
        self,
        *,
        key: str,
        matrix_path: Path,
        gene_means: np.ndarray,
        gene_stds: np.ndarray,
        gene_ids: Optional[Sequence] = None,
        panel_name: Optional[str] = None,
        missing: str = MISSING_REFUSE,
    ) -> None:
        if missing not in MISSING_POLICIES:
            raise ValueError(f"bulk_rna `missing` must be one of {sorted(MISSING_POLICIES)}, got '{missing}'")
        self.key = key
        self.missing = missing
        self._warned_imputed = False
        self.matrix_path = Path(matrix_path)
        self.gene_ids = gene_ids
        self.panel_name = panel_name
        self._set_stats(gene_means, gene_stds)

    def _set_stats(self, gene_means, gene_stds) -> None:
        """The one place the per-gene statistics are installed, so construction and per-fold replay
        cannot drift apart."""
        means = torch.as_tensor(np.asarray(gene_means), dtype=torch.float32)
        stds = torch.as_tensor(np.asarray(gene_stds), dtype=torch.float32)
        self.gene_means = means
        self.gene_stds = torch.where(stds > 0, stds, torch.ones_like(stds))  # a constant gene must not divide by 0

    def load_fit_state(self, state: dict) -> None:
        """Swap in one fold's recorded statistics, for scoring a cohort with several folds' models.

        Only the statistics may change. The gene ids may not: the adapter (and so which matrix columns
        the dataset reads, in which order) was fixed when this stream was constructed, so a
        different id list here would standardise the right numbers in the wrong places. That mismatch is
        the one thing making this swap unsafe, so it is checked rather than assumed."""
        ids = [str(g) for g in np.asarray(state["gene_ids"]).tolist()]
        current = [str(g) for g in (self.gene_ids or [])]
        if ids != current:
            raise ValueError(
                f"input '{self.key}' (bulk_rna): recorded state lists {len(ids)} gene(s) that differ from "
                f"the {len(current)} this stream was built for. Statistics can be swapped per fold; the "
                f"gene set cannot, because the adapter already reads a fixed set of columns."
            )
        self._set_stats(state["gene_means"], state["gene_stds"])

    @property
    def n_genes(self) -> int:
        """Active gene-panel size, the width this stream contributes to the model."""
        return int(self.gene_means.numel())

    @property
    def width(self) -> int:
        """The width this stream contributes to the model: the active gene panel."""
        return self.n_genes

    def adapter(self) -> RNAExpressionAdapter:
        return RNAExpressionAdapter(self.matrix_path, gene_ids=self.gene_ids, panel_name=self.panel_name)

    def standardize(self, raw: torch.Tensor) -> torch.Tensor:
        """Raw RSEM ``(B, G)`` -> log1p -> per-gene z-score, using the fit-split statistics.

        Missing values stay NaN. As a target that is what you want: the loss masks those genes rather
        than learning from a fabricated value. As an input, ``select`` applies the ``missing`` policy."""
        means = self.gene_means.to(raw.device)
        stds = self.gene_stds.to(raw.device)
        return (torch.log1p(raw) - means) / stds

    @classmethod
    def from_spec(cls, *, key: str, spec: dict, ctx: ModalityContext) -> "BulkRNA":
        """Construct from an ``inputs:`` entry, resolving this fold's per-gene statistics for the
        requested panel. Reads the fit split (via ctx) and nothing else."""
        if "impute_missing" in spec:
            raise ValueError(f"input '{key}' (bulk_rna): 'impute_missing' was renamed. Use missing: impute_mean.")
        unknown = set(spec) - {"modality", "gene_panel", "missing"}
        if unknown:
            raise ValueError(f"input '{key}' (bulk_rna): unknown option(s) {sorted(unknown)}")
        if ctx.cohorts_dir is None:
            raise ValueError(f"input '{key}' (bulk_rna) needs cohorts_dir to locate the RNA matrix")
        panel = str(spec.get("gene_panel", "full"))
        matrix_path = bulk_rna_matrix.matrix_path(ctx.cohorts_dir, ctx.cohort_name)
        recorded = ctx.recorded_state(key)
        if recorded is None:
            gene_ids, means, stds = cls.fit_stats(
                matrix_path=matrix_path,
                fit_patient_codes=ctx.fit_patient_codes(),
                gene_panel=panel,
                panels_dir=bulk_rna_matrix.panels_dir(ctx.cohorts_dir, ctx.cohort_name),
            )
        else:
            gene_ids, means, stds = cls.replay_stats(recorded, key=key, matrix_path=matrix_path, gene_panel=panel)
        return cls(
            key=key,
            matrix_path=matrix_path,
            gene_means=means,
            gene_stds=stds,
            gene_ids=gene_ids,
            panel_name=panel,
            missing=str(spec.get("missing", MISSING_REFUSE)),
        )

    def select(self, entry: dict) -> torch.Tensor:
        """The model-facing tensor for this stream, applying this input's ``missing`` policy.

        A patient with no matrix row arrives as an all-NaN vector. Imputing it to the per-gene mean makes
        that patient indistinguishable from an average one, so the model emits the same
        prediction for every such patient. That is sometimes a defensible modelling choice, but it must
        be chosen, and it must be visible, which is why ``refuse`` is the default and ``impute_mean``
        announces itself in the log."""
        raw = entry["expression"].float()
        missing = torch.isnan(raw).any(dim=-1)
        if not bool(missing.any()):
            return self.standardize(raw)

        n, total = int(missing.sum()), int(missing.numel())
        if self.missing == MISSING_REFUSE:
            raise ValueError(
                f"input '{self.key}' (bulk_rna): {n} of {total} samples in this batch have no RNA data. "
                f"Gate them out of the splits with `require_modalities: [bulk_rna]` on the study (then "
                f"rebuild the DB), or opt in explicitly with `missing: {MISSING_IMPUTE_MEAN}`."
            )
        if not self._warned_imputed:  # once per run, enough to make the choice visible in the log
            self._warned_imputed = True
            logger.warning(
                "input '%s' (bulk_rna): missing=%s — %d of %d samples in this batch have no RNA and are "
                "imputed to the per-gene mean. They share one constant vector, so this arm's predictions "
                "for them carry no RNA information; it is not comparable with an arm gated on bulk_rna.",
                self.key,
                MISSING_IMPUTE_MEAN,
                n,
                total,
            )
        return torch.nan_to_num(self.standardize(raw), nan=0.0)

    # -- build-time capabilities ---------------------------------------------------------------
    # These answer questions about the cohort's data rather than about one configured stream, so they
    # need no instance: which patients have this modality at all, and what its statistics are on a
    # given set of them.

    @staticmethod
    def coverage(*, cohorts_dir: Optional[Path], cohort_name: str, manifest_codes: set) -> set:
        """Manifest patients that have an RNA-matrix row.

        Both the expression endpoint's own coverage and the ``Study.require_modalities`` gate read this,
        so there is one definition of "this patient has bulk RNA".
        """
        if cohorts_dir is None:
            raise BuildDbError("the 'bulk_rna' modality needs cohorts_dir to locate the RNA matrix")
        matrix = bulk_rna_matrix.matrix_path(cohorts_dir, cohort_name)
        if not matrix.exists():
            raise BuildDbError(
                f"the 'bulk_rna' modality has no matrix at {matrix} (build the cohort's rnaseq/matrix.parquet)."
            )
        import pandas as pd

        rna_codes = set(pd.read_parquet(matrix, columns=[]).index.astype(str))
        return rna_codes & manifest_codes

    def fit_state(self) -> dict:
        """This stream's fit-derived state, for replay when scoring a cohort that has no fit split.

        The gene identities travel with the statistics because a panel name does not pin them down: a
        data-derived panel like ``hvg2000`` is computed per cohort, so two cohorts' ``hvg2000`` are
        different gene sets. Replaying by name would silently score a model on the wrong genes."""
        return {
            "gene_ids": np.asarray(self.gene_ids),
            "gene_means": self.gene_means.numpy(),
            "gene_stds": self.gene_stds.numpy(),
            "panel_name": self.panel_name,
        }

    @staticmethod
    def replay_stats(state: dict, *, key: str, matrix_path: Path, gene_panel: str) -> tuple:
        """Restore recorded per-gene statistics, checking this cohort's matrix can supply those genes.

        The recorded gene ids win: the run's panel is deliberately not re-resolved against this cohort,
        because a data-derived panel would resolve to a different gene set here. Genes are matched by
        identity, never by position, and a missing one is an error rather than a silent drop, the model
        expects a fixed-width vector in a fixed order."""
        import pandas as pd

        recorded_ids = [str(g) for g in np.asarray(state["gene_ids"]).tolist()]
        recorded_panel = state.get("panel_name")
        if recorded_panel is not None and str(recorded_panel) != gene_panel:
            raise ValueError(
                f"input '{key}' (bulk_rna): the run recorded gene_panel '{recorded_panel}' but this config "
                f"asks for '{gene_panel}'. Score the model with the panel it was trained on."
            )
        available = {str(c) for c in pd.read_parquet(matrix_path, columns=None).columns}
        absent = [g for g in recorded_ids if g not in available]
        if absent:
            raise ValueError(
                f"input '{key}' (bulk_rna): {len(absent)} of {len(recorded_ids)} recorded genes are absent "
                f"from {matrix_path} (e.g. {absent[:8]}). The recorded panel cannot be reconstructed here, "
                f"so this cohort cannot score that model."
            )
        return recorded_ids, np.asarray(state["gene_means"]), np.asarray(state["gene_stds"])

    @staticmethod
    def fit_stats(
        *,
        matrix_path: Path,
        fit_patient_codes: Sequence[str],
        gene_panel: str,
        panels_dir: Path,
    ) -> tuple:
        """Resolve the active panel and the per-gene μ/σ of log1p expression over ``fit_patient_codes``.

        The caller supplies the patient codes rather than a split to read, which keeps the leakage
        boundary here (``build.py`` reads the ``fit`` split and nothing else) and leaves this a
        pure function of the matrix. Returns ``(gene_ids, means, stds)`` in panel order.
        """
        import pandas as pd

        matrix = pd.read_parquet(matrix_path)
        matrix.index = matrix.index.astype(str)
        gene_ids = resolve_panel(gene_panel, panels_dir, list(matrix.columns))
        matrix = matrix[gene_ids]  # active panel, panel order
        fit = matrix.loc[matrix.index.intersection(list(fit_patient_codes))]
        if len(fit) < 2:
            raise ValueError(f"expression z-score needs >= 2 fit-split patients in the RNA matrix, got {len(fit)}.")
        log1p = np.log1p(fit.to_numpy(dtype=float))  # (n_fit, G)
        means = np.nanmean(log1p, axis=0).astype(np.float32)
        stds = np.nanstd(log1p, axis=0).astype(np.float32)
        logger.info("bulk_rna: per-gene fit-split stats over %d patients x %d genes", len(fit), matrix.shape[1])
        return gene_ids, means, stds
