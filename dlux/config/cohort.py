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
"""Typed schema for a dlux ``Cohort`` (a data source) and a ``Study`` (an experiment).

A **cohort** is one data source: where its data lives (``storage``) and how its raw
labels map to prediction targets (``contract``). It is reusable and decides nothing
about how it is used, no split strategy, no DB location.

A **study** composes cohorts by assigning each a **role** (``development`` /
``validation``) and holds the CV split parameters. The role determines the split
strategy (``ROLE_STRATEGY``); the DB for a cohort in a study is written to
``studies/<study>/db/<cohort>.db``.

The **contract** decouples a prediction target into three independent facets (see
``docs/specs/CONTRACT_SPEC.md``): ``source`` (modality + column + optional transform), ``objective``
(head/loss/metrics/arity), ``stratify`` (fold balancing). The ``{type: ..., map: ...}``
shorthand is kept as input sugar (``_expand_sugar``). Class weighting is not a facet: it is a
training-time decision rather than a statement about the endpoint, so it lives on the task config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# Image readers we accept for the WSI path (dlup ``ImageBackend`` member names).
_VALID_IMAGE_READERS = {"FASTSLIDE", "FIMAGE", "OPENSLIDE", "TIFFFILE"}


# -- objective + its resolution table ----------------------------------------
class Objective(str, Enum):
    """What head/loss/metrics/arity a target uses, the single dispatch point."""

    binary = "binary"  # 1 logit, BCE-with-logits, float{0,1}
    multiclass = "multiclass"  # K logits, softmax CE, long index
    regression = "regression"  # 1 output, MSE, float
    regression_vector = "regression_vector"  # N outputs (e.g. gene expression), MSE over the vector, float
    survival = "survival"  # (event, time) pair; n_bins hazard logits, discrete-time NLL, Harrell C-index
    # segmentation = deferred (see CONTRACT_SPEC.md)


@dataclass(frozen=True)
class _ObjectiveSpec:
    """Field-independent facts for an objective (the declaration locus).

    ``num_outputs`` is not here. It needs the field's map for multiclass, so it is a
    ``ContractField`` property. Metric names are tagged by consumer: ``stream`` = logged per epoch
    (torchmetrics); ``report`` = full pooled/OOF set (sklearn, aggregate).
    A metric may be in both (e.g. auroc)."""

    target_dtype: str  # "float" | "long"
    missing_sentinel: float  # -1 (categorical) | NaN (regression)
    stream_metrics: tuple[str, ...]
    report_metrics: tuple[str, ...]


_BINARY_REPORT = (
    "auroc",
    "ap",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "npv",
    "f1",
    "mcc",
    "kappa",
)

OBJECTIVE_TABLE: dict[Objective, _ObjectiveSpec] = {
    Objective.binary: _ObjectiveSpec("float", -1, ("auroc", "accuracy"), _BINARY_REPORT),
    Objective.multiclass: _ObjectiveSpec("long", -1, ("accuracy", "auroc"), ("accuracy", "auroc")),
    Objective.regression: _ObjectiveSpec("float", math.nan, ("mse", "mae"), ("mse", "mae", "rmse", "pearson", "r2")),
    Objective.regression_vector: _ObjectiveSpec(
        "float",
        math.nan,
        ("gene_pearson_mean",),  # per-gene Pearson, averaged across genes (logged per epoch)
        ("gene_pearson_mean", "gene_pearson_topk", "gene_spearman_median"),  # aggregate report set
    ),
    Objective.survival: _ObjectiveSpec("float", math.nan, ("c_index",), ("c_index",)),
}


# -- transform + its binning -------------------------------------------------
class StratifyMethod(str, Enum):
    """Binning of a continuous variable, shared by target ``binarize`` and fold ``stratify``."""

    median = "median"  # split at the median (== quantile k=2)
    quantile = "quantile"  # k equal-frequency bins; requires k >= 2
    threshold = "threshold"  # explicit edge(s)


class BinarizeSpec(BaseModel):
    """Continuous -> {0,1} target. One cut (2 bins). The threshold is resolved at build-db."""

    model_config = ConfigDict(extra="forbid")

    method: StratifyMethod  # median | quantile(==k=2) | threshold
    value: Optional[float] = None  # threshold only (the single cut point)

    @model_validator(mode="after")
    def _check(self) -> "BinarizeSpec":
        if self.method == StratifyMethod.threshold and self.value is None:
            raise ValueError("binarize 'threshold' requires 'value'")
        if self.method != StratifyMethod.threshold and self.value is not None:
            raise ValueError(f"binarize '{self.method.value}' takes no 'value'")
        return self


class DiscretizeSpec(BaseModel):
    """Continuous -> k class indices {0..k-1}. ``quantile`` = k equal-frequency bins; ``threshold`` =
    explicit interior edges (k-1 of them). The cut points are fold-invariant, resolved once at build-db
    exactly like ``binarize``, which is the k=2 case, so use that for a two-class cut."""

    model_config = ConfigDict(extra="forbid")

    method: StratifyMethod  # quantile | threshold (median is binarize's k=2 job)
    k: Optional[int] = None  # quantile: number of classes
    edges: Optional[list[float]] = None  # threshold: interior cut points, k-1 of them

    @model_validator(mode="after")
    def _check(self) -> "DiscretizeSpec":
        if self.method == StratifyMethod.quantile:
            if self.k is None or self.k < 3:
                raise ValueError("discretize 'quantile' requires k >= 3 (use binarize for a 2-class cut)")
            if self.edges is not None:
                raise ValueError("discretize 'quantile' takes no 'edges'")
        elif self.method == StratifyMethod.threshold:
            if not self.edges or len(self.edges) < 2:
                raise ValueError("discretize 'threshold' requires >= 2 'edges' (>= 3 classes); binarize takes one")
            if self.k is not None:
                raise ValueError("discretize 'threshold' takes no 'k'")
        else:
            raise ValueError("discretize 'method' must be 'quantile' or 'threshold' (median is binarize, k=2)")
        return self

    @property
    def num_classes(self) -> int:
        return int(self.k) if self.method == StratifyMethod.quantile else len(self.edges) + 1


class Transform(BaseModel):
    """Raw value -> modeled label. Exactly one kind set when present (or the whole Transform absent)."""

    model_config = ConfigDict(extra="forbid")

    categorical: Optional[dict[str, int]] = None  # raw str -> class idx; keys = inclusion whitelist
    binarize: Optional[BinarizeSpec] = None  # continuous -> {0,1}
    discretize: Optional[DiscretizeSpec] = None  # continuous -> {0..k-1}
    numeric: Optional[Literal["log1p"]] = None  # elementwise; z-scoring a target is task.target_normalize

    @model_validator(mode="after")
    def _exactly_one(self) -> "Transform":
        present = [k for k in ("categorical", "binarize", "discretize", "numeric") if getattr(self, k) is not None]
        if len(present) != 1:
            raise ValueError(
                f"Transform must set exactly one of categorical/binarize/discretize/numeric, got {present or 'none'}"
            )
        return self


class Source(BaseModel):
    """Where the raw label lives. ``modality``-discriminated so new storage kinds slot in additively."""

    model_config = ConfigDict(extra="forbid")

    # Where the values live, in the shared modality vocabulary: `patient_field` = a column of the patient
    # sheet, `bulk_rna` = a row of the cohort's RNA matrix. This is the one place a target's storage is
    # named; a run's task config references the field and never restates its modality, because storage is
    # a property of the cohort, not a per-run choice.
    modality: Literal["patient_field", "bulk_rna"] = "patient_field"
    column: Optional[str] = None  # defaults to the contract key (injected at Cohort level); None for bulk_rna
    # survival only: the follow-up-time column (source.column names the event column). None otherwise.
    time_column: Optional[str] = None
    # Granularity of a column label = which manifest table it lives in. Reserved: `image` (a distinct
    # value per slide, e.g. a per-section measurement) needs image_labels ingestion + a per-image reader,
    # which are not built, only `patient` is wired today. Drives storage by data granularity, not concept.
    level: Literal["patient", "image"] = "patient"
    transform: Optional[Transform] = None

    @model_validator(mode="after")
    def _check_level(self) -> "Source":
        if self.level == "image":
            raise ValueError("source.level='image' is reserved but not implemented (no image_labels ingestion yet)")
        return self


class Stratify(BaseModel):
    """How CV folds are balanced. Default (all None) -> derive from the modeled target."""

    model_config = ConfigDict(extra="forbid")

    column: Optional[str] = None  # None -> derive from the modeled target
    method: Optional[StratifyMethod] = None  # binning a continuous stratify variable
    k: Optional[int] = None  # quantile only; k >= 2
    edges: Optional[list[float]] = None  # threshold only

    @model_validator(mode="after")
    def _check(self) -> "Stratify":
        if self.method == StratifyMethod.quantile and (self.k is None or self.k < 2):
            raise ValueError("stratify 'quantile' requires k >= 2")
        if self.method == StratifyMethod.threshold and not self.edges:
            raise ValueError("stratify 'threshold' requires non-empty 'edges'")
        if self.method != StratifyMethod.quantile and self.k is not None:
            raise ValueError("stratify 'k' is for 'quantile' only")
        if self.method != StratifyMethod.threshold and self.edges is not None:
            raise ValueError("stratify 'edges' is for 'threshold' only")
        return self


# -- the contract field ------------------------------------------------------
class ContractField(BaseModel):
    """One predictable endpoint, declared as three independent facets.

    Authored either as facet form (``source``/``objective``/...) or the ``{type, map}``
    shorthand, which ``_expand_sugar`` expands to facet form before validation.

    Every facet here is read at build-db time. Training-time choices about the endpoint (loss weighting,
    target standardisation) belong to the task config instead. See ``dlux.config.task``."""

    model_config = ConfigDict(extra="forbid")

    source: Source
    objective: Objective
    stratify: Stratify = Stratify()

    @model_validator(mode="before")
    @classmethod
    def _expand_sugar(cls, data):
        """Expand `{type: ..., map: ...}` shorthand to facet form. Leaves source.column None
        (injected at Cohort level). Any non-sugar keys (stratify) pass through."""
        if not isinstance(data, dict) or "type" not in data:
            return data  # already facet form (or not a dict), validate as-is
        if "objective" in data or "source" in data:
            raise ValueError("contract field: use either `type:` shorthand or facet form (objective/source), not both")

        data = dict(data)
        type_ = data.pop("type")
        if type_ == "survival":  # (event, time) endpoint: name both columns explicitly (no key-injection)
            event, time = data.pop("event", None), data.pop("time", None)
            if not event or not time:
                raise ValueError("shorthand 'survival' requires 'event' and 'time' column names")
            return {"objective": "survival", "source": {"column": event, "time_column": time}, **data}
        if type_ in ("binary", "multiclass"):
            mapping = data.pop("map", None)
            if not mapping:
                raise ValueError(f"shorthand '{type_}' requires a non-empty 'map'")
            return {"objective": type_, "source": {"transform": {"categorical": mapping}}, **data}
        if type_ == "continuous":
            target_transform = data.pop("target_transform", None)
            stratify = data.pop("stratify", None)
            source = {"transform": {"numeric": target_transform}} if target_transform else {}
            return {
                "objective": "regression",
                "source": source,
                "stratify": stratify if stratify is not None else {"method": "median"},
                **data,
            }
        if type_ == "expression":  # RNA-seq gene-expression vector, target is an external matrix, not a sheet column
            return {"objective": "regression_vector", "source": {"modality": "bulk_rna"}, **data}
        raise ValueError(f"unknown shorthand type '{type_}'")

    @model_validator(mode="after")
    def _check(self) -> "ContractField":
        t = self.source.transform
        obj = self.objective

        # time_column is survival-only (a second sheet column: the follow-up time)
        if obj != Objective.survival and self.source.time_column is not None:
            raise ValueError("source.time_column is only valid for the survival objective")

        # matrix source is exclusively the expression endpoint (external matrix, no sheet column)
        if self.source.modality == "bulk_rna":
            if obj != Objective.regression_vector:
                raise ValueError("source.modality='bulk_rna' is only valid for the regression_vector objective")
            if self.source.column is not None:
                raise ValueError(
                    "source.modality='bulk_rna' takes no column — the target is keyed by patient in the matrix"
                )

        # objective <-> transform compatibility
        if obj == Objective.binary:
            if t is None or (t.categorical is None and t.binarize is None):
                raise ValueError("binary objective requires a categorical map (2 classes) or a binarize transform")
            if t.numeric is not None:
                raise ValueError("binary objective cannot use a numeric transform")
            if t.categorical is not None and len(set(t.categorical.values())) != 2:
                raise ValueError(
                    f"binary categorical map must have exactly 2 classes, got {len(set(t.categorical.values()))}"
                )
        elif obj == Objective.multiclass:
            if t is None or (t.categorical is None and t.discretize is None):
                raise ValueError("multiclass objective requires a categorical map or a discretize transform")
            if t.binarize is not None or t.numeric is not None:
                raise ValueError("multiclass objective supports only a categorical or discretize transform")
            if t.categorical is not None and len(set(t.categorical.values())) < 3:
                raise ValueError(f"multiclass map must have >= 3 classes, got {len(set(t.categorical.values()))}")
        elif obj == Objective.regression:
            if t is not None and (t.categorical is not None or t.binarize is not None or t.discretize is not None):
                raise ValueError("regression objective supports only a numeric transform (or none)")
        elif obj == Objective.regression_vector:
            if self.source.modality != "bulk_rna":
                raise ValueError(
                    "regression_vector requires source.modality='bulk_rna' — an external matrix, not a sheet column"
                )
            if t is not None:
                raise ValueError(
                    "regression_vector transforms are applied at train time (task target_transform), not on the source"
                )
        elif obj == Objective.survival:
            if self.source.modality != "patient_field" or self.source.column is None or self.source.time_column is None:
                raise ValueError("survival requires source.column (event indicator) + source.time_column (time)")
            if t is not None:
                raise ValueError("survival takes no transform — event is read as 0/1 and time as a raw float")

        # stratify coherence (the resolution itself lives in splits.py; here we validate + default)
        s = self.stratify
        if obj == Objective.regression_vector:
            if s.column is not None or s.method is not None or s.k is not None or s.edges is not None:
                raise ValueError(
                    "regression_vector is unstratified — a high-dim vector target has no stratification axis"
                )
        elif s.column is None:
            if obj in (Objective.binary, Objective.multiclass) and s.method is not None:
                raise ValueError("stratify.method is not allowed when stratifying on a categorical/binary target")
            if obj == Objective.regression and s.method is None:
                s.method = StratifyMethod.median  # regression target needs a binning method; default median
        return self

    @property
    def num_outputs(self) -> int:
        """Model head output count: 1 (binary/regression) or K (multiclass)."""
        if self.objective == Objective.multiclass:
            t = self.source.transform
            assert t is not None and (t.categorical is not None or t.discretize is not None)
            return t.discretize.num_classes if t.discretize is not None else len(set(t.categorical.values()))
        if self.objective == Objective.regression_vector:
            raise ValueError(
                "regression_vector output count = the gene-panel size, resolved at train time (not the contract)"
            )
        if self.objective == Objective.survival:
            raise ValueError("survival head width = task.n_bins (discrete-time intervals), resolved at train time")
        return 1

    @property
    def family(self) -> str:
        """The target's data shape, which per-endpoint subsystems (analyze, build-db) dispatch their
        handling on: ``"bulk_rna"`` for the external-matrix vector target, ``"sheet"`` for a manifest
        column (binary/multiclass/regression). Derived from the source modality the contract already
        declares, so the grouping is defined once instead of per subsystem."""
        return "bulk_rna" if self.source.modality == "bulk_rna" else "sheet"

    def target_dtype(self) -> str:
        return OBJECTIVE_TABLE[self.objective].target_dtype

    def missing_sentinel(self) -> float:
        return OBJECTIVE_TABLE[self.objective].missing_sentinel

    def metric_names(self, kind: Literal["stream", "report"]) -> tuple[str, ...]:
        spec = OBJECTIVE_TABLE[self.objective]
        return spec.stream_metrics if kind == "stream" else spec.report_metrics


# -- cohort (a data source: storage + contract, no splits, no DB location) ---
class Storage(BaseModel):
    """Where this cohort's external data lives on disk. The DB location is not here. It is derived
    from the study (``studies/<study>/db/<cohort>.db``), because the same cohort can be
    materialised differently under different studies."""

    model_config = ConfigDict(extra="forbid")

    image_dir: str  # WSI root; slides.csv image_path is relative to this
    mask_dir: Optional[str] = None  # mask root; None -> cohort has no masks
    default_reader: str = "FASTSLIDE"  # per-slide `reader` column overrides this
    default_staining: Optional[str] = None  # e.g. "H&E"; per-slide `staining` column overrides this

    @field_validator("default_reader")
    @classmethod
    def _valid_reader(cls, value: str) -> str:
        upper = value.upper()
        if upper not in _VALID_IMAGE_READERS:
            raise ValueError(f"default_reader must be one of {sorted(_VALID_IMAGE_READERS)}, got {value!r}")
        return upper


class Cohort(BaseModel):
    """One data source: identity + where its data lives + its label contract.

    Reusable across studies. Decides nothing about splitting or DB location, which are
    study concerns (see ``Study``)."""

    model_config = ConfigDict(extra="forbid")

    name: str  # == manifest.name in the DB
    storage: Storage
    contract: dict[str, ContractField]

    @model_validator(mode="after")
    def _check(self) -> "Cohort":
        # Inject the contract key as source.column when not given explicitly (name == column, common case).
        for name, field in self.contract.items():
            if field.source.modality == "patient_field" and field.source.column is None:
                field.source.column = name

        reserved = {"patient_id", "image_path", "mask_path", "reader"}
        clash = reserved & set(self.contract)
        if clash:
            raise ValueError(f"contract fields collide with reserved sheet columns: {sorted(clash)}")
        return self


# -- splits (strategy + params), a study composes these from role + SplitParams ---
class SplitCategory:
    """The four manifest split categories (mirror ahcore's CategoryEnum). Plain strings, not a str-Enum,
    so they interpolate into metric names (``f"{split}/auroc"``) and DB queries as the bare literals. One
    vocabulary for split generation, build-db, task metrics, and extraction."""

    FIT = "fit"
    VALIDATE = "validate"
    TEST = "test"
    PREDICT = "predict"


class SplitStrategy(str, Enum):
    nested_cv = "nested_cv"  # development: n_outer x n_inner nested CV, per label
    simple = "simple"  # one fixed FIT/VALIDATE(/TEST) partition, per label
    all_test = "all_test"  # validation: single split, all patients TEST
    all_predict = "all_predict"  # unlabeled inference: single split, all patients PREDICT


class Splits(BaseModel):
    """A concrete split plan (strategy + params). Not authored directly under the study,
    ``Study.splits_for`` composes it from a cohort's role + the study's ``SplitParams``."""

    model_config = ConfigDict(extra="forbid")

    strategy: SplitStrategy
    n_outer: int = 5  # nested_cv
    n_inner: int = 5  # nested_cv
    ratios: Optional[list[float]] = None  # simple: len 2 (fit/val) or 3 (fit/val/test)
    per_label: bool = True  # generate splits per contract field over its valid subset
    random_state: int = 42
    # Study-required input modalities (e.g. ["rna"]): every field's splits are restricted to patients
    # who have all of them, so a fusion study's arms all train/score on one patient set (fair comparison).
    # [] = no gate. Resolved to patient-coverage sets by build_db's modality registry.
    require_modalities: list[str] = []

    @model_validator(mode="after")
    def _check(self) -> "Splits":
        if self.strategy == SplitStrategy.simple:
            if not self.ratios or len(self.ratios) not in (2, 3):
                raise ValueError("simple split requires 'ratios' of length 2 or 3")
            if abs(sum(self.ratios) - 1.0) > 1e-6:
                raise ValueError(f"simple split 'ratios' must sum to 1, got {sum(self.ratios)}")
        if self.strategy == SplitStrategy.nested_cv and (self.n_outer < 2 or self.n_inner < 1):
            raise ValueError("nested_cv requires n_outer >= 2 and n_inner >= 1")
        return self


# -- study (composes cohorts by role; owns split params + DB grouping) -------
class Role(str, Enum):
    """A cohort's function within a study, determines its split strategy (``ROLE_STRATEGY``)."""

    development = "development"  # trains the models (internal nested-CV)
    validation = "validation"  # held-out cohort, scored by evaluate_external
    predict = "predict"  # unlabeled inference target, run by bin/predict (all_predict, no labels)


ROLE_STRATEGY: dict[Role, SplitStrategy] = {
    Role.development: SplitStrategy.nested_cv,
    Role.validation: SplitStrategy.all_test,
    Role.predict: SplitStrategy.all_predict,
}


class SplitParams(BaseModel):
    """Study-level CV parameters. The *strategy* is not here. It comes from each cohort's role."""

    model_config = ConfigDict(extra="forbid")

    n_outer: int = 5
    n_inner: int = 5
    ratios: Optional[list[float]] = None
    per_label: bool = True
    random_state: int = 42


class Study(BaseModel):
    """A named experiment: composes cohorts by role, owns the CV split parameters.

    The DB for ``cohort`` in this study is written to ``studies/<name>/db/<cohort>.db``. Its split
    strategy is ``ROLE_STRATEGY[role]`` composed with the study's ``splits`` params."""

    model_config = ConfigDict(extra="forbid")

    name: str
    cohorts: dict[str, Role]  # cohort name -> role
    splits: SplitParams = SplitParams()
    # The contract fields this study models. Required: "every endpoint the cohort happens to offer" is
    # exactly the kind of implicit default that makes a sweep's expected shape unknowable downstream.
    targets: list[str]
    # Patients must have all of these modalities to enter the splits (e.g. ["bulk_rna"] for a fusion
    # study, so every arm compares on one covered set). Required: `[]` is a decision to gate on nothing,
    # omission was an oversight. Validated where it is resolved, against the modalities that actually
    # declare a coverage capability (build_db).
    require_modalities: list[str]
    # Per-survival-target censoring horizon (time-column units). A patient past the horizon is censored
    # at it (event -> 0, time -> horizon); omit a target for no cap. Applied in build_db, so both the
    # stored labels and the stratified splits carry the capped endpoint. Needed when pooling cohorts of
    # unequal follow-up. build_db checks the target is survival.
    admin_censor: dict[str, float] = {}

    @model_validator(mode="after")
    def _check(self) -> "Study":
        if not self.cohorts:
            raise ValueError("a study must reference at least one cohort")
        if not any(r == Role.development for r in self.cohorts.values()):
            raise ValueError("a study needs at least one 'development' cohort (the models come from there)")
        if not self.targets:
            raise ValueError("study 'targets' must be a non-empty list of contract fields this study models")
        unknown = [t for t in self.admin_censor if t not in self.targets]
        if unknown:
            raise ValueError(f"admin_censor names {unknown}, which are not study targets {self.targets}")
        nonpositive = {t: h for t, h in self.admin_censor.items() if not h > 0}
        if nonpositive:
            raise ValueError(f"admin_censor horizons must be positive, got {nonpositive}")
        return self

    def filter_contract(self, cohort: Cohort) -> Cohort:
        """Restrict ``cohort``'s contract to this study's ``targets``.

        A cohort declares every endpoint it offers. A study models only a subset. Applying this at the
        build-db / analyze boundary means the study's DB carries splits (and its analysis shows panels)
        for only the endpoints it targets, not every endpoint the cohort happens to provide."""
        unknown = [t for t in self.targets if t not in cohort.contract]
        if unknown:
            raise ValueError(
                f"study '{self.name}' targets {unknown} not in cohort '{cohort.name}' contract {sorted(cohort.contract)}"
            )
        return cohort.model_copy(update={"contract": {t: cohort.contract[t] for t in self.targets}})

    def strategy_for(self, cohort: str) -> SplitStrategy:
        if cohort not in self.cohorts:
            raise KeyError(f"cohort '{cohort}' is not part of study '{self.name}' (has: {sorted(self.cohorts)})")
        return ROLE_STRATEGY[self.cohorts[cohort]]

    def splits_for(self, cohort: str) -> Splits:
        """The concrete split plan for ``cohort`` in this study: role -> strategy + params."""
        return Splits(
            strategy=self.strategy_for(cohort),
            require_modalities=self.require_modalities,
            **self.splits.model_dump(),
        )
