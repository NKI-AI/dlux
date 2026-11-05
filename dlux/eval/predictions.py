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
"""Single-pass prediction collection plus dual-format writers (CSV + NPZ).

One forward pass over *any* dataloader (internal test or external cohort) yields a
single ``predictions`` dict that both writers consume: the CSV is for human
inspection (rounded). The NPZ is the lossless artifact downstream cross-fold
aggregation reads.

**Endpoint-type aware.** ``PREDICTION_SPECS`` holds, per objective, how the head output becomes the
prediction, the label's shape and dtype, and the CSV a human reads. Binary -> ``sigmoid`` (``probs``
``(N,)``); multiclass -> ``softmax`` (``(N, K)``); regression -> the raw value (``(N,)``, float labels);
expression -> the gene vector (``(N, G)``, log1p labels); survival -> the scalar risk ``-sum_j S_j``
(``(N,)``) with ``labels`` the ``(N, 2)`` ``[time, event]`` pair, since concordance couples the two.

Survival additionally carries ``time_edges``: ``logits`` is the ``(N, n_bins)`` block of per-bin hazard
logits, and a hazard vector says nothing without the intervals its bins span. The edges are quantiles
of the fold's own fit split, so they differ from fold to fold. Anything comparing or ensembling curves
across folds must read them rather than assume a shared grid.

``patient_code`` is captured at prediction time (off each batch's ``slide_ref``) so downstream
aggregation does patient-level rollups without ever re-opening the manifest DB.
"""

from __future__ import annotations

import csv
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from dlux.config.cohort import Objective
from dlux.eval.attention import (
    capture_input,
    extract_records,
    find_attention_module,
    find_tile_modality,
    raw_attention_logits,
)
from dlux.eval.survival_metrics import bin_hazards, bin_ranges, bin_survival
from dlux.tasks.survival import survival_risk

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from ahcore.mm_lit_module import MultiModalLightningModule


def _to_device(value: Any, device: torch.device) -> Any:
    """Move tensors to ``device``, recursing into dicts, lists and tuples. Other values are returned unchanged."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(*(_to_device(v, device) for v in value))
    if isinstance(value, (list, tuple)):
        moved = [_to_device(v, device) for v in value]
        return type(value)(moved) if isinstance(value, tuple) else moved
    return value


CsvRows = Tuple[List[str], List[Dict[str, Any]]]


def _binary_csv(predictions: Dict[str, Any]) -> CsvRows:
    """``slide_id,patient_code,logit,prob,label`` (integer label)."""
    rows = [
        {
            "slide_id": sid,
            "patient_code": pcode,
            "logit": f"{float(lgt):.4f}",
            "prob": f"{float(prb):.4f}",
            "label": int(lbl),
        }
        for sid, pcode, lgt, prb, lbl in zip(
            predictions["slide_ids"],
            predictions["patient_codes"],
            predictions["logits"],
            predictions["probs"],
            predictions["labels"],
        )
    ]
    return ["slide_id", "patient_code", "logit", "prob", "label"], rows


def _multiclass_csv(predictions: Dict[str, Any]) -> CsvRows:
    """``slide_id,patient_code,prob_0..prob_{K-1},pred,label``: the per-class softmax probs, the argmax
    predicted class, and the true class index."""
    probs = np.asarray(predictions["probs"], dtype=np.float64)  # (N, K)
    prob_cols = [f"prob_{c}" for c in range(probs.shape[1])]
    rows = [
        {
            "slide_id": sid,
            "patient_code": pcode,
            **{col: f"{float(v):.4f}" for col, v in zip(prob_cols, row)},
            "pred": int(np.argmax(row)),
            "label": int(lbl),
        }
        for sid, pcode, row, lbl in zip(
            predictions["slide_ids"], predictions["patient_codes"], probs, predictions["labels"]
        )
    ]
    return ["slide_id", "patient_code", *prob_cols, "pred", "label"], rows


def _regression_csv(predictions: Dict[str, Any]) -> CsvRows:
    """``slide_id,patient_code,pred,label`` (float label)."""
    rows = [
        {"slide_id": sid, "patient_code": pcode, "pred": f"{float(pred):.4f}", "label": f"{float(lbl):.4f}"}
        for sid, pcode, pred, lbl in zip(
            predictions["slide_ids"], predictions["patient_codes"], predictions["logits"], predictions["labels"]
        )
    ]
    return ["slide_id", "patient_code", "pred", "label"], rows


def _survival_csv(predictions: Dict[str, Any]) -> CsvRows:
    """``slide_id,patient_code,risk,time,event,h_<range>...,S_<range>...``: the ranking score, the observed
    follow-up, then the per-bin hazard and survival probability with each column named by the interval it
    covers (``h_183-402``), so a row is readable without the metadata."""
    ranges = bin_ranges(_require_time_edges(predictions))
    hazards = bin_hazards(np.asarray(predictions["logits"], dtype=np.float64))  # (N, n_bins)
    surv = bin_survival(hazards)
    haz_cols = [f"h_{r}" for r in ranges]
    surv_cols = [f"S_{r}" for r in ranges]
    rows = [
        {
            "slide_id": sid,
            "patient_code": pcode,
            "risk": f"{float(risk):.4f}",
            "time": f"{float(lbl[0]):.4f}",
            "event": int(lbl[1]),
            **{col: f"{float(v):.4f}" for col, v in zip(haz_cols, h_row)},
            **{col: f"{float(v):.4f}" for col, v in zip(surv_cols, s_row)},
        }
        for sid, pcode, risk, lbl, h_row, s_row in zip(
            predictions["slide_ids"],
            predictions["patient_codes"],
            predictions["probs"],
            predictions["labels"],
            hazards,
            surv,
        )
    ]
    return ["slide_id", "patient_code", "risk", "time", "event", *haz_cols, *surv_cols], rows


def _scalar_label(targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    return targets["label"].reshape(-1)  # one class index or value per slide


def _vector_label(targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    return targets["label"]  # the (n, G) gene vector per slide


def _time_event_label(targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([targets["time"], targets["event"]], dim=-1)  # coupled label: (n, 2) [time, event]


def _identity(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


@dataclass(frozen=True)
class PredictionSpec:
    """How one objective's head output and labels become a predictions dict, and how a human reads it."""

    probs: Callable[[torch.Tensor], torch.Tensor]  # the head output in label units -> the scored prediction
    vector_logits: bool  # the head emits a vector per slide (K classes, G genes, n_bins hazards), else one
    zscored: bool  # the head trains in z-space, mapped back with the task's target_mean and target_std
    label: Callable[[Dict[str, torch.Tensor]], torch.Tensor]  # the batch targets -> the per-slide label
    label_dtype: type  # np.int64 class indices or np.float64 values
    csv: Optional[Callable[[Dict[str, Any]], CsvRows]]  # None: the NPZ is the only per-slide record
    label_space: Callable[[torch.Tensor], torch.Tensor] = _identity  # the label in the predictions' space
    time_edges: bool = False  # the hazard logits need the fold's bin edges to mean anything


PREDICTION_SPECS: Dict[Objective, PredictionSpec] = {
    Objective.binary: PredictionSpec(
        probs=torch.sigmoid,
        vector_logits=False,
        zscored=False,
        label=_scalar_label,
        label_dtype=np.int64,
        csv=_binary_csv,
    ),
    Objective.multiclass: PredictionSpec(
        probs=lambda logits: torch.softmax(logits, dim=-1),
        vector_logits=True,
        zscored=False,
        label=_scalar_label,
        label_dtype=np.int64,
        csv=_multiclass_csv,
    ),
    Objective.regression: PredictionSpec(
        probs=_identity,  # the raw predicted value, no activation
        vector_logits=False,
        zscored=True,
        label=_scalar_label,
        label_dtype=np.float64,
        csv=_regression_csv,
    ),
    Objective.regression_vector: PredictionSpec(
        probs=_identity,  # the predicted gene vector, no activation
        vector_logits=True,
        zscored=True,  # per gene, back to log1p space
        label=_vector_label,
        label_dtype=np.float64,
        csv=None,
        label_space=torch.log1p,  # raw RSEM -> log1p, so aggregate computes per-gene Pearson directly
    ),
    Objective.survival: PredictionSpec(
        probs=survival_risk,  # the scalar risk -sum_j S_j, for ranking
        vector_logits=True,
        zscored=False,
        label=_time_event_label,
        label_dtype=np.float64,
        csv=_survival_csv,
        time_edges=True,
    ),
}


def collect_predictions(
    lit_module: MultiModalLightningModule,
    dataloader: DataLoader,
    *,
    endpoint_type: str,
    num_classes: int,
    target_key: str,
    target_mean: float | np.ndarray = 0.0,
    target_std: float | np.ndarray = 1.0,
    time_edges: np.ndarray | None = None,
    require_labels: bool = True,
) -> Dict[str, Any]:
    """Run ``dataloader`` through ``lit_module`` once (eval, no grad) and return per-slide outputs.

    Returns a self-documenting dict::

        {
            "slide_ids":     list[str],                     # (N,)
            "patient_codes": list[str],                     # (N,)
            "logits":        np.ndarray[float64],           # (N,) binary, regression | (N, K) multiclass
                                                            # | (N, G) expression | (N, n_bins) survival
            "probs":         np.ndarray[float64],           # (N,) binary, regression, survival (risk)
                                                            # | (N, K) multiclass | (N, G) expression
            "labels":        np.ndarray | None,             # int64 (N,) binary, multiclass | float64 (N,)
                                                            # regression | (N, G) expression | (N, 2)
                                                            # survival [time, event]. None if not
                                                            # require_labels
            "endpoint_type": str,                           # the Objective value, e.g. "binary"
            "num_classes":   int,                           # as passed in
            "target_key":    str,                           # e.g. "patient.cancer_type"
            "attention":     list[tuple[str, np.ndarray]],  # (slide_id, (n_tiles, 3)) per slide, empty if none
            "time_edges":    np.ndarray | None,             # (n_bins-1,) survival only
        }

    ``time_edges`` is the survival task's interior bin edges (``task.fit_state().time_edges``). It is None for the
    inference-only construction, which has no fit split to quantile, scoring an external cohort needs
    only the risk score, so the hazards it collects are simply unlabelled.
    """
    objective = Objective(endpoint_type)  # normalize str|Objective -> the enum (fail-fast on an unknown value)
    spec = PREDICTION_SPECS[objective]

    was_training = lit_module.training
    lit_module.eval()
    device = next(lit_module.parameters()).device

    slide_ids: List[str] = []
    patient_codes: List[str] = []
    logit_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    # Captured on this pass; empty for models without an attention branch.
    attention_module = find_attention_module(lit_module)
    attention: List[tuple[str, np.ndarray]] = []
    try:
        captures: AbstractContextManager[list[torch.Tensor]] = (
            capture_input(attention_module) if attention_module is not None else nullcontext([])
        )
        with torch.no_grad(), captures as captured:
            task = lit_module._task
            if task is None:
                raise ValueError("collect_predictions needs a lit_module whose task is built.")
            for batch in dataloader:
                batch = _to_device(batch, device)
                inputs = task.select_inputs(batch)
                captured.clear()
                preds = lit_module(inputs)
                if attention_module is not None and captured:
                    modality = find_tile_modality(batch)
                    if modality is not None:
                        raw = raw_attention_logits(attention_module, captured[-1])
                        attention.extend(extract_records(raw, batch, modality))

                logit_chunks.append(preds["logits"].detach().cpu())
                if require_labels:  # unlabeled inference (bin/predict over all_predict) has no targets to select
                    label_chunks.append(spec.label(task.select_targets(batch)).detach().cpu())
                for ref in batch["slide_ref"]:
                    slide_ids.append(ref.slide_id)
                    patient_codes.append(ref.patient_code)
    finally:
        if was_training:
            lit_module.train()

    logits_t = torch.cat(logit_chunks)
    if spec.zscored:
        # The head trains in z-space when the task standardised the target. Its output goes back to label
        # units (a scalar, or per gene to log1p space) so persisted predictions match the labels. Identity
        # when mean=0, std=1.
        stds = torch.as_tensor(target_std, dtype=logits_t.dtype)
        means = torch.as_tensor(target_mean, dtype=logits_t.dtype)
        logits_t = logits_t * stds + means
    if not spec.vector_logits:
        logits_t = logits_t.reshape(logits_t.shape[0])  # one logit per slide: (N,) or (N, 1) -> (N,)
    probs_out = spec.probs(logits_t).numpy().astype(np.float64)
    logits_out = logits_t.numpy().astype(np.float64)
    # labels_out is None for unlabeled inference (require_labels=False), nothing was collected.
    labels_out: np.ndarray | None = (
        spec.label_space(torch.cat(label_chunks)).numpy().astype(spec.label_dtype) if require_labels else None
    )

    n = len(slide_ids)
    label_n = labels_out.shape[0] if labels_out is not None else n
    if not (n == len(patient_codes) == logits_out.shape[0] == probs_out.shape[0] == label_n):
        raise ValueError(
            f"Length mismatch in collected predictions: slide_ids={n}, "
            f"patient_codes={len(patient_codes)}, logits={logits_out.shape[0]}, "
            f"probs={probs_out.shape[0]}, labels={labels_out.shape[0] if labels_out is not None else 'n/a'}"
        )

    if attention and len(attention) != n:
        raise ValueError(f"attention captured for {len(attention)} slides but {n} were predicted")

    edges = None if time_edges is None else np.asarray(time_edges, dtype=np.float64).ravel()
    if edges is not None and spec.time_edges and edges.size != logits_out.shape[1] - 1:
        raise ValueError(
            f"survival head emits {logits_out.shape[1]} hazard logits per slide, which needs "
            f"{logits_out.shape[1] - 1} interior bin edges; got {edges.size} ({edges.tolist()})."
        )

    return {
        "slide_ids": slide_ids,
        "patient_codes": patient_codes,
        "logits": logits_out,
        "probs": probs_out,
        "labels": labels_out,
        "endpoint_type": objective.value,  # wire = the bare string ("binary"), not str(Objective.binary)
        "num_classes": int(num_classes),
        "target_key": target_key,
        # (slide_id, (n_tiles, 3)) per slide: x, y (level-0 px) + raw attention logit; empty if none.
        "attention": attention,
        "time_edges": edges,  # survival only: the intervals the hazard logits are indexed by
    }


def _require_time_edges(predictions: Dict[str, Any]) -> np.ndarray:
    """The survival bin edges, or a refusal to write hazards nobody can read."""
    edges = predictions.get("time_edges")
    if edges is None:
        raise ValueError(
            "survival predictions carry one hazard logit per time bin, and a bin index means nothing "
            "without the interval it spans — each fold quantiles its own fit split, so the same index "
            "is a different span in every fold. Pass the fold's fit_state.time_edges to collect_predictions."
        )
    return np.asarray(edges, dtype=np.float64)


def write_predictions_npz(predictions: Dict[str, Any], output_path: Path) -> None:
    """Lossless dump for downstream aggregation. Self-documenting: stores endpoint_type,
    num_classes and target_key alongside the arrays, plus ``time_edges`` for survival."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: Dict[str, Any] = {
        "slide_ids": np.array(predictions["slide_ids"], dtype=object),
        "patient_codes": np.array(predictions["patient_codes"], dtype=object),
        "logits": predictions["logits"].astype(np.float64),
        "probs": predictions["probs"].astype(np.float64),
        # int64 (classification) or float64 (regression) as collected
        "labels": np.asarray(predictions["labels"]),
        "endpoint_type": np.array(predictions["endpoint_type"], dtype=object),
        "num_classes": np.array(int(predictions["num_classes"])),
        "target_key": np.array(predictions["target_key"], dtype=object),
    }
    if PREDICTION_SPECS[Objective(predictions["endpoint_type"])].time_edges:
        arrays["time_edges"] = _require_time_edges(predictions)
    np.savez_compressed(output_path, **arrays)


def writes_csv(objective: Objective | str) -> bool:
    """Whether ``objective``'s per-slide predictions have a CSV. Expression has none: the NPZ holds the
    per-gene matrix."""
    return PREDICTION_SPECS[Objective(objective)].csv is not None


def write_predictions_csv(predictions: Dict[str, Any], output_path: Path) -> None:
    """Human-readable predictions, floats to 4 dp. Downstream aggregation uses the NPZ. The columns are the
    objective's (see the ``_*_csv`` functions). An objective without a CSV raises."""
    objective = Objective(predictions["endpoint_type"])
    csv_of = PREDICTION_SPECS[objective].csv
    if csv_of is None:
        raise NotImplementedError(
            f"'{objective.value}' writes no per-slide CSV (the NPZ holds its predictions). Check writes_csv first."
        )
    fieldnames, rows = csv_of(predictions)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
