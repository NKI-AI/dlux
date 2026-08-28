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

**Endpoint-type aware.** Binary -> ``sigmoid`` (``probs`` shape ``(N,)``); multiclass -> ``softmax``
(``(N, K)``); regression -> the raw value (``probs`` ``(N,)``, no activation; labels kept float, not
class indices); survival -> the scalar risk ``-sum_j S_j`` (``probs`` ``(N,)``) with ``labels`` the
``(N, 2)`` ``[time, event]`` pair, since concordance couples the two.

Survival additionally carries ``time_edges``: ``logits`` is the ``(N, n_bins)`` block of per-bin hazard
logits, and a hazard vector says nothing without the intervals its bins span. The edges are quantiles
of the fold's own fit split, so they differ from fold to fold. Anything comparing or ensembling curves
across folds must read them rather than assume a shared grid.

``patient_code`` is captured at prediction time (off ``SlideView``) so downstream
aggregation does patient-level rollups without ever re-opening the manifest DB.
"""

from __future__ import annotations

import csv
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np
import torch

from dlux.config.cohort import Objective
from dlux.eval.attention import capture_input, extract_records, find_attention_module, find_tile_modality
from dlux.eval.survival_metrics import bin_hazards, bin_ranges, bin_survival
from dlux.tasks.survival import survival_risk

if TYPE_CHECKING:
    import lightning.pytorch as pl
    from torch.utils.data import DataLoader

_SUPPORTED_OBJECTIVES = frozenset(Objective)  # every objective is collectable


def _to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensors to ``device``; pass non-tensors (e.g. SlideView) through."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        moved = [_to_device(v, device) for v in value]
        return type(value)(moved) if isinstance(value, tuple) else moved
    return value


def _probs_from_logits(logits: torch.Tensor, objective: Objective) -> torch.Tensor:
    """Map raw logits to probabilities per objective.

    binary     -> sigmoid, shape (N,)
    multiclass -> softmax over the class axis, shape (N, K)
    """
    if objective == Objective.binary:
        # Accept (N,), (N, 1) -> (N,).
        logits = logits.reshape(logits.shape[0]) if logits.dim() <= 1 else logits.reshape(logits.shape[0], -1)
        if logits.dim() == 2 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return torch.sigmoid(logits)
    if objective == Objective.multiclass:
        return torch.softmax(logits, dim=-1)
    if objective == Objective.regression:
        return logits.reshape(logits.shape[0])  # raw predicted value, no activation -> (N,)
    if objective == Objective.regression_vector:
        return logits  # the predicted gene vector, no activation -> (N, G)
    if objective == Objective.survival:
        return survival_risk(logits)  # scalar risk -sum_j S_j for ranking -> (N,)
    raise ValueError(f"collect_predictions supports {sorted(_SUPPORTED_OBJECTIVES)}, got {objective!r}.")


def collect_predictions(
    lit_module: pl.LightningModule,
    dataloader: DataLoader,
    *,
    endpoint_type: str,
    num_classes: int,
    target_key: str,
    target_mean: float = 0.0,
    target_std: float = 1.0,
    time_edges: np.ndarray | None = None,
    require_labels: bool = True,
) -> Dict[str, Any]:
    """Run ``dataloader`` through ``lit_module`` once (eval, no grad) and return per-slide outputs.

    Returns a self-documenting dict::

        {
            "slide_ids":     list[str],                     # (N,)
            "patient_codes": list[str],                     # (N,)
            "logits":        np.ndarray[float64],           # (N,) binary | (N, K) multiclass
            "probs":         np.ndarray[float64],           # (N,) binary | (N, K) multiclass
            "labels":        np.ndarray[int64],             # (N,)
            "endpoint_type": str,                           # "binary" | "multiclass"
            "num_classes":   int,                           # 1 (binary) | K (multiclass)
            "target_key":    str,                           # e.g. "patient.cancer_type"
            "time_edges":    np.ndarray | None,             # (n_bins-1,) survival only
        }

    ``time_edges`` is the survival task's interior bin edges (``task.time_edges``). It is None for the
    inference-only construction, which has no fit split to quantile, scoring an external cohort needs
    only the risk score, so the hazards it collects are simply unlabelled.
    """
    objective = Objective(endpoint_type)  # normalize str|Objective -> the enum (fail-fast on an unknown value)

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
        with torch.no_grad(), capture_input(attention_module) if attention_module else nullcontext([]) as captured:
            for batch in dataloader:
                batch = _to_device(batch, device)
                inputs = lit_module._task.select_inputs(batch)
                captured.clear()
                preds = lit_module(inputs)
                if attention_module is not None and captured:
                    modality = find_tile_modality(batch)
                    if modality is not None:
                        raw = attention_module.get_raw_attention_logits(captured[-1])
                        attention.extend(extract_records(raw, batch, modality))

                logit_chunks.append(preds["logits"].detach().cpu())
                if require_labels:  # unlabeled inference (bin/predict over all_predict) has no targets to select
                    targets = lit_module._task.select_targets(batch)
                    if objective == Objective.survival:  # coupled label: [time, event] per slide -> (n, 2)
                        labels = torch.stack([targets["time"], targets["event"]], dim=-1).detach().cpu()
                    else:
                        labels = targets["label"].detach().cpu()
                        if (
                            objective != Objective.regression_vector
                        ):  # scalar/class label per slide; keep vector 2D (N, G)
                            labels = labels.reshape(-1)
                    label_chunks.append(labels)
                for sv in batch.get("slide_view", []):
                    slide_ids.append(sv.slide_id)
                    patient_codes.append(sv.patient_code)
    finally:
        if was_training:
            lit_module.train()

    logits_t = torch.cat(logit_chunks)
    # Regression head trains in z-space when the task standardised the target. De-standardise its output
    # back to raw units so persisted predictions match the (raw) labels. Identity when mean=0, std=1.
    if objective == Objective.regression:
        logits_t = logits_t * target_std + target_mean
    elif objective == Objective.regression_vector:
        # Per-gene de-standardise z-space predictions back to log1p space. Labels are stored in the same
        # log1p space (below) so aggregate computes per-gene Pearson directly, no transform knowledge.
        stds = torch.as_tensor(target_std, dtype=logits_t.dtype)
        means = torch.as_tensor(target_mean, dtype=logits_t.dtype)
        logits_t = logits_t * stds + means
    probs_t = _probs_from_logits(logits_t, objective)
    if require_labels:
        labels_t = torch.cat(label_chunks)
        if objective == Objective.regression_vector:
            labels_t = torch.log1p(labels_t)  # raw RSEM -> log1p space (matches the de-standardised preds)

    if objective in (Objective.binary, Objective.regression):
        logits_out = logits_t.reshape(-1).numpy().astype(np.float64)
    else:  # multiclass (N, K) / regression_vector (N, G)
        logits_out = logits_t.numpy().astype(np.float64)
    probs_out = probs_t.numpy().astype(np.float64)
    # Continuous targets (scalar/vector) + survival's [time, event] -> float; class labels are indices.
    # labels_out is None for unlabeled inference (require_labels=False), nothing was collected.
    float_labels = objective in (Objective.regression, Objective.regression_vector, Objective.survival)
    labels_out = labels_t.numpy().astype(np.float64 if float_labels else np.int64) if require_labels else None

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
    if edges is not None and objective == Objective.survival and edges.size != logits_out.shape[1] - 1:
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
            "is a different span in every fold. Pass time_edges=task.time_edges to collect_predictions."
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
    if predictions["endpoint_type"] == Objective.survival.value:
        arrays["time_edges"] = _require_time_edges(predictions)
    np.savez_compressed(output_path, **arrays)


def write_predictions_csv(predictions: Dict[str, Any], output_path: Path) -> None:
    """Human-readable predictions, floats to 4 dp. Downstream aggregation uses the NPZ.

    Binary: header ``slide_id,patient_code,logit,prob,label`` (integer label).
    Regression: header ``slide_id,patient_code,pred,label`` (float label).
    Multiclass: header ``slide_id,patient_code,prob_0..prob_{K-1},pred,label``, the per-class
    softmax probs, the argmax predicted class, and the true class index. The NPZ still carries the
    lossless arrays; this CSV is for human inspection.
    Survival: header ``slide_id,patient_code,risk,time,event,h_<range>...,S_<range>...``, the ranking
    score, the observed follow-up, then the per-bin hazard and survival probability with each column
    named by the interval it covers (``h_183-402``), so a row is readable without the metadata.
    """
    endpoint_type = predictions["endpoint_type"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if endpoint_type == Objective.survival.value:
        ranges = bin_ranges(_require_time_edges(predictions))
        hazards = bin_hazards(np.asarray(predictions["logits"], dtype=np.float64))  # (N, n_bins)
        surv = bin_survival(hazards)
        haz_cols = [f"h_{r}" for r in ranges]
        surv_cols = [f"S_{r}" for r in ranges]
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["slide_id", "patient_code", "risk", "time", "event", *haz_cols, *surv_cols]
            )
            writer.writeheader()
            for sid, pcode, risk, lbl, h_row, s_row in zip(
                predictions["slide_ids"],
                predictions["patient_codes"],
                predictions["probs"],
                predictions["labels"],
                hazards,
                surv,
            ):
                writer.writerow(
                    {
                        "slide_id": sid,
                        "patient_code": pcode,
                        "risk": f"{float(risk):.4f}",
                        "time": f"{float(lbl[0]):.4f}",
                        "event": int(lbl[1]),
                        **{col: f"{float(v):.4f}" for col, v in zip(haz_cols, h_row)},
                        **{col: f"{float(v):.4f}" for col, v in zip(surv_cols, s_row)},
                    }
                )
        return

    if endpoint_type == "multiclass":
        probs = np.asarray(predictions["probs"], dtype=np.float64)  # (N, K)
        k = probs.shape[1]
        prob_cols = [f"prob_{c}" for c in range(k)]
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["slide_id", "patient_code", *prob_cols, "pred", "label"])
            writer.writeheader()
            for sid, pcode, row, lbl in zip(
                predictions["slide_ids"], predictions["patient_codes"], probs, predictions["labels"]
            ):
                writer.writerow(
                    {
                        "slide_id": sid,
                        "patient_code": pcode,
                        **{col: f"{float(v):.4f}" for col, v in zip(prob_cols, row)},
                        "pred": int(np.argmax(row)),
                        "label": int(lbl),
                    }
                )
        return

    if endpoint_type == "regression":
        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["slide_id", "patient_code", "pred", "label"])
            writer.writeheader()
            for sid, pcode, pred, lbl in zip(
                predictions["slide_ids"],
                predictions["patient_codes"],
                predictions["logits"],
                predictions["labels"],
            ):
                writer.writerow(
                    {
                        "slide_id": sid,
                        "patient_code": pcode,
                        "pred": f"{float(pred):.4f}",
                        "label": f"{float(lbl):.4f}",
                    }
                )
        return

    if endpoint_type != "binary":
        raise NotImplementedError(
            f"write_predictions_csv supports binary, multiclass, regression and survival (got "
            f"'{endpoint_type}'). regression_vector writes no per-slide CSV (the NPZ holds the "
            "per-gene matrix)."
        )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slide_id", "patient_code", "logit", "prob", "label"])
        writer.writeheader()
        for sid, pcode, lgt, prb, lbl in zip(
            predictions["slide_ids"],
            predictions["patient_codes"],
            predictions["logits"],
            predictions["probs"],
            predictions["labels"],
        ):
            writer.writerow(
                {
                    "slide_id": sid,
                    "patient_code": pcode,
                    "logit": f"{float(lgt):.4f}",
                    "prob": f"{float(prb):.4f}",
                    "label": int(lbl),
                }
            )
