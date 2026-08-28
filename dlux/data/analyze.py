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
"""Deterministic, contract-driven analysis of a built manifest DB.

Reads the sqlite manifest directly (no ahcore dependency, so it is runnable
standalone and testable) plus the ``Cohort`` contract, and emits a markdown
summary + a single multi-panel overview figure. No config knobs beyond the
dataset itself. Each field's raw values are read from its ``source.column``.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
# silence matplotlib's superfluous INFO logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from dlux.config.cohort import Cohort, Objective  # noqa: E402
from dlux.data import bulk_rna_matrix  # noqa: E402
from dlux.data.splits import parse_cv_split  # noqa: E402
from dlux.eval.survival_metrics import kaplan_meier  # noqa: E402
from dlux.modalities import BulkRNA  # noqa: E402


# -- facet helpers -----------------------------------------------------------
def _is_categorical(field) -> bool:
    t = field.source.transform
    return t is not None and t.categorical is not None


def _cat_map(field) -> dict:
    return field.source.transform.categorical


def _field_series(df: pd.DataFrame, field) -> pd.Series:
    """The field's raw values, read from its source.column (defaults to the field key).

    Empty for an expression endpoint, its target is the external RNA matrix, not a sheet column."""
    col = field.source.column
    return df[col] if (col is not None and col in df.columns) else pd.Series(dtype="object")


def _read_db(db_path: Path):
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        patients = pd.read_sql("SELECT id, patient_code FROM patient", con)
        labels = pd.read_sql("SELECT key, value, patient_id FROM patient_labels", con)
        images = pd.read_sql("SELECT patient_id, mpp, height, width FROM image", con)
        n_masks = int(pd.read_sql("SELECT COUNT(*) AS n FROM mask", con)["n"].iloc[0])
        splits = pd.read_sql(
            "SELECT sd.version AS version, s.category AS category, s.patient_id AS patient_id "
            "FROM split s JOIN split_definitions sd ON sd.id = s.split_definition_id",
            con,
        )
    finally:
        con.close()
    return patients, labels, images, n_masks, splits


def _per_patient(patients: pd.DataFrame, labels: pd.DataFrame, images: pd.DataFrame) -> pd.DataFrame:
    wide = labels.pivot_table(index="patient_id", columns="key", values="value", aggfunc="first")
    df = patients.set_index("id").join(wide)
    df = df.join(images.groupby("patient_id").size().rename("n_images"))
    df["n_images"] = df["n_images"].fillna(0).astype(int)
    return df


def _endpoint_class_frame(values: pd.Series, field) -> pd.DataFrame:
    """Rows of (raw value, mapped class) for values inside the contract map."""
    valid = values.dropna().astype(str)
    mapping = _cat_map(field)
    valid = valid[valid.isin(mapping)]
    return pd.DataFrame({"value": valid.values, "class": valid.map(mapping).astype(int).astype(str).values})


# max/mean slides-per-patient at or above which the per-slide weighting warrants a warning. Below it
# the distribution is near-uniform and slide- and patient-weighting coincide.
_SLIDE_SKEW_LIMIT = 5.0


def _slide_skew_warning(spp) -> list[str]:
    """Warn when slides-per-patient is skewed enough that per-slide quantities stop being per-patient.

    Loss and the streamed train/validate metrics are computed per slide, so a patient with many slides
    carries proportionally more weight. That is harmless when every patient has a similar count (the
    bias comes from the variance, not the mean), but `validate/loss` selects the checkpoint, so a
    skewed cohort lets one patient cast a heavier vote on which epoch is kept.

    This is a property of the cohort that the training setup is sensitive to."""
    mean, mx = float(spp.mean()), int(spp.max())
    if mean <= 0 or mx / mean < _SLIDE_SKEW_LIMIT:
        return []
    return [
        f"- **Slides per patient are skewed** (max {mx} vs mean {mean:.2f}). Loss and the streamed "
        f"train/validate metrics are per-SLIDE, so that patient weighs ~{mx / mean:.0f}x a typical one "
        f"— including in `validate/loss`, which selects the checkpoint. Reported TEST metrics are "
        f"per-patient and unaffected.",
    ]


# Majority/minority ratio at or above which the report names the class-weighting knob. Below it,
# inverse-frequency weights are near 1 and the override adds nothing.
_IMBALANCE_LIMIT = 3.0


def class_weight_snippet() -> str:
    """The `task.loss.weight` override, in the shape an experiment config sets it.

    Field-independent: the weighting is a per-run choice and a run trains one endpoint, so the arm
    does not name the field it is already targeting."""
    return "task:\n  loss:\n    weight: balanced"


def _imbalance_lines(counts: list[int]) -> list[str]:
    """Name the class-weighting override when the endpoint is imbalanced enough for it to matter.

    Reports the ratio and the override; whether to use it is a modelling choice, not a data fact."""
    counts = [n for n in counts if n]
    if len(counts) < 2 or max(counts) / min(counts) < _IMBALANCE_LIMIT:
        return []
    return [
        "",
        f"Class imbalance {max(counts) / min(counts):.1f}:1. For inverse-frequency loss weighting "
        f"(computed per fold on the fit split), add to the experiment config:",
        "",
        "```yaml",
        class_weight_snippet(),
        "```",
    ]


@dataclass
class _Ctx:
    """Inputs the endpoint renderers read from.

    Carries only modality-agnostic context; a renderer needing extra inputs (e.g. the expression
    rider's RNA matrix) resolves them itself from the cohort + ``cohorts_dir``."""

    df: pd.DataFrame  # per-patient wide frame (sheet columns)
    splits: pd.DataFrame  # split versions + categories
    cohort_name: str
    cohorts_dir: Optional[Path]  # cohort data root; None -> derived-artifact side-inputs unavailable
    n_patients: int
    _cache: dict = dc_field(default_factory=dict)  # renderer-owned memo for read-once side-inputs


# -- endpoint renderers (dispatched on objective via _RENDERERS) -------------
# A renderer owns one endpoint kind's summary bullets + figure panel + per-fold balance cell.
class _SheetRenderer:
    """Central sheet-tabular endpoints: categorical (binary/multiclass) + scalar (regression)."""

    def summary_lines(self, name: str, field, ctx: "_Ctx") -> list[str]:
        present = _field_series(ctx.df, field).dropna()
        n_missing = ctx.n_patients - len(present)
        if _is_categorical(field):
            frame = _endpoint_class_frame(present, field)
            n_excluded = len(present) - len(frame)
            lines = []
            counts = []
            for cls in sorted(frame["class"].unique(), key=int):
                raws = sorted(r for r, c in _cat_map(field).items() if str(c) == cls)
                n = int((frame["class"] == cls).sum())
                counts.append(n)
                lines.append(f"- class {cls} ({', '.join(map(str, raws))}): {n} ({100 * n / max(len(frame), 1):.1f}%)")
            lines.append(f"- excluded (value not in map): {n_excluded} · missing: {n_missing}")
            return lines + _imbalance_lines(counts)
        vals = pd.to_numeric(present, errors="coerce").dropna()
        if len(vals):
            return [
                f"- n {len(vals)} · median {vals.median():.3g} "
                f"[{vals.quantile(0.25):.3g}–{vals.quantile(0.75):.3g}] · "
                f"min {vals.min():.3g} · max {vals.max():.3g} · missing {n_missing}"
            ]
        return [f"- no numeric values · missing {n_missing}"]

    def draw_panel(self, ax, name: str, field, ctx: "_Ctx") -> None:
        series = _field_series(ctx.df, field)
        if _is_categorical(field):
            frame = _endpoint_class_frame(series, field)
            order = sorted(
                frame["value"].unique(), key=lambda v: (int(frame.loc[frame.value == v, "class"].iloc[0]), v)
            )
            sns.countplot(data=frame, x="value", hue="class", order=order, dodge=False, ax=ax)
            leg = ax.get_legend()  # countplot owns the hue legend; only restyle a real one (else empty-box warning)
            if leg is not None:
                leg.set_title("class")
            elif ax.get_legend_handles_labels()[0]:
                ax.legend(title="class", fontsize="small")
            ax.set_xlabel(name)
            ax.set_ylabel("patients")
        else:
            vals = pd.to_numeric(series, errors="coerce").dropna()
            label = name
            transform = field.source.transform
            if transform is not None and transform.numeric == "log1p":
                vals = np.log1p(vals)
                label = f"{name} (log1p)"
            sns.violinplot(y=vals.values, ax=ax, color="#8172B3", inner="quartile", cut=0)
            for collection in ax.collections:
                collection.set_alpha(0.35)
            sns.stripplot(y=vals.values, ax=ax, color="#2f2f2f", size=3, alpha=0.6, jitter=0.18)
            ax.set_ylabel(label)
        ax.set_title(f"{name} ({field.objective.value})")

    def fold_balance_header(self, field) -> Optional[str]:
        return "TEST class balance" if _is_categorical(field) else "TEST target median [IQR]"

    def fold_balance_cell(self, field, test_ids, ctx: "_Ctx") -> str:
        col = field.source.column
        present = (
            ctx.df.loc[ctx.df.index.isin(test_ids), col]
            if (col and col in ctx.df.columns)
            else pd.Series(dtype="object")
        )
        if _is_categorical(field):
            bal = present.dropna().astype(str).map(_cat_map(field)).value_counts().sort_index().to_dict()
            return ", ".join(f"cls {k}: {v}" for k, v in bal.items())
        vals = pd.to_numeric(present, errors="coerce").dropna()
        return f"{vals.median():.3g} [{vals.quantile(0.25):.3g}–{vals.quantile(0.75):.3g}]" if len(vals) else "—"


class _RegressionVectorRenderer:
    """RNA expression rider: target is the external matrix, not a sheet column."""

    def _summary(self, ctx: "_Ctx") -> Optional[dict]:
        """Characterise the cohort's RNA matrix (read once, cached on ctx). None if unavailable.

        Every expression endpoint in a cohort shares one matrix, so a single memo slot suffices."""
        if "expr_summary" not in ctx._cache:
            path = bulk_rna_matrix.matrix_path(ctx.cohorts_dir, ctx.cohort_name) if ctx.cohorts_dir else None
            ctx._cache["expr_summary"] = bulk_rna_matrix.matrix_summary(path)
        return ctx._cache["expr_summary"]

    def summary_lines(self, name: str, field, ctx: "_Ctx") -> list[str]:
        g = self._summary(ctx)
        genes = f" · {g['n_genes']} genes" if g else ""
        matrix_n = f" (matrix has {g['n_patients']})" if g else ""
        # How many of this cohort's patients the matrix can label, a fact about the data, so the
        # modality answers it by intersecting the matrix index with the manifest. Deriving it from the
        # split table instead would report 0 for any cohort scored end-to-end (no CV splits to count),
        # and would conflate "has RNA" with "entered a fold", patients can be dropped for slide
        # reasons too. Fold membership is a splits fact and is reported under `## Splits`.
        have_rna = len(
            BulkRNA.coverage(
                cohorts_dir=ctx.cohorts_dir,
                cohort_name=ctx.cohort_name,
                manifest_codes={str(c) for c in ctx.df["patient_code"]},
            )
        )
        return [
            f"- external RNA matrix target · {have_rna}/{ctx.n_patients} patients have RNA"
            f"{matrix_n}{genes} · unstratified folds"
        ]

    def draw_panel(self, ax, name: str, field, ctx: "_Ctx") -> None:
        expr = self._summary(ctx)
        if expr is not None:
            sns.histplot(x=expr["gene_means"], ax=ax, color="#8172B3", bins=60)
            ax.set_xlabel("per-gene mean log1p expression")
            ax.set_ylabel("genes")
            ax.set_title(f"{name} ({field.objective.value}) — {expr['n_patients']}×{expr['n_genes']}")
        else:
            ax.text(0.5, 0.5, "external RNA matrix\n(not found)", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{name} ({field.objective.value})")

    def fold_balance_header(self, field) -> Optional[str]:
        return None  # a vector target has no per-fold scalar balance

    def fold_balance_cell(self, field, test_ids, ctx: "_Ctx") -> str:
        return "—"  # never reached (header is None)


class _SurvivalRenderer:
    """Time-to-event endpoint (event, time). The discrete-time model uses the follow-up time (bin edges
    are the fit-split event-time quantiles), so a violin of the 0/1 event bit is meaningless. It drops
    the time axis entirely. This renders the cohort survival instead. Overview grid (one ax): single-arm
    Kaplan-Meier. Standalone figure: KM + a follow-up-time histogram split by event vs censored."""

    n_panels = 2

    @staticmethod
    def _time_event(field, ctx: "_Ctx") -> tuple[np.ndarray, np.ndarray]:
        """Aligned (time, event) over patients with a parseable event in {0,1} and a positive time."""
        ev_col, tm_col = ctx.df.get(field.source.column), ctx.df.get(field.source.time_column)
        if ev_col is None or tm_col is None:
            return np.array([], dtype=float), np.array([], dtype=int)
        ev = pd.to_numeric(ev_col, errors="coerce")
        tm = pd.to_numeric(tm_col, errors="coerce")
        ok = ev.isin([0, 1]) & tm.notna() & (tm > 0)
        return tm[ok].to_numpy(dtype=float), ev[ok].to_numpy(dtype=int)

    def summary_lines(self, name: str, field, ctx: "_Ctx") -> list[str]:
        time, event = self._time_event(field, ctx)
        n_missing = ctx.n_patients - len(time)
        if not len(time):
            return [f"- no usable (time, event) rows · missing {n_missing}"]
        n_ev = int(event.sum())
        q = np.quantile(time, [0.25, 0.5, 0.75])
        et = time[event == 1]
        med_ev = f"{np.median(et):.3g}" if len(et) else "—"
        return [
            f"- n {len(time)} · events {n_ev} ({100 * n_ev / len(time):.1f}%) · "
            f"censored {len(time) - n_ev} ({100 * (len(time) - n_ev) / len(time):.1f}%) · missing {n_missing}",
            f"- follow-up median {q[1]:.3g} [{q[0]:.3g}–{q[2]:.3g}] · median time-to-event {med_ev}",
        ]

    def _draw_km(self, ax, name: str, field, ctx: "_Ctx") -> None:
        time, event = self._time_event(field, ctx)
        if len(time):
            steps, surv = kaplan_meier(time, event)
            ax.step(steps, surv, where="post", color="#8172B3")
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("follow-up time")
            ax.set_ylabel("survival probability")
        else:
            ax.text(0.5, 0.5, "no usable\n(time, event) rows", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"{name} ({field.objective.value}) — {len(time)} pts, {int(event.sum())} events")

    def draw_panel(self, ax, name: str, field, ctx: "_Ctx") -> None:
        self._draw_km(ax, name, field, ctx)  # overview grid (one ax): KM only

    def draw_endpoint(self, axes, name: str, field, ctx: "_Ctx") -> None:
        self._draw_km(axes[0], name, field, ctx)
        time, event = self._time_event(field, ctx)
        hist_ax = axes[1]
        if len(time):
            frame = pd.DataFrame({"time": time, "status": np.where(event == 1, "event", "censored")})
            sns.histplot(
                data=frame,
                x="time",
                hue="status",
                multiple="stack",
                bins=30,
                ax=hist_ax,
                palette={"event": "#8172B3", "censored": "#B0B0B0"},
            )
            hist_ax.set_xlabel("follow-up time")
            hist_ax.set_ylabel("patients")
        else:
            hist_ax.text(
                0.5, 0.5, "no usable\n(time, event) rows", ha="center", va="center", transform=hist_ax.transAxes
            )
        hist_ax.set_title(f"{name} — follow-up time by status")

    def fold_balance_header(self, field) -> Optional[str]:
        return "TEST events/censored"

    def fold_balance_cell(self, field, test_ids, ctx: "_Ctx") -> str:
        ev = pd.to_numeric(ctx.df.loc[ctx.df.index.isin(test_ids), field.source.column], errors="coerce")
        ev = ev[ev.isin([0, 1])]
        n_ev = int((ev == 1).sum())
        return f"{n_ev} ev / {len(ev) - n_ev} cens"


_SHEET = _SheetRenderer()
_REGRESSION_VECTOR = _RegressionVectorRenderer()
_SURVIVAL = _SurvivalRenderer()

# One renderer per objective (the input-data view for analyze's summary + fold-balance panels): the
# scalar sheet endpoints share _SHEET, expression is the per-gene matrix, survival the (time, event) pair.
_RENDERERS = {
    Objective.binary: _SHEET,
    Objective.multiclass: _SHEET,
    Objective.regression: _SHEET,
    Objective.regression_vector: _REGRESSION_VECTOR,
    Objective.survival: _SURVIVAL,
}


def _renderer(field):
    return _RENDERERS[field.objective]


def _write_summary(
    cohort: Cohort,
    images: pd.DataFrame,
    n_masks: int,
    dataset_fig: str,
    endpoint_figs: dict[str, str],
    ctx: "_Ctx",
    out_dir: Path,
) -> None:
    n_images = len(images)
    spp = ctx.df["n_images"]
    lines = [
        f"# {cohort.name} — dataset analysis",
        "",
        "## Overview",
        "",
        f"- patients: **{ctx.n_patients}**",
        f"- slides: **{n_images}**  (per patient: mean {spp.mean():.2f}, median {spp.median():.0f}, min {spp.min()}, max {spp.max()})",
        *_slide_skew_warning(spp),
        f"- masks: **{n_masks}**",
        f"- slides with missing geometry: {int(images['mpp'].isna().sum())}",
        "",
        f"![dataset summary]({dataset_fig})",
        "",
        "[▸ combined overview (dataset + all endpoints)](overview.png)",
        "",
        "## Endpoints",
        "",
    ]
    for name, field in cohort.contract.items():
        lines.append(f"### `{name}` ({field.objective.value})")
        lines += _renderer(field).summary_lines(name, field, ctx)
        lines += ["", f"![{name}]({endpoint_figs[name]})", ""]

    # Splits, nested CV collapsed to outer-fold level (TEST is shared across inner folds).
    lines += ["## Splits", ""]
    cv_prefixes: dict[str, dict[int, str]] = {}
    plain_versions: list[str] = []
    for version in sorted(ctx.splits["version"].unique()):
        parsed = parse_cv_split(version)
        if parsed is not None:
            field_name, outer, _ = parsed
            cv_prefixes.setdefault(field_name, {}).setdefault(outer, version)  # one rep per outer
        else:
            plain_versions.append(version)

    for prefix, outer_reps in cv_prefixes.items():
        field = cohort.contract.get(prefix)
        header = _renderer(field).fold_balance_header(field) if field is not None else None
        lines.append(f"### `{prefix}` — nested CV ({len(outer_reps)} outer folds; TEST shared across inner)")
        lines.append("")
        lines += (
            [f"| outer fold | TEST n | {header} |", "| --- | --- | --- |"]
            if header is not None
            else ["| outer fold | TEST n |", "| --- | --- |"]
        )
        for outer in sorted(outer_reps):
            ver = outer_reps[outer]
            test_ids = ctx.splits[(ctx.splits.version == ver) & (ctx.splits.category == "TEST")]["patient_id"]
            if header is None:
                lines.append(f"| o{outer} | {len(test_ids)} |")
            else:
                cell = _renderer(field).fold_balance_cell(field, test_ids, ctx)
                lines.append(f"| o{outer} | {len(test_ids)} | {cell} |")
        lines.append("")

    for version in plain_versions:
        cats = ctx.splits[ctx.splits.version == version]["category"].value_counts().to_dict()
        lines.append(f"- `{version}`: " + ", ".join(f"{k} {v}" for k, v in cats.items()))
    lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines))


def _hist_or_bar(ax, series: pd.Series, xlabel: str, color: str, fmt: str = "{:.3g}") -> None:
    """Histogram, or a single labelled bar when the values are (near-)constant."""
    vals = series.dropna()
    if vals.nunique() <= 1:
        label = fmt.format(vals.iloc[0]) if len(vals) else "n/a"
        ax.bar([label], [len(vals)], color=color)
        ax.set_ylabel("count")
    else:
        sns.histplot(x=vals, ax=ax, color=color)
    ax.set_xlabel(xlabel)


def _draw_dataset_panels(axes, df: pd.DataFrame, images: pd.DataFrame) -> None:
    """Draw the three endpoint-agnostic panels onto axes[0:3]."""
    spp = df["n_images"].value_counts().sort_index()
    axes[0].bar([str(i) for i in spp.index], spp.values, color="#4C72B0", width=0.6)
    axes[0].set_title("Slides per patient")
    axes[0].set_xlabel("slides")
    axes[0].set_ylabel("patients")

    _hist_or_bar(axes[1], pd.to_numeric(images["mpp"], errors="coerce"), "µm/px", "#55A868", fmt="{:.4g}")
    axes[1].set_title("Slide mpp")

    megapixels = (
        pd.to_numeric(images["height"], errors="coerce") * pd.to_numeric(images["width"], errors="coerce") / 1e6
    )
    _hist_or_bar(axes[2], megapixels, "megapixels", "#C44E52", fmt="{:.0f}")
    axes[2].set_title("Slide size")


def _plot_dataset(cohort: Cohort, df: pd.DataFrame, images: pd.DataFrame, out_dir: Path) -> str:
    """Endpoint-agnostic dataset figure (for the markdown Overview section)."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), squeeze=False)
    _draw_dataset_panels(axes.flatten(), df, images)
    fig.suptitle(f"{cohort.name} — dataset summary", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "dataset.png", dpi=150)
    plt.close(fig)
    return "dataset.png"


def _plot_overview(cohort: Cohort, images: pd.DataFrame, ctx: "_Ctx", out_dir: Path) -> str:
    """The all-in-one dashboard: dataset panels + every endpoint panel (each drawn by its renderer)."""
    sns.set_theme(style="whitegrid")
    endpoints = list(cohort.contract.items())
    n_panels = 3 + len(endpoints)
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()
    _draw_dataset_panels(axes[:3], ctx.df, images)
    for i, (name, field) in enumerate(endpoints):
        _renderer(field).draw_panel(axes[3 + i], name, field, ctx)
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f"{cohort.name} — overview", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "overview.png", dpi=150)
    plt.close(fig)
    return "overview.png"


def _plot_endpoint(name: str, field, ctx: "_Ctx", out_dir: Path) -> str:
    """One standalone figure per endpoint (embedded in its markdown section), drawn by its renderer."""
    sns.set_theme(style="whitegrid")
    r = _renderer(field)
    n = getattr(r, "n_panels", 1)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    row = list(axes[0])
    if hasattr(r, "draw_endpoint"):
        r.draw_endpoint(row, name, field, ctx)  # renderer owns a multi-panel layout
    else:
        r.draw_panel(row[0], name, field, ctx)
    fig.tight_layout()
    filename = f"endpoint_{name}.png"
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)
    return filename


def analyze_db(cohort: Cohort, database_uri: str, out_dir: Path, cohorts_dir: Optional[Path] = None) -> Path:
    """Write summary.md + overview.png for ``cohort``'s DB (at ``database_uri``) into ``out_dir``.

    ``cohorts_dir`` is the cohort data root; renderers needing derived-artifact side-inputs (the
    expression rider's RNA matrix) resolve them from it. Analysis still runs without it, those
    panels just report the artifact as unavailable."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(database_uri.replace("sqlite:///", ""))
    patients, labels, images, n_masks, splits = _read_db(db_path)
    df = _per_patient(patients, labels, images)
    ctx = _Ctx(
        df=df,
        splits=splits,
        cohort_name=cohort.name,
        cohorts_dir=Path(cohorts_dir) if cohorts_dir is not None else None,
        n_patients=len(df),
    )
    _plot_overview(cohort, images, ctx, out_dir)
    dataset_fig = _plot_dataset(cohort, df, images, out_dir)
    endpoint_figs = {name: _plot_endpoint(name, field, ctx, out_dir) for name, field in cohort.contract.items()}
    _write_summary(cohort, images, n_masks, dataset_fig, endpoint_figs, ctx, out_dir)
    print(
        f"[analyze_db] wrote {out_dir}/summary.md + overview.png + dataset.png + {len(endpoint_figs)} endpoint figures"
    )
    return out_dir
