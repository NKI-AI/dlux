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
"""Compare resplit arms against a measured noise floor.

    compare_resplits study=tcga_subtyping cohort=tcga_brca_coad_christiana \
        comparison=tcga_subtyping/meanmil resplit_name=meanmil task.target.field=cancer_type

Stage 2 of ``docs/specs/COMPARE_SPEC.md``. ``compare`` judges paired per-fold deltas from one nested-CV
sweep against the fold SEM. This judges R independent random splits against training noise measured on
the same splits.

Reads only the rows ``train_resplit`` persisted under
``studies/<study>/resplits/<resplit_name>/<cohort>/rows/`` (rows are field-tagged, so one sweep holds
every endpoint). Nothing is loaded, re-run or re-fitted. Output is written to
``.../results/<comparison_name>/<field>/``, per-field, so one comparison config serves every endpoint.

Reuses ``compare``'s ``comparison=<study>/<name>`` config. The arms and reference are the same question
at a different level of rigour, so they are not restated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from dlux.config.cohort import Cohort, Study
from dlux.data.errors import BuildDbError
from dlux.eval.compare import resolve_metric
from dlux.eval.resplit import (
    check_invariants,
    compare_resplits,
    read_rows,
    summarize_arm,
    write_resplit_comparison,
)
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from ahcore.utils.io import get_logger, print_config

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="compare_resplits")
def main(cfg: DictConfig) -> None:
    print_config(cfg, fields=("study", "cohort", "comparison_name", "arms", "reference", "resplit_name"))

    try:
        study = Study(**OmegaConf.to_container(cfg.study, resolve=True))  # type: ignore[arg-type]
        cohort = Cohort(**OmegaConf.to_container(cfg.cohort, resolve=True))  # type: ignore[arg-type]
    except ValidationError as exc:
        sys.exit(f"\n[compare_resplits] {exc}")
    if cohort.name not in study.cohorts:
        sys.exit(
            f"\n[compare_resplits] cohort '{cohort.name}' is not part of study '{study.name}' "
            f"(has: {sorted(study.cohorts)})."
        )

    target = str(cfg.task.target.field)
    if target not in cohort.contract:
        sys.exit(f"\n[compare_resplits] '{target}' is not an endpoint of cohort '{cohort.name}'.")
    objective = cohort.contract[target].objective.value

    resplit_name = str(cfg.resplit_name)
    base = Path(cfg.paths.studies_dir) / study.name / "resplits" / resplit_name / cohort.name
    arms = list(OmegaConf.to_container(cfg.arms, resolve=True))  # type: ignore[arg-type]
    reference = str(cfg.reference)

    try:
        trials = read_rows(base / "rows")
        notes = check_invariants(trials)
        # The metric is baked into the stored rows by the sweep. Read which one it was from the rows (they
        # must agree) rather than re-choosing here, so the report can never mislabel what was scored.
        metric_keys = {t.metric_key for t in trials}
        if len(metric_keys) > 1:
            sys.exit(f"\n[compare_resplits] rows mix metrics {sorted(k or 'default' for k in metric_keys)}.")
        spec = resolve_metric(objective, metric_keys.pop())
        comparison = compare_resplits(
            trials,
            arms=arms,
            reference=reference,
            field_name=target,
            metric_name=spec.label,
        )
    except BuildDbError as exc:
        sys.exit(f"\n[compare_resplits] {exc}")

    out_dir = (
        base / "results" / str(cfg.comparison_name) / target
    )  # per-field: one comparison config serves every endpoint
    summary = write_resplit_comparison(comparison, str(cfg.comparison_name), out_dir, notes=notes, chance=spec.chance)

    margin = comparison.margin
    logger.info("[%s] R=%d  margin delta = %.4f", comparison.field, comparison.n_seeds, margin)
    for result in comparison.arms:
        stats = summarize_arm(result, margin, comparison.delta_null)
        logger.info(
            "[%s] %s vs %s: median %+.4f | %.0f%% > +delta, %.0f%% < -delta | pctile in null %.2f",
            comparison.field,
            result.arm,
            comparison.reference,
            stats["median"],
            100 * stats["frac_above"],
            100 * stats["frac_below"],
            stats["percentile_in_null"],
        )
    for note in notes:
        logger.warning("%s", note)
    logger.info("Wrote resplit comparison -> %s", summary.parent)


if __name__ == "__main__":
    main()
