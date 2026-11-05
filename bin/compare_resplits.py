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
        comparison=tcga_subtyping/meanmil resplit_name=meanmil

Stage 2 of ``docs/specs/COMPARE_SPEC.md``. ``compare`` judges paired per-fold deltas from one nested-CV
sweep against the fold SEM. This judges R independent random splits against training noise measured on
the same splits.

Reads only the rows ``train_resplit`` persisted under
``studies/<study>/resplits/<resplit_name>/<cohort>/rows/`` (rows are field-tagged, so one sweep holds
every endpoint). Nothing is loaded, re-run or re-fitted. Output is written to
``studies/<study>/comparisons/<comparison_name>/resplit_<resplit_name>/<field>/``, per field, beside
``compare``'s ``nested_cv/``. ``fields`` narrows the endpoints, and by default every
endpoint the rows hold is compared.

Reuses ``compare``'s ``comparison=<study>/<name>`` config. The arms and reference are the same question
at a different level of rigour, so they are not restated.
"""

from __future__ import annotations

import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from ahcore.utils.io import get_logger, print_config

from dlux.config.cohort import Cohort, Study
from dlux.config.resolvers import register_resolvers
from dlux.data import layout
from dlux.data.errors import BuildDbError
from dlux.eval.compare import check_comparison_choice, resolve_metric
from dlux.eval.resplit import (
    check_invariants,
    compare_resplits,
    read_rows,
    summarize_arm,
    write_resplit_comparison,
)

logger = get_logger(__name__)

register_resolvers()


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

    try:
        check_comparison_choice(HydraConfig.get().runtime.choices["comparison"], str(cfg.comparison_name), study.name)
    except BuildDbError as exc:
        sys.exit(f"\n[compare_resplits] {exc}")

    resplit_name = str(cfg.resplit_name)
    arms = list(OmegaConf.to_container(cfg.arms, resolve=True))  # type: ignore[arg-type]
    reference = str(cfg.reference)

    try:
        trials = read_rows(layout.resplit_rows_dir(cfg.paths.studies_dir, study.name, resplit_name, cohort.name))
        notes = check_invariants(trials)
    except BuildDbError as exc:
        sys.exit(f"\n[compare_resplits] {exc}")
    # The rows are field-tagged, so one sweep holds every endpoint it trained.
    fields = list(OmegaConf.to_container(cfg.fields, resolve=True)) if cfg.get("fields") else None  # type: ignore[arg-type]
    for field in fields or sorted({t.field for t in trials}):
        _compare_field(cfg, study.name, cohort, trials, notes, field, arms=arms, reference=reference)


def _compare_field(cfg, study_name, cohort, trials, notes, field, *, arms, reference) -> None:
    if field not in cohort.contract:
        sys.exit(f"\n[compare_resplits] '{field}' is not an endpoint of cohort '{cohort.name}'.")
    objective = cohort.contract[field].objective.value
    field_trials = [t for t in trials if t.field == field]
    try:
        # The metric is baked into the stored rows by the sweep. Read which one it was from the rows (they
        # must agree) rather than re-choosing here, so the report can never mislabel what was scored.
        metric_keys = {t.metric_key for t in field_trials}
        if len(metric_keys) > 1:
            sys.exit(f"\n[compare_resplits] '{field}' rows mix metrics {sorted(k or 'default' for k in metric_keys)}.")
        spec = resolve_metric(objective, metric_keys.pop() if metric_keys else None)
        comparison = compare_resplits(
            field_trials,
            arms=arms,
            reference=reference,
            field_name=field,
            metric_name=spec.label,
        )
    except BuildDbError as exc:
        sys.exit(f"\n[compare_resplits] {exc}")

    out_dir = layout.resplit_comparison_dir(
        cfg.paths.studies_dir, study_name, cfg.comparison_name, cfg.resplit_name, field
    )
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
