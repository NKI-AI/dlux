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
"""Compare experiment arms of one study, from their persisted aggregate output.

    compare study=tcga_subtyping cohort=tcga_brca_coad_christiana \
        comparison=tcga_subtyping/crippled_arms

A comparison is a named, study-scoped config naming the arms and the reference. It has the same shape
as an experiment, so it re-runs as an artifact. Output is written to
``studies/<study>/comparisons/<comparison_name>/``.

Reads only what ``aggregate`` persisted under ``results/<arm>/<cohort>/``: the ensemble table for
binary and regression, ``predictions_<field>.csv`` for multiclass and survival (whose metrics need the
class probabilities and event indicator the ensemble table drops). No model is loaded and no run
re-executed, so an un-aggregated arm does not appear here.

Arms that scored different patients are compared on the patients they share, with per-arm coverage
reported alongside, so a difference in reach stays visible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from dlux.config.cohort import Cohort, Study
from dlux.data.errors import BuildDbError
from dlux.eval.compare import compare_arms, resolve_metric, write_comparison
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

from ahcore.utils.io import get_logger, print_config

logger = get_logger(__name__)


def _check_comparison_choice(choice: str, comparison_name: str, study_name: str) -> None:
    """The config's folder must be the study and its leaf must be ``comparison_name``.

    Same guard the experiment configs carry: the output directory is named from ``comparison_name``,
    so a mismatch silently writes one comparison over another.
    """
    parent, _, leaf = str(choice).rpartition("/")
    if not parent:
        sys.exit(f"\n[compare] comparison must be study-scoped: comparison=<study>/<name>, got '{choice}'.")
    if parent != study_name:
        sys.exit(f"\n[compare] comparison '{choice}' belongs to study '{parent}', but study={study_name}.")
    if leaf != comparison_name:
        sys.exit(f"\n[compare] comparison '{choice}' sets comparison_name='{comparison_name}'; expected '{leaf}'.")


@hydra.main(version_base=None, config_path="../config", config_name="compare")
def main(cfg: DictConfig) -> None:
    print_config(cfg, fields=("study", "cohort", "paths", "comparison_name", "arms", "reference", "fields"))

    try:
        study = Study(**OmegaConf.to_container(cfg.study, resolve=True))  # type: ignore[arg-type]
        cohort = Cohort(**OmegaConf.to_container(cfg.cohort, resolve=True))  # type: ignore[arg-type]
    except ValidationError as exc:
        sys.exit(f"\n[compare] {exc}")
    if cohort.name not in study.cohorts:
        sys.exit(
            f"\n[compare] cohort '{cohort.name}' is not part of study '{study.name}' (has: {sorted(study.cohorts)})."
        )

    comparison_name = str(cfg.comparison_name)
    _check_comparison_choice(HydraConfig.get().runtime.choices["comparison"], comparison_name, study.name)

    arms = list(OmegaConf.to_container(cfg.arms, resolve=True))  # type: ignore[arg-type]
    reference = str(cfg.reference)
    fields = list(OmegaConf.to_container(cfg.fields, resolve=True)) if cfg.get("fields") else None  # type: ignore[arg-type]
    results_dir = Path(cfg.paths.studies_dir) / study.name / "results"

    objective = _objective(study, cohort, fields)
    metric_key = cfg.get("metric")
    try:
        spec = resolve_metric(objective, metric_key)  # the metric to score on; None -> endpoint default
        comparisons = compare_arms(
            results_dir=results_dir,
            cohort=cohort.name,
            arms=arms,
            reference=reference,
            objective=objective,
            fields=fields,
            metric_key=metric_key,
        )
    except BuildDbError as exc:
        sys.exit(f"\n[compare] {exc}")

    out_dir = Path(cfg.paths.studies_dir) / study.name / "comparisons" / comparison_name
    # `metric_name` in the config overrides the display label. Otherwise the metric names itself.
    label = str(cfg.get("metric_name") or spec.label)
    summary = write_comparison(comparisons, comparison_name, out_dir, metric_name=label, chance=spec.chance)
    for comparison in comparisons:
        for arm, delta in sorted(comparison.pooled_deltas.items()):
            logger.info("[%s] %s vs %s: pooled delta %+.4f", comparison.field, arm, comparison.reference, delta)
    logger.info("Wrote comparison -> %s", summary.parent)


def _objective(study: Study, cohort: Cohort, fields: list[str] | None) -> str:
    """The objective every compared endpoint shares.

    Scoped to ``study.targets``, not to the whole cohort contract: a cohort commonly offers endpoints
    a given study does not model (a cohort may declare several endpoints across multiple objectives,
    while a study targets only one), and reading the contract would refuse a study that is
    perfectly coherent. Still refuses a mixed set rather than scoring one endpoint with another's metric.
    """
    names = list(fields or study.targets)
    objectives = {cohort.contract[f].objective.value for f in names if f in cohort.contract}
    if len(objectives) != 1:
        sys.exit(
            f"\n[compare] endpoints {names} span objectives {sorted(objectives)}; compare one objective "
            f"at a time (pass fields=[...])."
        )
    return objectives.pop()


if __name__ == "__main__":
    main()
