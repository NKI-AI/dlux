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
"""Tests for dlux.tracking: the sqlite-backed MLflow logger builder (offline, no server)."""

from __future__ import annotations

from dlux.tracking import build_mlflow_logger


def test_build_logger_creates_artifact_dir_and_sqlite_uri(tmp_path):
    logger = build_mlflow_logger(mlflow_dir=tmp_path, experiment_name="exp_sweep", run_name="cancer_type_cv_o0_i0")
    assert (tmp_path / "artifacts").is_dir()
    assert logger._tracking_uri == f"sqlite:///{tmp_path}/mlflow.db"


def test_metric_round_trip_through_sqlite(tmp_path):
    """End-to-end proof the sqlite backend works on our mlflow: log -> read back."""
    logger = build_mlflow_logger(
        mlflow_dir=tmp_path,
        experiment_name="exp_sweep",
        run_name="cancer_type_cv_o0_i0",
        tags={"target": "cancer_type"},
    )
    logger.log_metrics({"test/auroc": 0.94}, step=0)
    logger.experiment.flush() if hasattr(logger.experiment, "flush") else None

    # the sqlite db file now exists and the run carries our metric + run name
    assert (tmp_path / "mlflow.db").exists()
    run = logger.experiment.get_run(logger.run_id)
    assert run.data.metrics["test/auroc"] == 0.94
    assert run.info.run_name == "cancer_type_cv_o0_i0"

    exp = logger.experiment.get_experiment_by_name("exp_sweep")
    assert exp is not None
    logger.finalize("success")


def test_nested_config_logs_as_flattened_params(tmp_path):
    """log_hyperparams flattens a nested config into '/'-delimited mlflow params."""
    logger = build_mlflow_logger(mlflow_dir=tmp_path, experiment_name="exp_sweep", run_name="fold")
    logger.log_hyperparams(
        {
            "study": {"name": "tcga_coad_regression"},
            "task": {"target": "leukocyte_fraction", "target_normalize": "zscore"},
            "trainer": {"max_epochs": 60},
        }
    )
    params = logger.experiment.get_run(logger.run_id).data.params
    assert params["study/name"] == "tcga_coad_regression"
    assert params["task/target"] == "leukocyte_fraction"
    assert params["task/target_normalize"] == "zscore"
    assert params["trainer/max_epochs"] == "60"
    logger.finalize("success")
