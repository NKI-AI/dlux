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
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from dlux.config.resolvers import project_dir, register_resolvers, repo_path, run_key


def _project(root: Path, name: str = "proj") -> Path:
    project = root / name
    (project / "catalog" / "cohort").mkdir(parents=True)
    (project / "config").mkdir()
    (project / "catalog" / "cohort" / "c1.yaml").write_text("name: c1\n")
    return project


def test_repo_path_takes_a_relative_path_from_the_bazel_workspace(tmp_path: Path, monkeypatch):
    (tmp_path / "a" / "b").mkdir(parents=True)
    monkeypatch.setenv("BUILD_WORKSPACE_DIRECTORY", str(tmp_path))
    monkeypatch.chdir("/")
    assert repo_path("a/b") == str(tmp_path / "a" / "b")


def test_repo_path_takes_a_relative_path_from_the_cwd_outside_bazel(tmp_path: Path, monkeypatch):
    (tmp_path / "a").mkdir()
    monkeypatch.delenv("BUILD_WORKSPACE_DIRECTORY", raising=False)
    monkeypatch.chdir(tmp_path)
    assert repo_path("a") == str(tmp_path / "a")


def test_repo_path_keeps_an_absolute_path(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BUILD_WORKSPACE_DIRECTORY", "/somewhere/else")
    assert repo_path(str(tmp_path)) == str(tmp_path)


def test_repo_path_names_the_resolved_location_when_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("BUILD_WORKSPACE_DIRECTORY", str(tmp_path))
    with pytest.raises(FileNotFoundError, match=f"'nope' resolves to {tmp_path}/nope"):
        repo_path("nope")


def test_project_dir_requires_dlux_project(monkeypatch):
    monkeypatch.delenv("DLUX_PROJECT", raising=False)
    with pytest.raises(RuntimeError, match="DLUX_PROJECT is not set"):
        project_dir()


def test_project_dir_resolves_a_relative_project(tmp_path: Path, monkeypatch):
    project = _project(tmp_path)
    monkeypatch.setenv("BUILD_WORKSPACE_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("DLUX_PROJECT", "proj")
    assert project_dir() == str(project)


def test_project_dir_names_the_missing_root(tmp_path: Path, monkeypatch):
    (tmp_path / "proj" / "config").mkdir(parents=True)
    monkeypatch.setenv("BUILD_WORKSPACE_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("DLUX_PROJECT", "proj")
    with pytest.raises(
        FileNotFoundError, match=f"DLUX_PROJECT='proj' resolves to {tmp_path}/proj, which has no catalog/"
    ):
        project_dir()


def test_run_key_prefers_the_split_version():
    assert run_key("o1_i2", "er", None) == "o1_i2"
    assert run_key(None, "er", 7) == "er_fold7"


def test_a_primary_config_finds_the_project_through_the_resolver(tmp_path: Path, monkeypatch):
    _project(tmp_path)
    config_dir = tmp_path / "entry"
    config_dir.mkdir()
    (config_dir / "base.yaml").write_text("defaults:\n  - cohort: ???\n  - _self_\nkey: 1\n")
    (config_dir / "entry.yaml").write_text(
        "defaults:\n  - base\n  - _self_\nhydra:\n  searchpath:\n    - file://${dlux.project:}/catalog\n"
    )
    monkeypatch.setenv("BUILD_WORKSPACE_DIRECTORY", str(tmp_path))
    monkeypatch.setenv("DLUX_PROJECT", "proj")
    register_resolvers()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="entry", overrides=["cohort=c1"])
    assert cfg.cohort.name == "c1"
    assert cfg.key == 1
