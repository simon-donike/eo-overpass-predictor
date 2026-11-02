"""Package integration checks; scientific regression fixtures are maintained separately."""

import ast
import importlib
import json
import subprocess
import sys
import tomllib
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import pytest

import eo_overpass_predictor as eo
from eo_overpass_predictor import _common, _paths, esa_plans

ROOT = Path(__file__).resolve().parents[1]


def test_core_works_without_notebook_dependencies(tmp_path):
    """Importing every public symbol and exporting HTML must not load UI libraries."""
    script = """
import importlib.abc
import sys

class NoNotebookImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'folium', 'matplotlib', 'ipyleaflet', 'ipywidgets', 'IPython', 'jupyterlab'
        }:
            raise ImportError(f'Unexpected notebook dependency: {fullname}')

sys.meta_path.insert(0, NoNotebookImports())
import eo_overpass_predictor as eo
for name in eo.__all__:
    getattr(eo, name)
roi = eo.load_roi((144.96, -37.82), name='Melbourne')
eo.export_live_tracker_html({}, roi, 'tracker.html', timezone_name='UTC')
"""
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)
    html = (tmp_path / "tracker.html").read_text(encoding="utf-8")
    assert "/*__PAYLOAD__*/null" not in html
    assert "/*__LAND__*/null" not in html
    assert "__TITLE__" not in html
    assert '"roi_name": "Melbourne"' in html
    land = files("eo_overpass_predictor").joinpath("resources", "world_land_50m.json")
    assert land.read_text(encoding="utf-8") in html


def test_legacy_imports_reexport_the_same_objects(monkeypatch):
    """Existing notebooks can keep adding Tools to their import path for one release."""
    monkeypatch.syspath_prepend(str(ROOT / "Tools"))
    legacy = importlib.import_module("overpass_helper_utility")
    for name in eo.__all__:
        assert getattr(legacy, name) is getattr(eo, name)
    for name in set(legacy.__all__) - set(eo.__all__):
        assert getattr(legacy, name) is getattr(_common, name)


@pytest.mark.parametrize(
    ("platform", "relative_root"),
    [("linux", "xdg"), ("darwin", "home/Library/Caches"), ("win32", "local")],
)
def test_default_caches_are_independent_of_working_directory(
    monkeypatch, tmp_path, platform, relative_root
):
    """Default paths follow each platform's cache convention without creating files."""
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.chdir(tmp_path)
    expected = tmp_path / relative_root / "eo-overpass-predictor" / "tle"
    assert _paths._cache_dir("tle") == expected
    assert not expected.exists()


def test_relative_cache_environment_falls_back_to_home(monkeypatch, tmp_path):
    """A relative XDG_CACHE_HOME must not introduce working-directory-dependent I/O."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", "relative-cache")
    assert _paths._cache_dir("tle") == tmp_path / ".cache/eo-overpass-predictor/tle"


def test_fetchers_and_plan_loader_use_the_same_user_cache(monkeypatch, tmp_path):
    """Omitted paths use user caches; explicit paths retain the notebook behavior."""
    monkeypatch.setattr(_paths, "_cache_dir", lambda kind: tmp_path / "cache" / kind)
    monkeypatch.setattr("eo_overpass_predictor.tle._cache_dir", _paths._cache_dir)
    monkeypatch.setattr(esa_plans, "_cache_dir", _paths._cache_dir)
    monkeypatch.setattr(esa_plans.requests, "get", lambda *args, **kwargs: SimpleNamespace(text=""))

    assert eo.load_s2_acquisition_plans().empty
    assert not (tmp_path / "cache").exists()
    assert eo.fetch_tles(satellites=[], verbose=False) == {}
    assert (tmp_path / "cache/tle").is_dir()
    assert eo.fetch_s2_acquisition_plans(start_utc="2026-10-01", verbose=False) == []
    assert (tmp_path / "cache/kml").is_dir()
    assert eo.fetch_tles(satellites=[], tle_dir=tmp_path / "explicit", verbose=False) == {}
    assert (tmp_path / "explicit").is_dir()


def test_conda_dependency_list_matches_package_dependencies():
    """Keep the supported conda route aligned with core and notebook requirements."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    requirements = project["dependencies"] + project["optional-dependencies"]["notebooks"]
    required = {item.split(";", 1)[0].strip().lower() for item in requirements}
    conda_lines = (ROOT / "Tools/env.base.yml").read_text().split("dependencies:", 1)[1]
    conda_requirements = {
        line.strip()[2:].split("#", 1)[0].strip().lower()
        for line in conda_lines.splitlines()
        if line.strip().startswith("- ")
    }
    assert required <= conda_requirements
    assert conda_requirements - required == {"python=3.11", "pip"}


def test_notebook_imports_resolve_to_the_installed_package():
    """Check notebook source syntax and its public API imports without live downloads."""
    for path in ROOT.glob("p0*.ipynb"):
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                tree = ast.parse(source, filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module == "eo_overpass_predictor":
                        for name in node.names:
                            assert hasattr(eo, name.name)
