from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_config_module():
    config_path = Path(__file__).resolve().parents[1] / "scripts" / "config.py"
    spec = importlib.util.spec_from_file_location("bento_legacy_config", config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_config_defaults(monkeypatch):
    monkeypatch.delenv("BENTO_WORKDIR", raising=False)
    monkeypatch.delenv("BENTO_DATABASES_DIR", raising=False)
    monkeypatch.delenv("BENTO_GLOSA_DIR", raising=False)

    module = _load_config_module()
    expected_root = str((Path(__file__).resolve().parents[1]).resolve())

    assert module.wrk_dir == expected_root
    assert module.databases_dir == expected_root
    assert module.glosa_path.endswith("external/glosa")


def test_config_env_override(monkeypatch, tmp_path):
    workdir = tmp_path / "workdir"
    dbdir = tmp_path / "dbdir"
    glosa = tmp_path / "glosa"
    workdir.mkdir()
    dbdir.mkdir()
    glosa.mkdir()

    monkeypatch.setenv("BENTO_WORKDIR", str(workdir))
    monkeypatch.setenv("BENTO_DATABASES_DIR", str(dbdir))
    monkeypatch.setenv("BENTO_GLOSA_DIR", str(glosa))

    module = _load_config_module()

    assert module.wrk_dir == str(workdir.resolve())
    assert module.databases_dir == str(dbdir.resolve())
    assert module.glosa_path == str(glosa.resolve())
