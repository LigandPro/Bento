from __future__ import annotations

from pathlib import Path

from bento import envcheck


def _patch_dependency_checks(monkeypatch):
    monkeypatch.setattr(
        envcheck,
        "_check_module",
        lambda module_name: envcheck.CheckResult(module_name, True, "module available"),
    )
    monkeypatch.setattr(
        envcheck,
        "_check_binary",
        lambda binary_name: envcheck.CheckResult(binary_name, True, "binary available"),
    )


def _index_results(results: list[envcheck.CheckResult]) -> dict[str, envcheck.CheckResult]:
    return {result.name: result for result in results}


def test_similarity_profile_requires_assign_features_class(tmp_path: Path, monkeypatch):
    _patch_dependency_checks(monkeypatch)
    glosa_dir = tmp_path / "glosa_v2.2"
    glosa_dir.mkdir()
    (glosa_dir / "glosa").write_text("", encoding="utf-8")

    results = envcheck.collect_checks("similarity", glosa_dir=glosa_dir)
    indexed = _index_results(results)

    assert indexed["glosa"].ok is True
    assert indexed["AssignChemicalFeatures.class"].ok is False
    assert "compile AssignChemicalFeatures.java" in indexed["AssignChemicalFeatures.class"].details


def test_similarity_profile_passes_when_assign_features_class_exists(tmp_path: Path, monkeypatch):
    _patch_dependency_checks(monkeypatch)
    glosa_dir = tmp_path / "glosa_v2.2"
    glosa_dir.mkdir()
    (glosa_dir / "glosa").write_text("", encoding="utf-8")
    (glosa_dir / "AssignChemicalFeatures.class").write_bytes(b"class")

    results = envcheck.collect_checks("similarity", glosa_dir=glosa_dir)
    indexed = _index_results(results)

    assert indexed["glosa"].ok is True
    assert indexed["AssignChemicalFeatures.class"].ok is True
