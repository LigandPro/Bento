"""Environment validation helpers for Bento pipelines."""

from __future__ import annotations

import importlib.util
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    """Single environment check result."""

    name: str
    ok: bool
    details: str


def _check_module(module_name: str) -> CheckResult:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return CheckResult(module_name, False, "module not found")
    return CheckResult(module_name, True, "module available")


def _check_binary(binary_name: str) -> CheckResult:
    found = shutil.which(binary_name)
    if found is None:
        return CheckResult(binary_name, False, "binary not found in PATH")
    return CheckResult(binary_name, True, f"found at {found}")


def collect_checks(profile: str, glosa_dir: Path | None = None) -> list[CheckResult]:
    """Return checks for a named profile."""
    profile = profile.lower()
    module_checks: list[CheckResult] = []
    binary_checks: list[CheckResult] = []

    if profile in {"annotation", "full"}:
        for module_name in ("pandas", "numpy", "rdkit", "pandarallel", "psutil"):
            module_checks.append(_check_module(module_name))

    if profile in {"similarity", "full"}:
        for module_name in ("pandas", "tqdm", "pymol"):
            module_checks.append(_check_module(module_name))
        for binary_name in ("java", "javac", "g++"):
            binary_checks.append(_check_binary(binary_name))
        if glosa_dir is not None:
            glosa_binary = glosa_dir / "glosa"
            if glosa_binary.exists():
                binary_checks.append(CheckResult("glosa", True, f"found at {glosa_binary}"))
            else:
                binary_checks.append(
                    CheckResult("glosa", False, f"not found at expected path {glosa_binary}")
                )
            assign_features_class = glosa_dir / "AssignChemicalFeatures.class"
            if assign_features_class.exists():
                binary_checks.append(
                    CheckResult(
                        "AssignChemicalFeatures.class",
                        True,
                        f"found at {assign_features_class}",
                    )
                )
            else:
                binary_checks.append(
                    CheckResult(
                        "AssignChemicalFeatures.class",
                        False,
                        "not found; compile AssignChemicalFeatures.java with javac in glosa dir",
                    )
                )

    return [*module_checks, *binary_checks]


def summarize(results: Iterable[CheckResult]) -> tuple[bool, list[str]]:
    """Return pass/fail and human-readable lines."""
    lines: list[str] = []
    all_ok = True
    for result in results:
        status = "OK" if result.ok else "MISSING"
        lines.append(f"[{status}] {result.name}: {result.details}")
        if not result.ok:
            all_ok = False
    return all_ok, lines
