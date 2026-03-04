"""Runtime path configuration for Bento legacy scripts."""

from __future__ import annotations

import os
from pathlib import Path


def _resolve_repo_root() -> Path:
    default_root = Path(__file__).resolve().parents[1]
    return Path(os.environ.get("BENTO_WORKDIR", default_root)).resolve()


_repo_root = _resolve_repo_root()

# Historical variable names are kept for compatibility with existing scripts.
wrk_dir = str(_repo_root)
databases_dir = str(Path(os.environ.get("BENTO_DATABASES_DIR", _repo_root)).resolve())
glosa_path = str(
    Path(os.environ.get("BENTO_GLOSA_DIR", _repo_root / "external" / "glosa")).resolve()
)
