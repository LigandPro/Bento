"""Compatibility wrapper for environment checks."""

from __future__ import annotations

import sys

from bento.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["check-env", *sys.argv[1:]]))
