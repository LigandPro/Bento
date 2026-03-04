"""Command-line interface for Bento utilities."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from bento.envcheck import collect_checks, summarize


def _repo_root() -> Path:
    root_override = os.environ.get("BENTO_REPO_ROOT")
    if root_override:
        return Path(root_override).resolve()
    return Path(__file__).resolve().parents[2]


def _legacy_script_path(script_name: str) -> Path:
    return _repo_root() / "scripts" / script_name


def _run_legacy_script(script_name: str, script_args: Sequence[str]) -> int:
    script_path = _legacy_script_path(script_name)
    if not script_path.exists():
        print(f"ERROR: script not found: {script_path}", file=sys.stderr)
        return 2

    command = [sys.executable, str(script_path), *script_args]
    print(f"Running: {' '.join(command)}")
    completed = subprocess.run(command, check=False)
    return completed.returncode


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bento",
        description="Bento benchmark CLI wrapper around analysis pipeline scripts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    annotate = subparsers.add_parser(
        "annotate-ligands",
        help="Run ligand annotation pipeline (scripts/01_generate_ligands_annotation.py).",
    )
    annotate.add_argument("--dataset-file", required=True, help="Input dataset TSV file.")
    annotate.add_argument("--output-dir", required=True, help="Output directory.")

    similarity = subparsers.add_parser(
        "compute-pocket-similarity",
        help="Run pocket similarity pipeline (scripts/02_compute_pockets_similarity.py).",
    )
    similarity.add_argument("--output-file", required=True, help="Output CSV file.")
    similarity.add_argument("--data-csv", required=True, help="Input CSV table.")
    similarity.add_argument("--protein-path", required=True, help="Protein path column.")
    similarity.add_argument("--ligand-path", required=True, help="Ligand path column.")
    similarity.add_argument("--glosa-dir", required=True, help="GLoSA directory path.")
    similarity.add_argument(
        "--bs-dir",
        default="bs",
        help="Directory where generated binding sites are stored (default: bs).",
    )
    similarity.add_argument("--bs-column", help="Existing binding-site column name.")

    map_cmd = subparsers.add_parser(
        "map-annotations",
        help="Run annotation mapping pipeline (scripts/03_map_annotations.py).",
    )
    map_cmd.add_argument(
        "--tests-file",
        default="datasets/tests.tsv",
        help="Input tests TSV file (default: datasets/tests.tsv).",
    )
    map_cmd.add_argument(
        "--annotations-dir",
        default="annotations",
        help="Annotations directory (default: annotations).",
    )
    map_cmd.add_argument(
        "--output-tests-file",
        default="datasets/tests_annotated.tsv",
        help="Output path for annotated tests table.",
    )
    map_cmd.add_argument(
        "--output-tests-exploded-file",
        default="datasets/tests_exploded_annotated.tsv",
        help="Output path for exploded annotated tests table.",
    )
    map_cmd.add_argument(
        "--no-save",
        action="store_true",
        help="Run transformations without writing output files.",
    )

    check_env = subparsers.add_parser("check-env", help="Validate required environment components.")
    check_env.add_argument(
        "--profile",
        choices=("annotation", "similarity", "full"),
        default="full",
        help="Check profile to validate.",
    )
    check_env.add_argument(
        "--glosa-dir",
        help="Optional GLoSA directory to validate glosa binary path.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "annotate-ligands":
        return _run_legacy_script(
            "01_generate_ligands_annotation.py",
            [
                "--dataset_file",
                args.dataset_file,
                "--output_dir",
                args.output_dir,
            ],
        )

    if args.command == "compute-pocket-similarity":
        script_args = [
            "-output-file",
            args.output_file,
            "-data-csv",
            args.data_csv,
            "-protein-path",
            args.protein_path,
            "-ligand-path",
            args.ligand_path,
            "-bs-dir",
            args.bs_dir,
            "-glosa-dir",
            args.glosa_dir,
        ]
        if args.bs_column:
            script_args.extend(["-bs-column", args.bs_column])
        return _run_legacy_script("02_compute_pockets_similarity.py", script_args)

    if args.command == "map-annotations":
        script_args = [
            "--tests-file",
            args.tests_file,
            "--annotations-dir",
            args.annotations_dir,
            "--output-tests-file",
            args.output_tests_file,
            "--output-tests-exploded-file",
            args.output_tests_exploded_file,
        ]
        if args.no_save:
            script_args.append("--no-save")
        return _run_legacy_script("03_map_annotations.py", script_args)

    if args.command == "check-env":
        glosa_dir = Path(args.glosa_dir).resolve() if args.glosa_dir else None
        results = collect_checks(args.profile, glosa_dir=glosa_dir)
        all_ok, lines = summarize(results)
        print(f"Environment check profile: {args.profile}")
        for line in lines:
            print(line)
        return 0 if all_ok else 1

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
