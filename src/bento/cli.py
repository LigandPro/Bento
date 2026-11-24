"""Command-line interface for Bento utilities."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from bento.envcheck import collect_checks, summarize
from bento.glosa_parallel import (
    merge_glosa_shards,
    prepare_dataset_pockets,
    run_glosa_shards,
)
from bento.pockets import build_glosa_binary


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
        choices=("annotation", "similarity", "pocket-similarity", "full"),
        default="full",
        help="Check profile to validate.",
    )
    check_env.add_argument(
        "--glosa-dir",
        help="Optional GLoSA directory to validate glosa binary path.",
    )

    pocket_sim = subparsers.add_parser(
        "pocket-similarity",
        help="Parallel, sharded, Parquet-native GLoSA pocket similarity workflow.",
    )
    pocket_sub = pocket_sim.add_subparsers(dest="pocket_command", required=True)

    build_cmd = pocket_sub.add_parser(
        "build-glosa",
        help="Compile GLoSA glosa.cpp with -O3 -ffp-contract=off.",
    )
    build_cmd.add_argument("--glosa-dir", required=True, help="Directory containing glosa.cpp.")
    build_cmd.add_argument("--output-bin", help="Optional output executable path.")
    build_cmd.add_argument("--cxx", default="g++", help="C++ compiler executable (default: g++).")
    build_cmd.add_argument(
        "--no-optimize",
        action="store_true",
        help="Disable -O3 -ffp-contract=off compiler flags.",
    )

    def _add_dataset_args(cmd: argparse.ArgumentParser) -> None:
        cmd.add_argument("--dataset-tsv-1", required=True, help="First dataset TSV (e.g. trains).")
        cmd.add_argument("--dataset-tsv-2", required=True, help="Second dataset TSV (e.g. tests).")
        cmd.add_argument(
            "--bs-dir", required=True, help="Directory for binding-site PDB/-cf files."
        )
        cmd.add_argument(
            "--databases-dir",
            default=os.environ.get("BENTO_DATABASES_DIR"),
            help="Optional root directory prepended to relative protein/ligand paths.",
        )

    prep_cmd = pocket_sub.add_parser(
        "prepare",
        help="Extract per-uid binding sites and GLoSA chemical features in parallel.",
    )
    _add_dataset_args(prep_cmd)
    prep_cmd.add_argument(
        "--glosa-dir", required=True, help="GLoSA directory with AssignChemicalFeatures."
    )
    prep_cmd.add_argument("--workers", type=int, default=8, help="Parallel worker processes.")
    prep_cmd.add_argument(
        "--threshold", type=float, default=4.5, help="Contact cutoff in angstroms."
    )
    prep_cmd.add_argument(
        "--fallback-threshold",
        type=float,
        help="Optional larger cutoff in angstroms used only when a pocket is empty at --threshold.",
    )
    prep_cmd.add_argument("--java-bin", default="java", help="Java executable (default: java).")
    prep_cmd.add_argument(
        "--overwrite", action="store_true", help="Regenerate existing pocket files."
    )

    score_cmd = pocket_sub.add_parser(
        "score",
        help="Score a resumable shard range [start-shard, end-shard) in parallel.",
    )
    _add_dataset_args(score_cmd)
    score_cmd.add_argument("--glosa-dir", required=True, help="GLoSA directory.")
    score_cmd.add_argument("--glosa-bin", help="Optional override path to compiled glosa binary.")
    score_cmd.add_argument(
        "--shards-dir", required=True, help="Output directory for shard TSV/.ok files."
    )
    score_cmd.add_argument(
        "--slots-dir",
        help="Per-worker scratch cwd directory (e.g. /dev/shm/bento-glosa-slots).",
    )
    score_cmd.add_argument("--workers", type=int, default=8, help="Parallel worker processes.")
    score_cmd.add_argument(
        "--shard-size", type=int, default=10, help="Unique dataset-1 complexes per shard."
    )
    score_cmd.add_argument(
        "--start-shard", type=int, default=0, help="First shard index (inclusive)."
    )
    score_cmd.add_argument("--end-shard", type=int, help="Last shard index (exclusive).")
    score_cmd.add_argument(
        "--timeout", type=int, default=500, help="Per-pair GLoSA timeout in seconds."
    )

    merge_cmd = pocket_sub.add_parser(
        "merge",
        help="Merge completed shards into Parquet (default) or TSV plus optional max summary.",
    )
    _add_dataset_args(merge_cmd)
    merge_cmd.add_argument(
        "--shards-dir", required=True, help="Directory with completed shard files."
    )
    merge_cmd.add_argument("--output-dir", required=True, help="Directory for merged output file.")
    merge_cmd.add_argument(
        "--shard-size", type=int, default=10, help="Shard size used during scoring."
    )
    merge_cmd.add_argument("--tsv", action="store_true", help="Write TSV instead of Parquet.")
    merge_cmd.add_argument(
        "--max-output",
        help="Optional output TSV path for per-dataset-2 maximum GLoSA scores.",
    )

    run_cmd = pocket_sub.add_parser(
        "run",
        help="End-to-end single-host parallel workflow (prepare -> score -> merge).",
    )
    _add_dataset_args(run_cmd)
    run_cmd.add_argument("--glosa-dir", required=True, help="GLoSA directory.")
    run_cmd.add_argument("--glosa-bin", help="Optional override path to compiled glosa binary.")
    run_cmd.add_argument("--output-dir", required=True, help="Output directory for Parquet/TSV.")
    run_cmd.add_argument(
        "--shards-dir", help="Shard checkpoint directory (default: <output-dir>/shards)."
    )
    run_cmd.add_argument(
        "--slots-dir", help="Per-worker scratch cwd directory (e.g. /dev/shm/...)."
    )
    run_cmd.add_argument("--workers", type=int, default=8, help="Parallel worker processes.")
    run_cmd.add_argument(
        "--prep-workers",
        type=int,
        help="Worker count for pocket extraction (default: min(workers, 16)).",
    )
    run_cmd.add_argument(
        "--shard-size", type=int, default=10, help="Unique dataset-1 complexes per shard."
    )
    run_cmd.add_argument(
        "--threshold", type=float, default=4.5, help="Contact cutoff in angstroms."
    )
    run_cmd.add_argument(
        "--fallback-threshold", type=float, help="Optional fallback cutoff for empty pockets."
    )
    run_cmd.add_argument("--java-bin", default="java", help="Java executable (default: java).")
    run_cmd.add_argument(
        "--timeout", type=int, default=500, help="Per-pair GLoSA timeout in seconds."
    )
    run_cmd.add_argument("--tsv", action="store_true", help="Write TSV instead of Parquet.")
    run_cmd.add_argument(
        "--max-output", help="Optional output TSV path for max scores per dataset-2 entry."
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

    if args.command == "pocket-similarity":
        return _run_pocket_similarity(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _run_pocket_similarity(args: argparse.Namespace) -> int:
    if args.pocket_command == "build-glosa":
        built = build_glosa_binary(
            args.glosa_dir,
            output_path=args.output_bin,
            cxx=args.cxx,
            optimize=not args.no_optimize,
        )
        print(f"Built GLoSA binary: {built}")
        return 0

    if args.pocket_command == "prepare":
        targets = [("dataset1", args.dataset_tsv_1)]
        if os.path.abspath(args.dataset_tsv_1) != os.path.abspath(args.dataset_tsv_2):
            targets.append(("dataset2", args.dataset_tsv_2))
        for label, tsv in targets:
            summary = prepare_dataset_pockets(
                tsv,
                args.bs_dir,
                args.glosa_dir,
                databases_dir=args.databases_dir,
                workers=args.workers,
                threshold=args.threshold,
                fallback_threshold=args.fallback_threshold,
                java_bin=args.java_bin,
                overwrite=args.overwrite,
            )
            print(
                f"[{label}] rows={summary.total_rows} unique={summary.unique_complexes} "
                f"reused={summary.reused_existing} extracted={summary.extracted_fresh} "
                f"fallback={summary.fallback_repaired} empty={summary.empty_pockets} "
                f"errors={len(summary.errors)}"
            )
            if summary.errors:
                for uid, err in summary.errors[:10]:
                    print(f"  ERROR {uid}: {err}", file=sys.stderr)
                return 1
        return 0

    if args.pocket_command == "score":
        score_summary = run_glosa_shards(
            args.dataset_tsv_1,
            args.dataset_tsv_2,
            args.bs_dir,
            args.glosa_dir,
            args.shards_dir,
            databases_dir=args.databases_dir,
            glosa_bin=args.glosa_bin,
            slots_dir=args.slots_dir,
            workers=args.workers,
            shard_size=args.shard_size,
            start_shard=args.start_shard,
            end_shard=args.end_shard,
            timeout=args.timeout,
        )
        print(
            f"Scored shards: executed={score_summary.executed_shards} "
            f"skipped={score_summary.skipped_shards} total_shards={score_summary.total_shards} "
            f"pairs={score_summary.total_pairs} finite={score_summary.finite_scores} "
            f"blank={score_summary.blank_scores} elapsed={score_summary.elapsed_seconds:.1f}s"
        )
        return 0

    if args.pocket_command == "merge":
        merge_summary = merge_glosa_shards(
            args.dataset_tsv_1,
            args.dataset_tsv_2,
            args.bs_dir,
            args.shards_dir,
            args.output_dir,
            databases_dir=args.databases_dir,
            shard_size=args.shard_size,
            write_tsv=args.tsv,
            max_output=args.max_output,
        )
        print(
            f"Merged output: {merge_summary.output_path} "
            f"rows={merge_summary.total_rows} finite={merge_summary.finite_scores} "
            f"null={merge_summary.null_scores} elapsed={merge_summary.elapsed_seconds:.1f}s"
        )
        if merge_summary.max_output_path is not None:
            print(f"Max summary: {merge_summary.max_output_path}")
        return 0

    if args.pocket_command == "run":
        prep_workers = args.prep_workers or min(args.workers, 16)
        shards_dir = args.shards_dir or str(Path(args.output_dir) / "shards")
        targets = [("dataset1", args.dataset_tsv_1)]
        if os.path.abspath(args.dataset_tsv_1) != os.path.abspath(args.dataset_tsv_2):
            targets.append(("dataset2", args.dataset_tsv_2))
        for label, tsv in targets:
            summary = prepare_dataset_pockets(
                tsv,
                args.bs_dir,
                args.glosa_dir,
                databases_dir=args.databases_dir,
                workers=prep_workers,
                threshold=args.threshold,
                fallback_threshold=args.fallback_threshold,
                java_bin=args.java_bin,
            )
            print(
                f"[{label}] rows={summary.total_rows} unique={summary.unique_complexes} "
                f"reused={summary.reused_existing} extracted={summary.extracted_fresh} "
                f"fallback={summary.fallback_repaired} empty={summary.empty_pockets} "
                f"errors={len(summary.errors)}"
            )
            if summary.errors:
                for uid, err in summary.errors[:10]:
                    print(f"  ERROR {uid}: {err}", file=sys.stderr)
                return 1

        score_summary = run_glosa_shards(
            args.dataset_tsv_1,
            args.dataset_tsv_2,
            args.bs_dir,
            args.glosa_dir,
            shards_dir,
            databases_dir=args.databases_dir,
            glosa_bin=args.glosa_bin,
            slots_dir=args.slots_dir,
            workers=args.workers,
            shard_size=args.shard_size,
            timeout=args.timeout,
        )
        print(
            f"Scored shards: executed={score_summary.executed_shards} "
            f"skipped={score_summary.skipped_shards} total_shards={score_summary.total_shards} "
            f"pairs={score_summary.total_pairs} finite={score_summary.finite_scores} "
            f"blank={score_summary.blank_scores} elapsed={score_summary.elapsed_seconds:.1f}s"
        )

        merge_summary = merge_glosa_shards(
            args.dataset_tsv_1,
            args.dataset_tsv_2,
            args.bs_dir,
            shards_dir,
            args.output_dir,
            databases_dir=args.databases_dir,
            shard_size=args.shard_size,
            write_tsv=args.tsv,
            max_output=args.max_output,
        )
        print(
            f"Merged output: {merge_summary.output_path} "
            f"rows={merge_summary.total_rows} finite={merge_summary.finite_scores} "
            f"null={merge_summary.null_scores} elapsed={merge_summary.elapsed_seconds:.1f}s"
        )
        if merge_summary.max_output_path is not None:
            print(f"Max summary: {merge_summary.max_output_path}")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
