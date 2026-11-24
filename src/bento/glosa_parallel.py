"""Parallel sharded GLoSA pocket similarity runner and Parquet/TSV merger."""

from __future__ import annotations

import csv
import math
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bento.pockets import (
    chemical_features_path,
    has_atoms,
    prepare_binding_site,
    prepare_chemical_features,
)


@dataclass(frozen=True)
class DatasetEntry:
    """Single row from an input Bento dataset TSV."""

    uid: str
    path_protein: str
    path_ligand: str
    pocket_path: Path


@dataclass(frozen=True)
class PreparationSummary:
    """Summary of parallel binding-site and chemical-feature preparation."""

    total_rows: int
    unique_complexes: int
    reused_existing: int
    extracted_fresh: int
    fallback_repaired: int
    empty_pockets: int
    errors: list[tuple[str, str]]


@dataclass(frozen=True)
class ShardRunSummary:
    """Summary of a parallel sharded GLoSA scoring run."""

    total_shards: int
    executed_shards: int
    skipped_shards: int
    total_pairs: int
    finite_scores: int
    blank_scores: int
    elapsed_seconds: float


@dataclass(frozen=True)
class MergeSummary:
    """Summary of merging scored shards into Parquet or TSV."""

    output_path: Path
    max_output_path: Path | None
    total_rows: int
    finite_scores: int
    null_scores: int
    elapsed_seconds: float


# Worker process globals populated via ProcessPoolExecutor initializer.
_WORKER_SLOT_ROOT: Path | None = None
_WORKER_GLOSA_BIN: Path | None = None
_WORKER_TEST_ARGS: list[tuple[str, str]] = []
_WORKER_TIMEOUT: int = 500


def _validate_uid(uid: str, dataset_tsv: Path) -> str:
    if not isinstance(uid, str) or not uid.strip() or "/" in uid or "\\" in uid:
        raise ValueError(f"Invalid uid in {dataset_tsv}: {uid!r}")
    return uid


def _resolve_input_path(raw_value: str, databases_dir: Path | None) -> str:
    value = raw_value.strip()
    if databases_dir is not None and value and not os.path.isabs(value):
        return str(databases_dir / value)
    return value


def load_dataset_entries(
    dataset_tsv: str | Path,
    bs_dir: str | Path,
    *,
    databases_dir: str | Path | None = None,
) -> list[DatasetEntry]:
    """Load dataset rows and assign collision-free `<dataset_filename>_<uid>.pdb` pocket paths."""
    tsv_path = Path(dataset_tsv)
    bs_root = Path(bs_dir).resolve()
    db_root = Path(databases_dir).resolve() if databases_dir else None
    dataset_name = tsv_path.name

    entries: list[DatasetEntry] = []
    seen_uids: set[str] = set()
    with tsv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "uid" not in reader.fieldnames:
            raise ValueError(f"{tsv_path} must have a 'uid' column")
        for row in reader:
            uid = _validate_uid(row["uid"], tsv_path)
            if uid in seen_uids:
                raise ValueError(f"Duplicate uid in {tsv_path}: {uid}")
            seen_uids.add(uid)
            protein = _resolve_input_path(row.get("path_protein", ""), db_root)
            ligand = _resolve_input_path(row.get("path_ligand", ""), db_root)
            pocket = bs_root / f"{dataset_name}_{uid}.pdb"
            entries.append(
                DatasetEntry(
                    uid=uid,
                    path_protein=protein,
                    path_ligand=ligand,
                    pocket_path=pocket,
                )
            )
    return entries


def _unique_complex_groups(
    entries: list[DatasetEntry],
) -> tuple[list[tuple[str, Path]], dict[tuple[str, str], str]]:
    """Group rows sharing identical `(path_protein, path_ligand)` while preserving order.

    Rows with different ligands on the same protein receive distinct keys, avoiding
    multi-ligand pocket collisions while eliminating redundant GLoSA calls for exact
    duplicate `(protein, ligand)` entries.
    """
    unique_items: list[tuple[str, Path]] = []
    key_to_canonical_uid: dict[tuple[str, str], str] = {}
    for entry in entries:
        key = (entry.path_protein, entry.path_ligand)
        if key not in key_to_canonical_uid:
            key_to_canonical_uid[key] = entry.uid
            unique_items.append((entry.uid, entry.pocket_path))
    return unique_items, key_to_canonical_uid


def _prepare_one_complex(task: tuple[Any, ...]) -> tuple[str, bool, bool, bool, str]:
    (
        canonical_uid,
        protein_path,
        ligand_path,
        target_pockets,
        glosa_dir,
        threshold,
        fallback_threshold,
        java_bin,
        overwrite,
    ) = task
    primary_pocket = Path(target_pockets[0])
    try:
        prep = prepare_binding_site(
            protein_path=protein_path,
            ligand_path=ligand_path,
            bs_path=primary_pocket,
            threshold=threshold,
            fallback_threshold=fallback_threshold,
            overwrite=overwrite,
        )
        if not has_atoms(primary_pocket):
            for duplicate in target_pockets[1:]:
                dup_path = Path(duplicate)
                if overwrite or not dup_path.is_file():
                    dup_path.write_bytes(primary_pocket.read_bytes())
            return canonical_uid, prep.reused_existing, prep.used_fallback, True, ""

        primary_cf = prepare_chemical_features(
            primary_pocket,
            glosa_dir,
            java_bin=java_bin,
            overwrite=overwrite,
        )
        if not has_atoms(primary_cf):
            return (
                canonical_uid,
                prep.reused_existing,
                prep.used_fallback,
                True,
                "empty chemical features",
            )

        for duplicate in target_pockets[1:]:
            dup_pdb = Path(duplicate)
            dup_cf = chemical_features_path(dup_pdb)
            if overwrite or not has_atoms(dup_pdb):
                dup_pdb.write_bytes(primary_pocket.read_bytes())
            if overwrite or not has_atoms(dup_cf):
                dup_cf.write_bytes(primary_cf.read_bytes())

        return canonical_uid, prep.reused_existing, prep.used_fallback, False, ""
    except Exception as exc:  # noqa: BLE001
        return canonical_uid, False, False, False, str(exc)


def prepare_dataset_pockets(
    dataset_tsv: str | Path,
    bs_dir: str | Path,
    glosa_dir: str | Path,
    *,
    databases_dir: str | Path | None = None,
    workers: int = 8,
    threshold: float = 4.5,
    fallback_threshold: float | None = None,
    java_bin: str = "java",
    overwrite: bool = False,
) -> PreparationSummary:
    """Extract pockets and GLoSA chemical features in parallel for one dataset."""
    entries = load_dataset_entries(dataset_tsv, bs_dir, databases_dir=databases_dir)
    Path(bs_dir).mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str], list[DatasetEntry]] = {}
    for entry in entries:
        grouped.setdefault((entry.path_protein, entry.path_ligand), []).append(entry)

    tasks = [
        (
            group[0].uid,
            prot,
            lig,
            [str(item.pocket_path) for item in group],
            str(Path(glosa_dir).resolve()),
            threshold,
            fallback_threshold,
            java_bin,
            overwrite,
        )
        for (prot, lig), group in grouped.items()
    ]

    reused = 0
    extracted = 0
    fallback = 0
    empty = 0
    errors: list[tuple[str, str]] = []

    if workers <= 1:
        results = [_prepare_one_complex(task) for task in tasks]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_prepare_one_complex, task) for task in tasks]
            for future in as_completed(futures):
                results.append(future.result())

    for uid, was_reused, used_fb, is_empty, err in results:
        if err:
            errors.append((uid, err))
        elif was_reused:
            reused += 1
        else:
            extracted += 1
        if used_fb:
            fallback += 1
        if is_empty:
            empty += 1

    return PreparationSummary(
        total_rows=len(entries),
        unique_complexes=len(grouped),
        reused_existing=reused,
        extracted_fresh=extracted,
        fallback_repaired=fallback,
        empty_pockets=empty,
        errors=errors,
    )


def parse_glosa_stdout(stdout: bytes) -> str:
    """Extract the GA-score from GLoSA stdout matching Bento's legacy parser."""
    if not stdout:
        return ""
    lines = stdout.splitlines()
    if len(lines) <= 4:
        return ""
    fields = lines[4].split()
    if len(fields) < 2:
        return ""
    token = fields[1]
    if token == b"-nan":
        return ""
    try:
        value = float(token)
    except ValueError:
        return ""
    return str(value) if math.isfinite(value) else ""


def _init_glosa_worker(
    slot_root_str: str,
    glosa_bin_str: str,
    test_args: list[tuple[str, str]],
    timeout: int,
) -> None:
    global _WORKER_SLOT_ROOT, _WORKER_GLOSA_BIN, _WORKER_TEST_ARGS, _WORKER_TIMEOUT
    _WORKER_SLOT_ROOT = Path(slot_root_str)
    _WORKER_GLOSA_BIN = Path(glosa_bin_str)
    _WORKER_TEST_ARGS = test_args
    _WORKER_TIMEOUT = timeout

    slot = _WORKER_SLOT_ROOT / str(os.getpid())
    slot.mkdir(parents=True, exist_ok=True)
    link = slot / "glosa"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(_WORKER_GLOSA_BIN)


def _format_glosa_path(target: Path, cwd: Path) -> str:
    try:
        rel = os.path.relpath(target, cwd)
    except ValueError:
        return str(target)
    return rel if len(rel) < len(str(target)) else str(target)


def _score_one_shard(task: tuple[Any, ...]) -> tuple[int, int, int, int, float, bool]:
    shard_id, items, shards_dir_str = task
    shards_dir = Path(shards_dir_str)
    out_file = shards_dir / f"{shard_id:04d}.tsv"
    ok_file = shards_dir / f"{shard_id:04d}.ok"
    n_tests = len(_WORKER_TEST_ARGS)
    expected_pairs = len(items) * n_tests

    if ok_file.is_file() and out_file.is_file():
        parts = ok_file.read_text(encoding="utf-8").strip().split()
        if len(parts) >= 4 and int(parts[0]) == len(items) and int(parts[1]) == expected_pairs:
            return shard_id, int(parts[1]), int(parts[2]), int(parts[3]), 0.0, True

    assert _WORKER_SLOT_ROOT is not None
    slot = _WORKER_SLOT_ROOT / str(os.getpid())
    slot.mkdir(parents=True, exist_ok=True)
    link = slot / "glosa"
    if not link.exists():
        assert _WORKER_GLOSA_BIN is not None
        link.symlink_to(_WORKER_GLOSA_BIN)

    tmp_file = shards_dir / f".{shard_id:04d}.tsv.{os.getpid()}.tmp"
    t0 = time.monotonic()
    finite = 0
    blank = 0
    output_lines: list[str] = []

    run_proc = subprocess.run
    PIPE = subprocess.PIPE
    DEVNULL = subprocess.DEVNULL
    timeout = _WORKER_TIMEOUT

    test_rel = [
        (_format_glosa_path(Path(p), slot), _format_glosa_path(Path(c), slot))
        for p, c in _WORKER_TEST_ARGS
    ]

    for canonical_uid, pocket_str in items:
        pocket = Path(pocket_str)
        cf = chemical_features_path(pocket)
        s1 = _format_glosa_path(pocket, slot)
        s1cf = _format_glosa_path(cf, slot)
        s1_ready = has_atoms(pocket) and has_atoms(cf)

        row = [canonical_uid]
        if not s1_ready:
            row.extend([""] * n_tests)
            blank += n_tests
            output_lines.append("\t".join(row) + "\n")
            continue

        for s2, s2cf in test_rel:
            try:
                completed = run_proc(
                    ("./glosa", "-s1", s1, "-s1cf", s1cf, "-s2", s2, "-s2cf", s2cf),
                    cwd=slot,
                    stdout=PIPE,
                    stderr=DEVNULL,
                    timeout=timeout,
                )
                val = parse_glosa_stdout(completed.stdout)
            except Exception:  # noqa: BLE001
                val = ""
            row.append(val)
            if val:
                finite += 1
            else:
                blank += 1
        output_lines.append("\t".join(row) + "\n")

    with tmp_file.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
        handle.writelines(output_lines)
    os.replace(tmp_file, out_file)

    elapsed = time.monotonic() - t0
    ok_file.write_text(
        f"{len(items)} {expected_pairs} {finite} {blank} {elapsed:.2f}\n",
        encoding="utf-8",
    )
    return shard_id, expected_pairs, finite, blank, elapsed, False


def total_shard_count(
    dataset_tsv_1: str | Path,
    bs_dir: str | Path,
    *,
    shard_size: int = 10,
    databases_dir: str | Path | None = None,
) -> int:
    """Return total number of shards for `dataset_tsv_1` at a given `shard_size`."""
    entries = load_dataset_entries(dataset_tsv_1, bs_dir, databases_dir=databases_dir)
    unique_items, _ = _unique_complex_groups(entries)
    return math.ceil(len(unique_items) / shard_size)


def run_glosa_shards(
    dataset_tsv_1: str | Path,
    dataset_tsv_2: str | Path,
    bs_dir: str | Path,
    glosa_dir: str | Path,
    shards_dir: str | Path,
    *,
    databases_dir: str | Path | None = None,
    glosa_bin: str | Path | None = None,
    slots_dir: str | Path | None = None,
    workers: int = 8,
    shard_size: int = 10,
    start_shard: int = 0,
    end_shard: int | None = None,
    timeout: int = 500,
) -> ShardRunSummary:
    """Score a range of resumable shards `[start_shard, end_shard)` in parallel."""
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if workers <= 0:
        raise ValueError("workers must be positive")

    entries1 = load_dataset_entries(dataset_tsv_1, bs_dir, databases_dir=databases_dir)
    entries2 = load_dataset_entries(dataset_tsv_2, bs_dir, databases_dir=databases_dir)
    unique_items, _ = _unique_complex_groups(entries1)

    resolved_glosa_bin = (
        Path(glosa_bin).resolve() if glosa_bin else (Path(glosa_dir).resolve() / "glosa")
    )
    if not resolved_glosa_bin.is_file():
        raise FileNotFoundError(f"GLoSA binary not found: {resolved_glosa_bin}")

    shards_root = Path(shards_dir).resolve()
    shards_root.mkdir(parents=True, exist_ok=True)
    slot_root = Path(slots_dir).resolve() if slots_dir else shards_root / "_slots"
    slot_root.mkdir(parents=True, exist_ok=True)

    test_args = [
        (str(entry.pocket_path), str(chemical_features_path(entry.pocket_path)))
        for entry in entries2
    ]

    all_shards = [
        (
            sid,
            [
                (uid, str(pocket))
                for uid, pocket in unique_items[sid * shard_size : (sid + 1) * shard_size]
            ],
            str(shards_root),
        )
        for sid in range(math.ceil(len(unique_items) / shard_size))
    ]
    resolved_end = len(all_shards) if end_shard is None else min(end_shard, len(all_shards))
    selected = [s for s in all_shards if start_shard <= s[0] < resolved_end]

    t0 = time.monotonic()
    executed = 0
    skipped = 0
    tot_pairs = 0
    tot_finite = 0
    tot_blank = 0

    if workers == 1:
        _init_glosa_worker(str(slot_root), str(resolved_glosa_bin), test_args, timeout)
        for task in selected:
            _, npairs, nfin, nblk, _, was_skipped = _score_one_shard(task)
            tot_pairs += npairs
            tot_finite += nfin
            tot_blank += nblk
            if was_skipped:
                skipped += 1
            else:
                executed += 1
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_glosa_worker,
            initargs=(str(slot_root), str(resolved_glosa_bin), test_args, timeout),
        ) as pool:
            futures = [pool.submit(_score_one_shard, task) for task in selected]
            for future in as_completed(futures):
                _, npairs, nfin, nblk, _, was_skipped = future.result()
                tot_pairs += npairs
                tot_finite += nfin
                tot_blank += nblk
                if was_skipped:
                    skipped += 1
                else:
                    executed += 1

    return ShardRunSummary(
        total_shards=len(all_shards),
        executed_shards=executed,
        skipped_shards=skipped,
        total_pairs=tot_pairs,
        finite_scores=tot_finite,
        blank_scores=tot_blank,
        elapsed_seconds=time.monotonic() - t0,
    )


def merge_glosa_shards(
    dataset_tsv_1: str | Path,
    dataset_tsv_2: str | Path,
    bs_dir: str | Path,
    shards_dir: str | Path,
    output_dir: str | Path,
    *,
    databases_dir: str | Path | None = None,
    shard_size: int = 10,
    write_tsv: bool = False,
    max_output: str | Path | None = None,
    batch_rows: int = 100,
) -> MergeSummary:
    """Merge all scored shards into Parquet (default) or TSV in exact dataset order."""
    t0 = time.monotonic()
    tsv1 = Path(dataset_tsv_1)
    tsv2 = Path(dataset_tsv_2)
    same_input = os.path.abspath(tsv1) == os.path.abspath(tsv2)

    entries1 = load_dataset_entries(tsv1, bs_dir, databases_dir=databases_dir)
    entries2 = load_dataset_entries(tsv2, bs_dir, databases_dir=databases_dir)
    unique_items, key_to_canonical = _unique_complex_groups(entries1)

    shards_root = Path(shards_dir).resolve()
    n_shards = math.ceil(len(unique_items) / shard_size)
    missing = [
        sid
        for sid in range(n_shards)
        if not (shards_root / f"{sid:04d}.ok").is_file()
        or not (shards_root / f"{sid:04d}.tsv").is_file()
    ]
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} of {n_shards} shards in {shards_root}: {missing[:10]}"
        )

    raw_by_canonical: dict[str, str] = {}
    for sid in range(n_shards):
        with (shards_root / f"{sid:04d}.tsv").open("r", encoding="utf-8") as handle:
            for line in handle:
                tab = line.find("\t")
                if tab <= 0:
                    raise ValueError(f"Malformed line in shard {sid:04d}")
                raw_by_canonical[line[:tab]] = line[tab + 1 :].rstrip("\n")

    expected_keys = {uid for uid, _ in unique_items}
    if set(raw_by_canonical.keys()) != expected_keys:
        raise RuntimeError(
            f"Shard key mismatch: loaded {len(raw_by_canonical)}, expected {len(expected_keys)}"
        )

    name1 = tsv1.name
    name2 = tsv2.name
    col1 = f"uid_{name1}"
    col2 = f"uid_{name2}"
    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    ext = "tsv" if write_tsv else "parquet"
    out_path = out_root / f"pocket_scores_{name1}_{name2}.{ext}"
    tmp_path = out_path.with_name(f".{out_path.name}.{os.getpid()}.tmp")

    test_uids = [entry.uid for entry in entries2]
    n_tests = len(test_uids)
    best_score = np.full(n_tests, -1.0, dtype=np.float64)
    best_uid1: list[str | None] = [None] * n_tests

    total_rows = 0
    finite_scores = 0
    null_scores = 0

    if write_tsv:
        with tmp_path.open("w", encoding="utf-8", buffering=4 * 1024 * 1024) as out:
            out.write(f"{col1}\t{col2}\tglosa_score\n")
            for i, entry1 in enumerate(entries1):
                canonical = key_to_canonical[(entry1.path_protein, entry1.path_ligand)]
                parts = raw_by_canonical[canonical].split("\t")
                if len(parts) != n_tests:
                    raise ValueError(f"Bad score count for {canonical}: {len(parts)} != {n_tests}")
                start_j = i + 1 if same_input else 0
                buf: list[str] = []
                for j in range(start_j, n_tests):
                    uid2 = test_uids[j]
                    if entry1.uid == uid2:
                        continue
                    sc = parts[j]
                    buf.append(f"{entry1.uid}\t{uid2}\t{sc}\n")
                    total_rows += 1
                    if sc:
                        finite_scores += 1
                        fv = float(sc)
                        if fv > best_score[j]:
                            best_score[j] = fv
                            best_uid1[j] = entry1.uid
                    else:
                        null_scores += 1
                out.writelines(buf)
        os.replace(tmp_path, out_path)
    else:
        import pyarrow as pa
        import pyarrow.parquet as pq

        schema = pa.schema(
            [
                (col1, pa.string()),
                (col2, pa.string()),
                ("glosa_score", pa.float64()),
            ]
        )

        with pq.ParquetWriter(tmp_path, schema, compression="snappy") as writer:
            u1_buf: list[str] = []
            u2_buf: list[str] = []
            sc_buf: list[float | None] = []
            rows_in_batch = 0

            def flush_batch() -> None:
                nonlocal u1_buf, u2_buf, sc_buf, rows_in_batch
                if not u1_buf:
                    return
                table = pa.Table.from_arrays(
                    [
                        pa.array(u1_buf, type=pa.string()),
                        pa.array(u2_buf, type=pa.string()),
                        pa.array(sc_buf, type=pa.float64()),
                    ],
                    schema=schema,
                )
                writer.write_table(table)
                u1_buf = []
                u2_buf = []
                sc_buf = []
                rows_in_batch = 0

            for i, entry1 in enumerate(entries1):
                canonical = key_to_canonical[(entry1.path_protein, entry1.path_ligand)]
                parts = raw_by_canonical[canonical].split("\t")
                if len(parts) != n_tests:
                    raise ValueError(f"Bad score count for {canonical}: {len(parts)} != {n_tests}")
                start_j = i + 1 if same_input else 0
                for j in range(start_j, n_tests):
                    uid2 = test_uids[j]
                    if entry1.uid == uid2:
                        continue
                    sc = parts[j]
                    u1_buf.append(entry1.uid)
                    u2_buf.append(uid2)
                    total_rows += 1
                    if sc:
                        fv = float(sc)
                        sc_buf.append(fv)
                        finite_scores += 1
                        if fv > best_score[j]:
                            best_score[j] = fv
                            best_uid1[j] = entry1.uid
                    else:
                        sc_buf.append(None)
                        null_scores += 1

                rows_in_batch += 1
                if rows_in_batch >= batch_rows:
                    flush_batch()
            flush_batch()

        os.replace(tmp_path, out_path)

    resolved_max: Path | None = None
    if max_output is not None:
        resolved_max = Path(max_output).resolve()
        resolved_max.parent.mkdir(parents=True, exist_ok=True)
        tmp_max = resolved_max.with_name(f".{resolved_max.name}.{os.getpid()}.tmp")
        with tmp_max.open("w", encoding="utf-8", newline="") as handle:
            writer_csv = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer_csv.writerow([col1, col2, "glosa_score"])
            for j, uid2 in enumerate(test_uids):
                if best_uid1[j] is None:
                    writer_csv.writerow(["", uid2, ""])
                else:
                    writer_csv.writerow([best_uid1[j], uid2, str(best_score[j])])
        os.replace(tmp_max, resolved_max)

    return MergeSummary(
        output_path=out_path,
        max_output_path=resolved_max,
        total_rows=total_rows,
        finite_scores=finite_scores,
        null_scores=null_scores,
        elapsed_seconds=time.monotonic() - t0,
    )
