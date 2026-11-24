from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from bento import cli, envcheck
from bento.glosa_parallel import (
    load_dataset_entries,
    merge_glosa_shards,
    parse_glosa_stdout,
    prepare_dataset_pockets,
    run_glosa_shards,
)
from bento.pockets import (
    chemical_features_path,
    clean_pdb_file,
    extract_pocket,
    prepare_binding_site,
    prepare_chemical_features,
)

SAMPLE_PROTEIN_PDB = """\
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N  
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C  
ATOM      3  N   GLY A   2      10.000   0.000   0.000  1.00 20.00           N  
ATOM      4  CA  GLY A   2      11.458   0.000   0.000  1.00 20.00           C  
ANISOU    1  N   ALA A   1      100    100    100      0      0      0       N  
TER
END
"""

SAMPLE_LIGAND_NEAR_A1_PDB = """\
HETATM    1  C1  LIG B   1       1.500   1.000   0.000  1.00 10.00           C  
END
"""

SAMPLE_LIGAND_NEAR_A2_PDB = """\
HETATM    1  C1  LIG B   2      10.500   1.000   0.000  1.00 10.00           C  
END
"""

SAMPLE_LIGAND_FAR_PDB = """\
HETATM    1  C1  LIG B   3       0.000   5.000   0.000  1.00 10.00           C  
END
"""

SAMPLE_LIGAND_V2000_SDF = """\
lig
  test

  1  0  0  0  0  0  0  0  0  0999 V2000
    1.5000    1.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
M  END
$$$$
"""


def _make_executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def test_clean_pdb_file_removes_anisou_and_normalizes_ter(tmp_path: Path):
    pdb = tmp_path / "pocket.pdb"
    pdb.write_text(SAMPLE_PROTEIN_PDB, encoding="utf-8")
    clean_pdb_file(pdb)
    lines = pdb.read_text(encoding="utf-8").splitlines()
    assert not any(line.startswith("ANISOU") for line in lines)
    assert lines[-2:] == ["TER", "END"]
    assert lines.count("TER") == 1


def test_extract_pocket_pdb_and_sdf_and_fallback(tmp_path: Path):
    pytest.importorskip("Bio")
    pytest.importorskip("scipy")

    prot = tmp_path / "prot.pdb"
    lig_near = tmp_path / "lig_near.pdb"
    lig_far = tmp_path / "lig_far.pdb"
    lig_sdf = tmp_path / "lig.sdf"
    prot.write_text(SAMPLE_PROTEIN_PDB, encoding="utf-8")
    lig_near.write_text(SAMPLE_LIGAND_NEAR_A1_PDB, encoding="utf-8")
    lig_far.write_text(SAMPLE_LIGAND_FAR_PDB, encoding="utf-8")
    lig_sdf.write_text(SAMPLE_LIGAND_V2000_SDF, encoding="utf-8")

    out_pdb = tmp_path / "bs_near.pdb"
    residues = extract_pocket(prot, lig_near, threshold=4.5, save=True, output_path=out_pdb)
    assert residues == [["A", 1]]
    assert out_pdb.is_file()

    sdf_residues = extract_pocket(prot, lig_sdf, threshold=4.5)
    assert sdf_residues == [["A", 1]]

    # At 4.5 A, lig_far (5.0 A away) yields empty pocket, then succeeds at fallback 5.31 A.
    out_fb = tmp_path / "bs_fb.pdb"
    prep = prepare_binding_site(prot, lig_far, out_fb, threshold=4.5, fallback_threshold=5.31)
    assert prep.used_fallback is True
    assert prep.cutoff_used == 5.31
    assert prep.residue_count == 1
    assert "ALA A   1" in out_fb.read_text(encoding="utf-8")


def test_prepare_chemical_features_handles_pdb_substring_in_filename(tmp_path: Path):
    glosa_dir = tmp_path / "glosa"
    glosa_dir.mkdir()
    pocket = tmp_path / "trains.tsv_7pdb_complex.pdb"
    pocket.write_text(SAMPLE_PROTEIN_PDB, encoding="utf-8")

    # Fake java reproducing upstream GLoSA's bug: splits on first 'pdb' in basename.
    fake_java = _make_executable(
        tmp_path / "fake_java",
        f"""#!{sys.executable}
import sys
from pathlib import Path
target = Path(sys.argv[-1])
name = target.name
if "pdb" in name[:-4].lower():
    # Upstream bug corrupts output filename when 'pdb' appears before extension.
    bad = target.with_name(name.split("pdb")[0] + "-cf.pdb")
    bad.write_text("CORRUPTED\\n", encoding="utf-8")
else:
    cf = target.with_name(target.stem + "-cf.pdb")
    cf.write_text("ATOM      1  HD  ALA A   1       0.000   0.000   0.000\\nEND\\n", encoding="utf-8")
""",
    )

    cf = prepare_chemical_features(pocket, glosa_dir, java_bin=str(fake_java))
    assert cf == chemical_features_path(pocket)
    assert cf.is_file()
    assert "ATOM" in cf.read_text(encoding="utf-8")


def test_parse_glosa_stdout():
    sample = b"line0\nline1\nline2\nline3\nGA-score: 0.390705\n"
    assert parse_glosa_stdout(sample) == "0.390705"
    assert parse_glosa_stdout(b"0\n1\n2\n3\nGA-score: -nan\n") == ""
    assert parse_glosa_stdout(b"") == ""


def test_parallel_sharded_score_and_merge_separates_multiligand_pockets(tmp_path: Path):
    pytest.importorskip("Bio")
    pytest.importorskip("scipy")
    pa_pq = pytest.importorskip("pyarrow.parquet")

    db = tmp_path / "db"
    db.mkdir()
    (db / "shared_prot.pdb").write_text(SAMPLE_PROTEIN_PDB, encoding="utf-8")
    (db / "lig1.pdb").write_text(SAMPLE_LIGAND_NEAR_A1_PDB, encoding="utf-8")
    (db / "lig2.pdb").write_text(SAMPLE_LIGAND_NEAR_A2_PDB, encoding="utf-8")

    # Dataset 1 has 3 rows: two different ligands on shared_prot.pdb, plus one exact duplicate pair.
    ds1 = tmp_path / "trains.tsv"
    ds1.write_text(
        "uid\tpath_protein\tpath_ligand\n"
        "u1_lig1\tshared_prot.pdb\tlig1.pdb\n"
        "u1_lig2\tshared_prot.pdb\tlig2.pdb\n"
        "u1_lig1_dup\tshared_prot.pdb\tlig1.pdb\n",
        encoding="utf-8",
    )
    ds2 = tmp_path / "tests.tsv"
    ds2.write_text(
        "uid\tpath_protein\tpath_ligand\n"
        "t1\tshared_prot.pdb\tlig1.pdb\n"
        "t2\tshared_prot.pdb\tlig2.pdb\n",
        encoding="utf-8",
    )

    glosa_dir = tmp_path / "glosa"
    glosa_dir.mkdir()
    fake_java = _make_executable(
        tmp_path / "fake_java",
        f"""#!{sys.executable}
import sys
from pathlib import Path
target = Path(sys.argv[-1])
content = target.read_text(encoding="utf-8")
res = "ALA" if "ALA" in content else "GLY"
cf = target.with_name(target.stem + "-cf.pdb")
cf.write_text(f"ATOM      1  HD  {{res}} A   1       0.000   0.000   0.000\\nEND\\n", encoding="utf-8")
""",
    )
    # Mock glosa binary returning 1.0 when both pockets contain same residue, else 0.25.
    _make_executable(
        glosa_dir / "glosa",
        f"""#!{sys.executable}
import sys
from pathlib import Path
args = sys.argv[1:]
s1 = Path(args[args.index("-s1") + 1]).read_text(encoding="utf-8")
s2 = Path(args[args.index("-s2") + 1]).read_text(encoding="utf-8")
r1 = "ALA" if "ALA" in s1 else "GLY"
r2 = "ALA" if "ALA" in s2 else "GLY"
score = 1.0 if r1 == r2 else 0.25
print("0\\n1\\n2\\n3\\nGA-score:", score)
""",
    )

    bs_dir = tmp_path / "bs"
    prep1 = prepare_dataset_pockets(
        ds1, bs_dir, glosa_dir, databases_dir=db, workers=1, java_bin=str(fake_java)
    )
    prep2 = prepare_dataset_pockets(
        ds2, bs_dir, glosa_dir, databases_dir=db, workers=1, java_bin=str(fake_java)
    )
    assert prep1.total_rows == 3
    assert prep1.unique_complexes == 2
    assert prep1.errors == []
    assert prep2.errors == []

    # Verify per-uid pocket files exist and have distinct contents for lig1 vs lig2.
    entries1 = load_dataset_entries(ds1, bs_dir, databases_dir=db)
    assert "ALA" in entries1[0].pocket_path.read_text(encoding="utf-8")
    assert "GLY" in entries1[1].pocket_path.read_text(encoding="utf-8")

    # Score in two disjoint shard ranges (simulating multi-host / SLURM array).
    shards_dir = tmp_path / "shards"
    s_part1 = run_glosa_shards(
        ds1,
        ds2,
        bs_dir,
        glosa_dir,
        shards_dir,
        databases_dir=db,
        workers=1,
        shard_size=1,
        start_shard=0,
        end_shard=1,
    )
    s_part2 = run_glosa_shards(
        ds1,
        ds2,
        bs_dir,
        glosa_dir,
        shards_dir,
        databases_dir=db,
        workers=1,
        shard_size=1,
        start_shard=1,
        end_shard=2,
    )
    assert s_part1.executed_shards == 1 and s_part2.executed_shards == 1

    # Re-running skips already-completed shards.
    s_resume = run_glosa_shards(
        ds1,
        ds2,
        bs_dir,
        glosa_dir,
        shards_dir,
        databases_dir=db,
        workers=1,
        shard_size=1,
        start_shard=0,
        end_shard=2,
    )
    assert s_resume.executed_shards == 0 and s_resume.skipped_shards == 2

    # Merge to Parquet + max TSV.
    out_dir = tmp_path / "out"
    max_tsv = out_dir / "max.tsv"
    summary = merge_glosa_shards(
        ds1,
        ds2,
        bs_dir,
        shards_dir,
        out_dir,
        databases_dir=db,
        shard_size=1,
        write_tsv=False,
        max_output=max_tsv,
    )
    assert summary.total_rows == 6
    assert summary.finite_scores == 6
    assert summary.null_scores == 0

    table = pa_pq.read_table(summary.output_path)
    rows = table.to_pylist()
    scores_map = {(r["uid_trains.tsv"], r["uid_tests.tsv"]): r["glosa_score"] for r in rows}
    assert scores_map[("u1_lig1", "t1")] == 1.0
    assert scores_map[("u1_lig1", "t2")] == 0.25
    assert scores_map[("u1_lig2", "t1")] == 0.25
    assert scores_map[("u1_lig2", "t2")] == 1.0
    assert scores_map[("u1_lig1_dup", "t1")] == 1.0

    max_lines = max_tsv.read_text(encoding="utf-8").splitlines()
    assert max_lines[0] == "uid_trains.tsv\tuid_tests.tsv\tglosa_score"
    assert max_lines[1] == "u1_lig1\tt1\t1.0"
    assert max_lines[2] == "u1_lig2\tt2\t1.0"

    # Also verify TSV merge mode.
    tsv_summary = merge_glosa_shards(
        ds1,
        ds2,
        bs_dir,
        shards_dir,
        out_dir,
        databases_dir=db,
        shard_size=1,
        write_tsv=True,
    )
    assert tsv_summary.output_path.suffix == ".tsv"
    assert len(tsv_summary.output_path.read_text(encoding="utf-8").splitlines()) == 7


def test_envcheck_pocket_similarity_profile(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        envcheck,
        "_check_module",
        lambda module_name: envcheck.CheckResult(module_name, True, "ok"),
    )
    monkeypatch.setattr(
        envcheck,
        "_check_binary",
        lambda binary_name: envcheck.CheckResult(binary_name, True, "ok"),
    )
    glosa_dir = tmp_path / "glosa"
    glosa_dir.mkdir()
    (glosa_dir / "glosa").write_text("", encoding="utf-8")
    (glosa_dir / "AssignChemicalFeatures.class").write_bytes(b"cls")

    results = envcheck.collect_checks("pocket-similarity", glosa_dir=glosa_dir)
    names = {r.name for r in results}
    assert {
        "numpy",
        "pandas",
        "scipy",
        "Bio",
        "pyarrow",
        "glosa",
        "AssignChemicalFeatures.class",
    } <= names
    assert "pymol" not in names
    assert all(r.ok for r in results)


def test_cli_pocket_similarity_build_glosa_dispatch(tmp_path: Path, monkeypatch):
    called = {}

    def fake_build(glosa_dir, *, output_path=None, cxx="g++", optimize=True):
        called["glosa_dir"] = glosa_dir
        called["optimize"] = optimize
        return tmp_path / "glosa"

    monkeypatch.setattr(cli, "build_glosa_binary", fake_build)
    rc = cli.main(["pocket-similarity", "build-glosa", "--glosa-dir", str(tmp_path)])
    assert rc == 0
    assert called["glosa_dir"] == str(tmp_path)
    assert called["optimize"] is True
    assert os.environ.get("BENTO_NONEXISTENT_SENTINEL") is None
