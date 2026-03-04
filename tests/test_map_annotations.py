from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_map_annotations_script_smoke(tmp_path):
    tests_tsv = tmp_path / "tests.tsv"
    annotations_dir = tmp_path / "annotations"
    ligand_classes_dir = annotations_dir / "ligand_classes"
    ligand_classes_dir.mkdir(parents=True)

    tests_tsv.write_text("uid\tligand\nu1\t('ATP',)\n", encoding="utf-8")

    (annotations_dir / "molecular_weight.json").write_text('{"u1": 507.0}\n', encoding="utf-8")
    (ligand_classes_dir / "saccharide_like.json").write_text("[]\n", encoding="utf-8")
    (ligand_classes_dir / "cofactors.json").write_text("[]\n", encoding="utf-8")
    (ligand_classes_dir / "modres_aa.json").write_text("[]\n", encoding="utf-8")
    (ligand_classes_dir / "ligands_data.tsv").write_text(
        "ligand_id\tname\ttype\tpdbx_type\tmw\nATP\tATP\tOTHER\tnucleic acid\t507\n",
        encoding="utf-8",
    )
    (ligand_classes_dir / "test_ligand_classes.csv").write_text(
        "uid,alpha_amino_acids\nu1,0\n",
        encoding="utf-8",
    )

    output_tests = tmp_path / "tests_annotated.tsv"
    output_exploded = tmp_path / "tests_exploded_annotated.tsv"

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "03_map_annotations.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--tests-file",
            str(tests_tsv),
            "--annotations-dir",
            str(annotations_dir),
            "--output-tests-file",
            str(output_tests),
            "--output-tests-exploded-file",
            str(output_exploded),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_tests.exists()
    assert output_exploded.exists()
