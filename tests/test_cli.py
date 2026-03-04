from __future__ import annotations

from bento import cli


def test_annotate_ligands_dispatch(monkeypatch):
    calls = {}

    def fake_run(script_name, script_args):
        calls["script_name"] = script_name
        calls["script_args"] = script_args
        return 0

    monkeypatch.setattr(cli, "_run_legacy_script", fake_run)
    exit_code = cli.main(
        [
            "annotate-ligands",
            "--dataset-file",
            "datasets/tests.tsv",
            "--output-dir",
            "outputs",
        ]
    )

    assert exit_code == 0
    assert calls["script_name"] == "01_generate_ligands_annotation.py"
    assert calls["script_args"] == [
        "--dataset_file",
        "datasets/tests.tsv",
        "--output_dir",
        "outputs",
    ]


def test_compute_similarity_dispatch(monkeypatch):
    calls = {}

    def fake_run(script_name, script_args):
        calls["script_name"] = script_name
        calls["script_args"] = script_args
        return 0

    monkeypatch.setattr(cli, "_run_legacy_script", fake_run)
    exit_code = cli.main(
        [
            "compute-pocket-similarity",
            "--output-file",
            "scores.csv",
            "--data-csv",
            "input.csv",
            "--protein-path",
            "path_protein",
            "--ligand-path",
            "path_ligand",
            "--bs-dir",
            "bs",
            "--glosa-dir",
            "/opt/glosa",
            "--bs-column",
            "bs_col",
        ]
    )

    assert exit_code == 0
    assert calls["script_name"] == "02_compute_pockets_similarity.py"
    assert "-bs-column" in calls["script_args"]


def test_check_env_returns_status():
    exit_code = cli.main(["check-env", "--profile", "annotation"])
    assert exit_code in {0, 1}
