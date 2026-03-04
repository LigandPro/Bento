# Reproducibility Guide

This document describes a reproducible workflow for Bento using `uv`.

## Environment

1. Clone repository.
2. Sync environment with pinned lock:

```bash
uv sync --frozen --no-editable --extra annotation --extra similarity
```

3. Validate runtime:

```bash
uv run --extra annotation bento check-env --profile annotation
uv run bento check-env --profile similarity --glosa-dir /path/to/glosa_v2.2
```

## Deterministic Inputs

- Keep the same dataset versions in `datasets/`.
- Keep annotation maps in `annotations/` unchanged for a reproducible run.
- Record `git rev-parse --short HEAD` for each experiment.

## Pipeline Execution Order

1. Ligand annotation:

```bash
uv run --extra annotation bento annotate-ligands \
  --dataset-file datasets/tests.tsv \
  --output-dir outputs/
```

2. Pocket similarity:

```bash
uv run bento compute-pocket-similarity \
  --data-csv test_run/path_tests.tsv \
  --protein-path path_protein \
  --ligand-path path_ligand \
  --bs-dir bs \
  --glosa-dir /path/to/glosa_v2.2 \
  --output-file similarity_scores/test_pocket_scores.tsv
```

3. Mapping:

```bash
uv run bento map-annotations \
  --tests-file datasets/tests.tsv \
  --annotations-dir annotations \
  --output-tests-file datasets/tests_annotated.tsv \
  --output-tests-exploded-file datasets/tests_exploded_annotated.tsv
```

## Recommended Run Metadata

Store this metadata in each run artifact:

- Commit SHA.
- Branch name.
- Python version (`uv run python --version`).
- `uv --version`.
- Environment variables:
  - `BENTO_WORKDIR`
  - `BENTO_DATABASES_DIR`
  - `BENTO_GLOSA_DIR`
- Full command lines used for each step.

## Notes

- Java and GLoSA are external runtime dependencies for script 02.
- Build `glosa` and `AssignChemicalFeatures.class` inside `BENTO_GLOSA_DIR` before runs.
- PyMOL is installed from `--extra similarity` on Linux/HPC.
- Use `uv run bento check-env --profile similarity --glosa-dir ...` before long jobs.
