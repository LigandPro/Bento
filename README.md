# Bento

[![CI](https://github.com/LigandPro/Bento/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/LigandPro/Bento/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10--3.12-blue)
![uv](https://img.shields.io/badge/env-uv-5C4EE5)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/LigandPro/Bento/blob/main/LICENSE)

Bento is a benchmark repository for evaluating protein-ligand docking methods across curated datasets and prediction outputs.

## Highlights

- Unified benchmark assets for ML and classical docking methods.
- Legacy analysis pipeline preserved in `scripts/`.
- UV-first workflow for local development and CI.
- Reproducibility and HPC execution notes.

## Repository Layout

```text
bento/
├── annotated_ligands/          # Train/test tables with ligand-level annotations
├── annotations/                # Annotation maps and ligand class assets
├── datasets/                   # Core benchmark datasets
├── predictions_full_raw/       # Raw predictions from docking methods
├── scripts/                    # Legacy pipeline scripts (01..04)
├── similarity_scores/          # Similarity score artifacts
├── src/bento/                  # Bento CLI wrapper and environment tooling
├── tests/                      # Automated tests
├── docs/                       # Reproducibility and HPC docs
└── slurm/                      # Example SLURM job templates
```

## Quick Start (UV)

### 1. Sync environment

```bash
uv sync --no-editable
```

`--no-editable` is recommended to guarantee the `bento` console entrypoint is available in all Python 3.12 environments.

For development checks:

```bash
uv sync --no-editable --extra lint --extra test
```

For ligand annotation dependencies:

```bash
uv sync --no-editable --extra annotation
```

For pocket similarity dependencies (Linux/HPC):

```bash
uv sync --no-editable --extra similarity
```

### 2. Validate environment

```bash
uv run --extra annotation bento check-env --profile annotation
uv run bento check-env --profile similarity --glosa-dir /path/to/glosa_v2.2
```

### 3. Run pipeline commands

Ligand annotation:

```bash
uv run --extra annotation bento annotate-ligands \
  --dataset-file datasets/tests.tsv \
  --output-dir outputs/
```

Pocket similarity:

```bash
uv run bento compute-pocket-similarity \
  --data-csv test_run/path_tests.tsv \
  --protein-path path_protein \
  --ligand-path path_ligand \
  --bs-dir bs \
  --glosa-dir /path/to/glosa_v2.2 \
  --output-file similarity_scores/test_pocket_scores.tsv
```

Annotation mapping:

```bash
uv run bento map-annotations \
  --tests-file datasets/tests.tsv \
  --annotations-dir annotations \
  --output-tests-file datasets/tests_annotated.tsv \
  --output-tests-exploded-file datasets/tests_exploded_annotated.tsv
```

## Configuration

Legacy scripts use environment variables:

- `BENTO_WORKDIR`: repository root path (default: current repository root).
- `BENTO_DATABASES_DIR`: root path for external dataset files (default: `BENTO_WORKDIR`).
- `BENTO_GLOSA_DIR`: path to GLoSA directory (default: `<repo>/external/glosa`).
- `BENTO_REPO_ROOT`: optional override for CLI location of legacy scripts.

Example:

```bash
export BENTO_WORKDIR=/path/to/Bento
export BENTO_DATABASES_DIR=/path/to/datasets
export BENTO_GLOSA_DIR=/path/to/glosa_v2.2
```

## External Tool Requirements

Some steps require non-Python tools:

- PyMOL (for pocket extraction in script 02; installed with `--extra similarity`
  on Linux/HPC).
- Java JDK and `g++` (for GLoSA tooling).
- GLoSA v2.2 executable and `AssignChemicalFeatures` class.

Build GLoSA once inside your `BENTO_GLOSA_DIR`:

```bash
g++ -c glosa.cpp
g++ -o glosa glosa.o
javac AssignChemicalFeatures.java
```

These tools are validated by:

```bash
uv run bento check-env --profile similarity --glosa-dir /path/to/glosa_v2.2
```

## Quality Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Reproducibility and HPC

- Reproducibility guide: [docs/reproducibility.md](docs/reproducibility.md)
- HPC setup: [docs/hpc.md](docs/hpc.md)
- SLURM templates: `slurm/`

## Citation

```bibtex
@software{bento_benchmark,
  title={Bento: A Comprehensive Benchmark for Protein-Ligand Docking Methods},
  author={LigandPro Team},
  year={2024},
  url={https://github.com/LigandPro/Bento}
}
```

## License

MIT. See [LICENSE](LICENSE).
