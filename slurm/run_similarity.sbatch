#!/bin/bash
#SBATCH --job-name=bento-similarity
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

cd "${BENTO_WORKDIR:-$PWD}"
mkdir -p logs bs similarity_scores

module purge
module load python/3.12
module load java/17
module load gcc

uv sync --frozen --no-editable --extra similarity
uv run bento check-env --profile similarity --glosa-dir "${BENTO_GLOSA_DIR:?BENTO_GLOSA_DIR is not set}"

uv run bento compute-pocket-similarity \
  --data-csv test_run/path_tests.tsv \
  --protein-path path_protein \
  --ligand-path path_ligand \
  --bs-dir bs \
  --glosa-dir "${BENTO_GLOSA_DIR}" \
  --output-file similarity_scores/test_pocket_scores.tsv
