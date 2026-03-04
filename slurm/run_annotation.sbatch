#!/bin/bash
#SBATCH --job-name=bento-annotate
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

cd "${BENTO_WORKDIR:-$PWD}"
mkdir -p logs outputs

module purge
module load python/3.12

uv sync --frozen --no-editable --extra annotation
uv run bento check-env --profile annotation

uv run --extra annotation bento annotate-ligands \
  --dataset-file datasets/tests.tsv \
  --output-dir outputs/
