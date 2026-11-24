#!/bin/bash
#SBATCH --job-name=bento-pocket-sim
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --array=0-7

set -euo pipefail

cd "${BENTO_WORKDIR:-$PWD}"
mkdir -p logs

DATASET_1="${BENTO_DATASET_1:?Set BENTO_DATASET_1 (e.g. datasets/trains.tsv)}"
DATASET_2="${BENTO_DATASET_2:?Set BENTO_DATASET_2 (e.g. datasets/tests.tsv)}"
BS_DIR="${BENTO_BS_DIR:?Set BENTO_BS_DIR}"
SHARDS_DIR="${BENTO_SHARDS_DIR:?Set BENTO_SHARDS_DIR}"
GLOSA_DIR="${BENTO_GLOSA_DIR:?Set BENTO_GLOSA_DIR}"
SHARD_SIZE="${BENTO_SHARD_SIZE:-10}"
SHARDS_PER_TASK="${BENTO_SHARDS_PER_TASK:-500}"
WORKERS="${SLURM_CPUS_PER_TASK:-32}"

START_SHARD=$((SLURM_ARRAY_TASK_ID * SHARDS_PER_TASK))
END_SHARD=$(((SLURM_ARRAY_TASK_ID + 1) * SHARDS_PER_TASK))
SLOTS_DIR="/dev/shm/bento-glosa-${SLURM_ARRAY_JOB_ID:-local}-${SLURM_ARRAY_TASK_ID:-0}"
trap 'rm -rf "${SLOTS_DIR}"' EXIT

uv run --extra pocket-similarity bento pocket-similarity score \
  --dataset-tsv-1 "${DATASET_1}" \
  --dataset-tsv-2 "${DATASET_2}" \
  --bs-dir "${BS_DIR}" \
  --glosa-dir "${GLOSA_DIR}" \
  --shards-dir "${SHARDS_DIR}" \
  --slots-dir "${SLOTS_DIR}" \
  --workers "${WORKERS}" \
  --shard-size "${SHARD_SIZE}" \
  --start-shard "${START_SHARD}" \
  --end-shard "${END_SHARD}"
