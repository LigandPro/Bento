# Running Bento on HPC

This guide provides a practical baseline for SLURM-based execution in a generic HPC environment.

## 1. Prepare Workspace

```bash
git clone https://github.com/LigandPro/Bento.git
cd Bento
cp .env.example .env
```

Edit `.env` with cluster-specific paths:

```bash
BENTO_WORKDIR=/path/to/Bento
BENTO_DATABASES_DIR=/path/to/shared/datasets
BENTO_GLOSA_DIR=/path/to/glosa_v2.2
```

## 2. Load Cluster Modules

Module names can differ by site configuration. Use local equivalents:

```bash
module purge
module load python/3.12
module load java/17
module load gcc
```

## 3. Build GLoSA Runtime (One-Time)

In your `BENTO_GLOSA_DIR`:

```bash
cd "$BENTO_GLOSA_DIR"
g++ -c glosa.cpp
g++ -o glosa glosa.o
javac AssignChemicalFeatures.java
```

## 4. Sync Environment

```bash
uv sync --frozen --no-editable --extra annotation --extra similarity
```

Validate toolchain:

```bash
uv run --extra annotation bento check-env --profile annotation
uv run bento check-env --profile similarity --glosa-dir "$BENTO_GLOSA_DIR"
```

## 5. Submit Jobs

Example templates are provided in `slurm/`:

- `slurm/run_annotation.sbatch`
- `slurm/run_similarity.sbatch`

Submit:

```bash
sbatch slurm/run_annotation.sbatch
sbatch slurm/run_similarity.sbatch
```

## 6. Operational Recommendations

- Keep `uv.lock` committed and use `uv sync --frozen`.
- Run environment checks before expensive jobs.
- Write outputs to job-specific directories to avoid collisions.
- Capture `git` commit hash and full command line in each job log.
