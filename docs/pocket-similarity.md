# Parallel GLoSA Pocket Similarity

`bento pocket-similarity` computes all-against-all GLoSA v2.2 binding-site similarities between two dataset tables (`dataset_tsv_1 × dataset_tsv_2`) using PyMOL-free coordinate extraction (`scipy.spatial.cKDTree` + Biopython), per-`uid` pocket naming, isolated per-worker working directories, resumable sharding, and streaming Parquet output.

## Key Properties

- **Collision-free pocket naming**: every complex is stored as `<dataset_filename>_<uid>.pdb` and `<dataset_filename>_<uid>-cf.pdb`, preventing multi-ligand proteins from overwriting each other's pockets.
- **Exact deduplication without accuracy loss**: identical `(path_protein, path_ligand)` pairs in `dataset_tsv_1` are extracted and scored once, then expanded to every matching `uid` during merge. Distinct ligands bound to the same protein always receive separate pockets and scores.
- **Safe `AssignChemicalFeatures` handling**: upstream GLoSA's Java feature generator strips the first `"pdb"` substring from the input filename, which corrupts identifiers such as `2pdb`, `3pdb`, or `7pdb`. `bento pocket-similarity prepare` automatically runs Java through a neutral temporary filename whenever `"pdb"` appears in the stem.
- **3x faster deterministic GLoSA build**: compiling `glosa.cpp` with `g++ -O3 -ffp-contract=off` preserves bit-for-bit IEEE-754 GA-scores while increasing per-core throughput roughly threefold compared to unoptimized `-O0` builds.
- **Isolated per-worker slots**: each worker runs `./glosa` in its own directory (ideally on `/dev/shm` RAM disk) so concurrent workers never collide on GLoSA's temporary files (`ali_struct.pdb`, `matrix.txt`, `pairs.rst`, `product_graph.rst`).
- **Compact Parquet output**: merging writes `<output_dir>/pocket_scores_<dataset1>_<dataset2>.parquet` (`uid_<dataset1>`, `uid_<dataset2>`, `glosa_score`) via streaming `pyarrow` row groups (~1.5 GB for 190M pairs instead of ~49 GB uncompressed TSV).

## 1. Install Dependencies and Validate

```bash
uv sync --frozen --no-editable --extra pocket-similarity
uv run --extra pocket-similarity bento check-env \
  --profile pocket-similarity \
  --glosa-dir "$BENTO_GLOSA_DIR"
```

Include `--extra annotation` as well if your datasets contain `.mol` or `.mol2` ligands requiring RDKit (`.pdb`, `.cif`, `.mmcif`, and V2000/V3000 `.sdf` work without RDKit).

## 2. Build Optimized GLoSA Runtime (One-Time)

Compile `glosa` with deterministic optimization flags and ensure `AssignChemicalFeatures.class` is compiled:

```bash
uv run --extra pocket-similarity bento pocket-similarity build-glosa \
  --glosa-dir "$BENTO_GLOSA_DIR"
javac "$BENTO_GLOSA_DIR/AssignChemicalFeatures.java"
```

## 3. Single-Host End-to-End Run

Run pocket extraction, sharded parallel scoring, and Parquet + max-score merging in one command:

```bash
uv run --extra pocket-similarity bento pocket-similarity run \
  --dataset-tsv-1 datasets/trains.tsv \
  --dataset-tsv-2 datasets/tests.tsv \
  --databases-dir "$BENTO_DATABASES_DIR" \
  --bs-dir outputs/pocket_similarity/glosa_bs_files \
  --glosa-dir "$BENTO_GLOSA_DIR" \
  --output-dir outputs/pocket_similarity \
  --slots-dir /dev/shm/bento-glosa-slots \
  --workers 64 \
  --shard-size 10 \
  --max-output outputs/pocket_similarity/pocket_scores_max.tsv
```

Optional flags:

- `--fallback-threshold 5.31`: if a complex produces zero contacting residues at the default `4.5` Å cutoff, retry once at the fallback radius instead of leaving an empty pocket (`null` scores).
- `--tsv`: write a 3-column TSV instead of Parquet.

If interrupted, re-running the exact same command skips already-completed shards (`<shard>.ok`) and already-extracted pockets automatically.

## 4. Multi-Host or SLURM Array Execution

For large runs (tens to hundreds of millions of pairs), split the workflow into three explicit stages.

### Stage A: Prepare Pockets Once

```bash
uv run --extra pocket-similarity bento pocket-similarity prepare \
  --dataset-tsv-1 datasets/trains.tsv \
  --dataset-tsv-2 datasets/tests.tsv \
  --databases-dir "$BENTO_DATABASES_DIR" \
  --bs-dir /shared/bento/glosa_bs_files \
  --glosa-dir "$BENTO_GLOSA_DIR" \
  --workers 16
```

### Stage B: Score Disjoint Shard Ranges Across Hosts or Array Tasks

Choose non-overlapping `[--start-shard, --end-shard)` ranges. For maximum throughput on Linux servers with ample RAM, copy `glosa_bs_files` and the compiled `glosa` binary onto local `/dev/shm` so scoring performs zero network or disk reads.

Host A (for example, shards `0..1999`):

```bash
uv run --extra pocket-similarity bento pocket-similarity score \
  --dataset-tsv-1 /dev/shm/bento/trains.tsv \
  --dataset-tsv-2 /dev/shm/bento/tests.tsv \
  --bs-dir /dev/shm/bento/glosa_bs_files \
  --glosa-dir /dev/shm/bento \
  --shards-dir /local/bento/shards \
  --slots-dir /dev/shm/bento/slots \
  --workers 128 \
  --shard-size 10 \
  --start-shard 0 \
  --end-shard 2000
```

Host B (remaining shards `2000..`):

```bash
uv run --extra pocket-similarity bento pocket-similarity score \
  --dataset-tsv-1 /dev/shm/bento/trains.tsv \
  --dataset-tsv-2 /dev/shm/bento/tests.tsv \
  --bs-dir /dev/shm/bento/glosa_bs_files \
  --glosa-dir /dev/shm/bento \
  --shards-dir /local/bento/shards \
  --slots-dir /dev/shm/bento/slots \
  --workers 96 \
  --shard-size 10 \
  --start-shard 2000
```

On SLURM clusters, submit the provided array template:

```bash
BENTO_DATASET_1=datasets/trains.tsv \
BENTO_DATASET_2=datasets/tests.tsv \
BENTO_BS_DIR=/shared/bento/glosa_bs_files \
BENTO_SHARDS_DIR=/shared/bento/shards \
BENTO_GLOSA_DIR="$BENTO_GLOSA_DIR" \
sbatch slurm/run_pocket_similarity_array.sbatch
```

### Stage C: Collect Shards and Merge to Parquet

Copy all completed `<shard>.tsv` and `<shard>.ok` files into one directory, then merge:

```bash
uv run --extra pocket-similarity bento pocket-similarity merge \
  --dataset-tsv-1 datasets/trains.tsv \
  --dataset-tsv-2 datasets/tests.tsv \
  --bs-dir /shared/bento/glosa_bs_files \
  --shards-dir /shared/bento/shards \
  --output-dir /shared/bento/output \
  --shard-size 10 \
  --max-output /shared/bento/output/pocket_scores_max.tsv
```

## Output Schema

Default Parquet file (`pocket_scores_<dataset1>_<dataset2>.parquet`):

| Column | Type | Description |
| --- | --- | --- |
| `uid_<dataset1>` | `string` | `uid` from `--dataset-tsv-1` |
| `uid_<dataset2>` | `string` | `uid` from `--dataset-tsv-2` |
| `glosa_score` | `double` | GLoSA GA-score (`null` if unavailable) |
