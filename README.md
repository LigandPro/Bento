# Bento

A comprehensive benchmark for evaluating protein-ligand docking methods, featuring predictions from 20+ state-of-the-art docking algorithms including both machine learning-based and traditional approaches.

## Overview

Bento provides a unified framework for analyzing and comparing molecular docking methods on standardized datasets. The benchmark includes predictions from modern ML-based methods (AlphaFold3, Boltz, Chai, DiffDock, NeuralPlexer, etc.) and established classical methods (AutoDock Vina, Smina, Gnina, etc.).

## Repository Structure

```
bento/
├── annotated_ligands/       # Train and test datasets with ligand annotations
├── annotations/             # Ligand property annotations
│   ├── ligand_classes/      # Ligand classification data (cofactors, saccharides, etc.)
│   ├── buried_frac.json     # Buried fraction annotations
│   ├── molecular_weight.json
│   ├── num_atoms.json
│   ├── num_chains.json
│   └── num_rotbonds.json
├── datasets/                # Benchmark datasets (MOAD, PDBbind, Astex, PoseBusters, etc.)
├── predictions_full_raw/    # Raw predictions from all docking methods
├── scripts/                 # Analysis and annotation pipeline
└── similarity_scores/       # Pocket and ligand similarity metrics
```

## Docking Methods Evaluated

### ML-Based Methods
- **AlphaFold3** (af3)
- **Boltz** (multiple variants: standard, pocket-based 4Å/10Å)
- **Chai**
- **DiffDock**
- **Matcha** (multiple variants: all-chains, from-true)
- **NeuralPlexer**
- **Uni-Mol** (standard and p2rank variants)

### Classical Methods
- **AutoDock Vina** (ligand box and p2rank variants)
- **Smina** (ligand box and p2rank variants)
- **Gnina** (ligand box and p2rank variants)
- **FlexAID** (FD)

## Benchmark Datasets

- **Astex**: Diverse protein-ligand complexes
- **PDBbind**: Comprehensive binding affinity dataset
- **DockGen**: Generated docking poses
- **PoseBusters**: Challenging docking benchmark
- **MOAD**: Mother of All Databases
- **Timesplit test**: Temporal split for robust evaluation

## Analysis Pipeline

### 1. Ligand Annotation (`01_generate_ligands_annotation.py`)

Generates comprehensive ligand annotations including:
- Chemical class membership (based on SMARTS patterns)
- Peptide-like characteristics
- Aromatic condensed systems
- Molecular properties (MW, atoms, rotatable bonds)

**Usage:**
```bash
python3 scripts/01_generate_ligands_annotation.py \
  --dataset_file datasets/tests.tsv \
  --output_dir outputs/
```

**Requirements:**
- RDKit
- pandas
- pandarallel

### 2. Pocket Similarity Computation (`02_compute_pockets_similarity.py`)

Computes binding site similarity using GLoSA (Global-Local Structure Alignment):
- Extracts binding pockets (4.5Å cutoff from ligand)
- Calculates structural similarity scores
- Supports batch processing

**Requirements:**
- PyMOL
- GLoSA v2.2
- Java JDK
- g++ compiler

### 3. Annotation Mapping (`03_map_annotations.py`)

Maps generated annotations to prediction datasets for downstream analysis.

### 4. Visualization (`04_make_subsets_and_draw_plots.ipynb`)

Jupyter notebook for:
- Creating dataset subsets
- Generating comparative visualizations
- Statistical analysis of docking performance

## Data Notes

### Large Files

The following files exceed GitHub's 100MB limit and are excluded via `.gitignore`:
- `similarity_scores/pocket_scores.tsv` (123 MB)
- `similarity_scores/tanimoto_distances.csv` (562 MB)

These files contain comprehensive similarity matrices for all analyzed pockets and ligands.

## Configuration

Edit `scripts/config.py` to set paths:
```python
wrk_dir = '/path/to/bento/'
databases_dir = '/path/to/datasets/'
glosa_path = '/path/to/glosa_v2.2/'
```

## Dependencies

Core requirements:
- Python 3.7+
- RDKit
- pandas
- pandarallel
- PyMOL (for pocket extraction)
- GLoSA v2.2 (for similarity computation)
- Java JDK (for GLoSA)

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{bento_benchmark,
  title={Bento: A Comprehensive Benchmark for Protein-Ligand Docking Methods},
  author={LigandPro Team},
  year={2024},
  url={https://github.com/LigandPro/Bento}
}
```

## License

[Add appropriate license information]

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or collaboration inquiries, please open an issue on GitHub.
