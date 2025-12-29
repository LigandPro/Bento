'''
Input:
- A dataset of complexes containing a column `path_ligand` with paths to ligand files.

Requirements:
- A SMARTS pattern file defining each ligand class:
  `annotations/ligand_classes/classes_smarts.csv`

Output:
- A file `ligand_classes_{input_dataset_name}.tsv` containing, for each ligand in the dataset,
  annotations of ligand class membership
  (0 = not a member; otherwise, the number of matched patterns).

Test run:
python3 01_generate_ligands_annotation.py \
  --dataset_file ../test_run/path_train.tsv \
  --output_dir ../test_run/outputs
'''

import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem
from typing import List, Tuple, Set
from collections import defaultdict
import psutil
from rdkit.Chem import Descriptors
import os
import argparse
from config import wrk_dir, databases_dir

smarts_path = os.path.join(wrk_dir, "annotations/ligand_classes/classes_smarts.csv")
SMARTS_DF = pd.read_csv(smarts_path)

# Pre-compile SMARTS patterns once at module level to avoid recompilation
COMPILED_SMARTS = {}
print("Pre-compiling SMARTS patterns...")
for _, row in SMARTS_DF.iterrows():
    try:
        compiled_smarts = Chem.MolFromSmarts(row['smarts'])
        COMPILED_SMARTS[row['local_group']] = compiled_smarts
        if compiled_smarts is None:
            print(f"Warning: Failed to compile SMARTS {row['smarts']} for {row['local_group']}")
    except Exception as e:
        print(f"Error compiling SMARTS {row['smarts']} for {row['local_group']}: {e}")
        COMPILED_SMARTS[row['local_group']] = None
print(f"Compiled {len([v for v in COMPILED_SMARTS.values() if v is not None])} SMARTS patterns successfully")


def get_mol(path: str):
    ext = path[path.rfind(".") + 1:]
    if ext == "pdb":
        mol = Chem.MolFromPDBFile(path)
    elif ext == "sdf":
        mol = Chem.MolFromMolFile(path)
    else:
        mol = None
    return mol


def find_peptide_chain_length(mol: Chem.Mol) -> Tuple[int, List[List[int]]]:
    if mol is None:
        return 0, []

    peptide_bond_smarts = Chem.MolFromSmarts("[$([#6](=O))][#6H,#6H2][NH,NH2]")
    matches = mol.GetSubstructMatches(peptide_bond_smarts)

    if len(matches) < 2:
        return len(matches), [list(match) for match in matches]

    peptide_units = _build_peptide_connectivity_graph(mol, matches)
    chains = _find_peptide_chains(peptide_units)
    max_length = max(len(chain) for chain in chains) if chains else 0

    return max_length, chains


def _build_peptide_connectivity_graph(mol: Chem.Mol, matches: Tuple) -> defaultdict:
    peptide_graph = defaultdict(set)
    for i, match1 in enumerate(matches):
        for j, match2 in enumerate(matches):
            if i != j and _are_peptide_units_connected(mol, match1, match2):
                peptide_graph[i].add(j)
                peptide_graph[j].add(i)
    return peptide_graph


def _are_peptide_units_connected(mol: Chem.Mol, match1: Tuple[int], match2: Tuple[int]) -> bool:
    if set(match1) & set(match2):
        return True

    for atom_idx1 in match1:
        atom1 = mol.GetAtomWithIdx(atom_idx1)
        for neighbor in atom1.GetNeighbors():
            if neighbor.GetIdx() in match2:
                return True
    return False


def _find_peptide_chains(peptide_graph: defaultdict) -> List[List[int]]:
    visited = set()
    chains = []

    for start_unit in peptide_graph:
        if start_unit not in visited:
            chain = _get_longest_path_from_unit(peptide_graph, start_unit, visited)
            if chain:
                chains.append(chain)

    all_units = set(peptide_graph.keys())
    isolated_units = all_units - visited
    for unit in isolated_units:
        chains.append([unit])

    return chains


def _get_longest_path_from_unit(peptide_graph: defaultdict, start_unit: int, visited: Set[int]) -> List[int]:
    def _dfs_longest_path(current: int, path: List[int], current_visited: Set[int]) -> List[int]:
        longest = path[:]
        for neighbor in peptide_graph[current]:
            if neighbor not in current_visited:
                current_visited.add(neighbor)
                new_path = _dfs_longest_path(neighbor, path + [neighbor], current_visited)
                if len(new_path) > len(longest):
                    longest = new_path
                current_visited.remove(neighbor)
        return longest

    visited.add(start_unit)
    longest_chain = _dfs_longest_path(start_unit, [start_unit], {start_unit})
    for unit in longest_chain:
        visited.add(unit)
    return longest_chain


def analyze_peptide_content(mol: Chem.Mol) -> dict:
    max_length, chains = find_peptide_chain_length(mol)
    return {
        'is_peptide_like': max_length >= 2,
        'is_long_peptide': max_length >= 5
    }


def is_aromatic_condensed_system(mol: Chem.Mol, min_rings: int = 2) -> bool:
    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()

    if len(rings) < min_rings:
        return False

    aromatic_rings = []
    for ring in rings:
        ring_atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
        if all(atom.GetIsAromatic() for atom in ring_atoms):
            aromatic_rings.append(ring)

    if len(aromatic_rings) < min_rings:
        return False

    condensed_count = 0
    for i, ring1 in enumerate(aromatic_rings):
        for ring2 in aromatic_rings[i + 1:]:
            if len(set(ring1) & set(ring2)) >= 2:
                condensed_count += 1

    return condensed_count >= min_rings - 1


def annotate(row):
    mol = row["mol"]

    if (mol is None) or (Descriptors.MolWt(mol) > 2500):
        for group_name in COMPILED_SMARTS.keys():
            row[group_name] = None
        row["peptide_like"] = None
        row["long_peptide"] = None
        row["condensed_system"] = None
        return row

    for group_name, smarts_mol in COMPILED_SMARTS.items():
        if smarts_mol is not None:
            row[group_name] = len(mol.GetSubstructMatches(smarts_mol))
        else:
            row[group_name] = None

    pept_info = analyze_peptide_content(mol)
    row["peptide_like"] = int(pept_info["is_peptide_like"])
    row["long_peptide"] = int(pept_info["is_long_peptide"])
    row["condensed_system"] = int(is_aromatic_condensed_system(mol))

    return row


def process_in_batches(df: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
    results = []
    total_batches = (len(df) - 1) // batch_size + 1

    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size].copy()
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches}")

        try:
            processed_batch = batch.parallel_apply(annotate, axis=1)
        except Exception:
            processed_batch = batch.apply(annotate, axis=1)

        results.append(processed_batch)

    return pd.concat(results, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", required=True, help="Input dataset TSV file")
    parser.add_argument("--output_dir", required=True, help="Output directory to save result file ligand_classes.tsv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    suffix = os.path.basename(args.dataset_file)
    suffix = os.path.splitext(suffix)[0]

    print("Loading dataset...")
    df = pd.read_csv(args.dataset_file, sep="\t")
    print(f"Loaded {len(df)} samples")

    print("Loading molecules...")
    df["path_ligand"] = df["path_ligand"].map(lambda x: os.path.join(databases_dir, x))
    df["mol"] = df["path_ligand"].apply(get_mol)
    print(f"Successfully loaded {df['mol'].notna().sum()}/{len(df)} molecules")

    optimal_workers = max(1, psutil.cpu_count(logical=False) - 2)
    pandarallel.initialize(nb_workers=optimal_workers, progress_bar=True, verbose=1)

    df = process_in_batches(df, batch_size=10000)

    print("Cleaning up and saving results...")
    df = df.drop(columns=["mol"])
    output_path = os.path.join(args.output_dir, f"ligand_classes_{suffix}.tsv")
    df.to_csv(output_path, index=False, sep='\t')

    print(f"Results saved to {output_path}")
    print("Processing completed successfully!")


if __name__ == "__main__":
    main()
