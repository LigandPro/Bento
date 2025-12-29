import pandas as pd
from pandarallel import pandarallel
from rdkit import Chem
from typing import List, Tuple, Set
from collections import defaultdict
import psutil
from rdkit.Chem import Descriptors
import os
from config import wrk_dir

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

def get_mol(path:str):
    ext = path[path.rfind(".")+1:]
    if ext == "pdb":
        mol = Chem.MolFromPDBFile(path)
    elif ext == "sdf":
        mol = Chem.MolFromMolFile(path)
    else:
        mol = None
    return mol

def find_peptide_chain_length(mol: Chem.Mol) -> Tuple[int, List[List[int]]]:
    """
    Determines the length of peptide chains in a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Tuple containing:
        - Maximum peptide chain length
        - List of all peptide chains (each chain is a list of atom indices)
    """
    if mol is None:
        return 0, []
    
    # Define peptide bond SMARTS pattern
    peptide_bond_smarts = Chem.MolFromSmarts("[$([#6](=O))][#6H,#6H2][NH,NH2]")
    
    # Get all peptide bond matches
    matches = mol.GetSubstructMatches(peptide_bond_smarts)
    
    if len(matches) < 2:
        return len(matches), [list(match) for match in matches]
    
    # Build connectivity graph between peptide units
    peptide_units = _build_peptide_connectivity_graph(mol, matches)
    
    # Find all connected chains
    chains = _find_peptide_chains(peptide_units)
    
    # Return the longest chain length and all chains
    max_length = max(len(chain) for chain in chains) if chains else 0
    
    return max_length, chains


def _build_peptide_connectivity_graph(mol: Chem.Mol, matches: Tuple) -> defaultdict:
    """
    Builds a graph showing which peptide units are connected.
    
    Args:
        mol: RDKit molecule object
        matches: Tuple of peptide bond matches from GetSubstructMatches
        
    Returns:
        Dictionary mapping peptide unit indices to their connected neighbors
    """
    # Create adjacency list for peptide units
    peptide_graph = defaultdict(set)
    
    # For each pair of peptide bonds, check if they share atoms (are connected)
    for i, match1 in enumerate(matches):
        for j, match2 in enumerate(matches):
            if i != j and _are_peptide_units_connected(mol, match1, match2):
                peptide_graph[i].add(j)
                peptide_graph[j].add(i)
    
    return peptide_graph


def _are_peptide_units_connected(mol: Chem.Mol, match1: Tuple[int], match2: Tuple[int]) -> bool:
    """
    Checks if two peptide units share atoms or are directly connected.
    
    Args:
        mol: RDKit molecule object
        match1: First peptide unit atom indices
        match2: Second peptide unit atom indices
        
    Returns:
        True if peptide units are connected, False otherwise
    """
    # Check if peptide units share any atoms
    if set(match1) & set(match2):
        return True
    
    # Check if any atom in match1 is bonded to any atom in match2
    for atom_idx1 in match1:
        atom1 = mol.GetAtomWithIdx(atom_idx1)
        for neighbor in atom1.GetNeighbors():
            if neighbor.GetIdx() in match2:
                return True
    
    return False


def _find_peptide_chains(peptide_graph: defaultdict) -> List[List[int]]:
    """
    Finds all maximal peptide chains in the connectivity graph.
    
    Args:
        peptide_graph: Graph of connected peptide units
        
    Returns:
        List of peptide chains, where each chain is a list of peptide unit indices
    """
    visited = set()
    chains = []
    
    # Find all connected components (chains)
    for start_unit in peptide_graph:
        if start_unit not in visited:
            chain = _get_longest_path_from_unit(peptide_graph, start_unit, visited)
            if chain:
                chains.append(chain)
    
    # Handle isolated peptide units (not connected to others)
    all_units = set(peptide_graph.keys())
    isolated_units = all_units - visited
    for unit in isolated_units:
        chains.append([unit])
    
    return chains


def _get_longest_path_from_unit(peptide_graph: defaultdict, start_unit: int, visited: Set[int]) -> List[int]:
    """
    Finds the longest path starting from a given peptide unit using DFS.
    
    Args:
        peptide_graph: Graph of connected peptide units
        start_unit: Starting peptide unit index
        visited: Set of already visited units
        
    Returns:
        Longest chain starting from the given unit
    """
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
    
    # Mark all units in the longest chain as visited
    for unit in longest_chain:
        visited.add(unit)
    
    return longest_chain


def analyze_peptide_content(mol: Chem.Mol) -> dict:
    """
    Comprehensive analysis of peptide content in a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Dictionary with peptide analysis results
    """
    max_length, chains = find_peptide_chain_length(mol)
    
    return {
        'is_peptide_like': max_length >= 2,
        'is_long_peptide': max_length >= 5
    }
    

def is_aromatic_condensed_system(mol: Chem.Mol, min_rings: int = 2) -> bool:
    """
    Determine if a molecule is an aromatic condensed system with specified minimum rings.
    
    Args:
        mol: RDKit molecule object
        min_rings: Minimum number of aromatic rings to consider as condensed system
        
    Returns:
        True if molecule has at least min_rings condensed aromatic rings
    """
    # Get ring info
    ring_info = mol.GetRingInfo()
    rings = ring_info.AtomRings()
    
    if len(rings) < min_rings:
        return False
    
    # Filter only aromatic rings
    aromatic_rings = []
    for ring in rings:
        ring_atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
        if all(atom.GetIsAromatic() for atom in ring_atoms):
            aromatic_rings.append(ring)
    
    if len(aromatic_rings) < min_rings:
        return False
    
    # Check if aromatic rings share edges (condensed)
    condensed_count = 0
    for i, ring1 in enumerate(aromatic_rings):
        for j, ring2 in enumerate(aromatic_rings[i+1:], i+1):
            # Check if rings share at least one edge
            shared_atoms = set(ring1) & set(ring2)
            if len(shared_atoms) >= 2:  # At least 2 atoms shared
                condensed_count += 1
    
    return condensed_count >= min_rings - 1


def annotate(row):
    """
    Optimized annotation function with better error handling and memory usage.
    
    Args:
        row: DataFrame row containing molecule data
        
    Returns:
        Row with annotations added
    """
    mol = row["mol"]
    
    # Early return for None molecules
    if (mol is None) or (Descriptors.MolWt(mol) > 2500):
        for group_name in COMPILED_SMARTS.keys():
            row[group_name] = None
        row["peptide_like"] = None
        row["long_peptide"] = None
        row["condensed_system"] = None
        return row
    
    try:
        # Process SMARTS patterns using pre-compiled patterns
        for group_name, smarts_mol in COMPILED_SMARTS.items():
            if smarts_mol is not None:
                try:
                    matches = mol.GetSubstructMatches(smarts_mol)
                    row[group_name] = len(matches)
                except Exception as e:
                    print(f"Error matching SMARTS for {group_name}: {e}")
                    row[group_name] = 0
            else:
                row[group_name] = None
        
        # Process peptide analysis
        try:
            pept_info = analyze_peptide_content(mol)
            row["peptide_like"] = int(pept_info["is_peptide_like"])
            row["long_peptide"] = int(pept_info["is_long_peptide"])
        except Exception as e:
            print(f"Error in peptide analysis for row {getattr(row, 'name', 'unknown')}: {e}")
            row["peptide_like"] = None
            row["long_peptide"] = None
        
        # Process condensed system analysis
        try:
            row["condensed_system"] = int(is_aromatic_condensed_system(mol))
        except Exception as e:
            print(f"Error in condensed system analysis for row {getattr(row, 'name', 'unknown')}: {e}")
            row["condensed_system"] = None
            
    except Exception as e:
        print(f"General error processing row {getattr(row, 'name', 'unknown')}: {e}")
        # Set all to None on general failure
        for group_name in COMPILED_SMARTS.keys():
            row[group_name] = None
        row["peptide_like"] = None
        row["long_peptide"] = None
        row["condensed_system"] = None
    
    return row


def process_in_batches(df: pd.DataFrame, batch_size: int = 1000) -> pd.DataFrame:
    """
    Process DataFrame in smaller batches to avoid memory issues.
    
    Args:
        df: DataFrame to process
        batch_size: Size of each batch
        
    Returns:
        Processed DataFrame
    """
    results = []
    total_batches = (len(df) - 1) // batch_size + 1
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size].copy()
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches} (rows {i}-{min(i+batch_size-1, len(df)-1)})")
        
        try:
            processed_batch = batch.parallel_apply(annotate, axis=1)
        except Exception as e:
            print(f"Batch {batch_num} failed with parallel processing: {e}")
            print("Falling back to sequential processing for this batch...")
            processed_batch = batch.apply(annotate, axis=1)
        
        results.append(processed_batch)
    
    return pd.concat(results, ignore_index=True)


def main():
    train_path = os.path.join(wrk_dir, "test_run/path_tests.tsv") #os.path.join(wrk_dir, "/datasets/path_tests.tsv")
    test_path = os.path.join(wrk_dir, "test_run/path_train.tsv") #os.path.join(wrk_dir, "/datasets/path_train.tsv")
    SAVEDIR = os.path.join(wrk_dir, "test_run") #os.path.join(wrk_dir, "annotated_ligands")
    
    print("Loading datasets...")
    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")
    print(f"Loaded {len(train_df)} training samples and {len(test_df)} test samples")

    print("Loading molecules...")
    train_df["mol"] = train_df["path_ligand"].apply(get_mol)
    test_df["mol"] = test_df["path_ligand"].apply(get_mol)
    
    # Check for failed molecule loading
    train_none = train_df["mol"].isna().sum()
    test_none = test_df["mol"].isna().sum()
    print(f"Successfully loaded molecules: {len(train_df) - train_none}/{len(train_df)} train, {len(test_df) - test_none}/{len(test_df)} test")
    if train_none > 0 or test_none > 0:
        print(f"Warning: Failed to load {train_none} train and {test_none} test molecules")

    # Conservative pandarallel initialization
    optimal_workers = max(1, psutil.cpu_count(logical=False) - 2)  # Physical cores minus 1
    print(f"Initializing pandarallel with {optimal_workers} workers (detected {psutil.cpu_count(logical=False)} physical cores)")
    
    try:
        pandarallel.initialize(
            nb_workers=optimal_workers,
            progress_bar=True,
            verbose=1,
    #        use_memory_fs=False  # Disable memory filesystem to avoid issues
        )
        
        print("Processing training data...")
        train_df = process_in_batches(train_df, batch_size=10000)
        
        print("Processing test data...")
        test_df = process_in_batches(test_df, batch_size=10000)
        
    except Exception as e:
        print(f"Pandarallel initialization or processing failed: {e}")
        print("Falling back to sequential processing...")
        train_df = train_df.apply(annotate, axis=1)
        test_df = test_df.apply(annotate, axis=1)

    # Clean up molecules before saving to reduce file size
    print("Cleaning up and saving results...")
    try:
        train_df = train_df.drop(columns=["mol"])
        test_df = test_df.drop(columns=["mol"])
        print("Removed molecule objects from DataFrames")
    except KeyError:
        print("Warning: mol column was not found for dropping")

    train_df.to_csv(f"{SAVEDIR}/train_ligand_classes.csv", index=False)
    test_df.to_csv(f"{SAVEDIR}/test_ligand_classes.csv", index=False)
    print(f"Results saved to {SAVEDIR}/")
    print("Processing completed successfully!")


if __name__ == "__main__":
    main()
