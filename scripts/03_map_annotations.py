import os, re, glob, json, ast
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

# --- 1. LOAD DATASETS ---
tests = pd.read_table('datasets/tests.tsv')
tests['ligand'] = tests['ligand'].map(ast.literal_eval).map(tuple)

# --- 2. ADD ANNOTATIONS (PHYS-CHEM PROPS) ---
for annotation in glob.glob('annotations/*json'):
    with open(annotation) as f:
        data = json.load(f)
    name = os.path.basename(annotation).split('.')[0]
    tests[name] = tests['uid'].map(data)

# --- 3. LOAD ANNOTATIONS (LIGAND CLASSES) ---

# Saccharide like
with open('annotations/ligand_classes/saccharide_like.json') as f:
    saccharide_like = json.load(f)
# Cofactors, manually assembled
with open('annotations/ligand_classes/cofactors.json') as f:
    cofactors = json.load(f)
# Modified residues from PDB that are amino acids    
with open('annotations/ligand_classes/modres_aa.json') as f:
    modres_aa = json.load(f)
# Data about all ligands from PDB
ligands_data = pd.read_table('annotations/ligand_classes/ligands_data.tsv') # keep_default_na=False if you need Sodium (NA)

# Annotation of train and test ligands by SMARTS patterns
test_annotation = pd.read_csv('annotations/ligand_classes/test_ligand_classes.csv').set_index('uid')
ligand_groups_dict = {
    'aa':    ['alpha_amino_acids', 'peptide_like', 'long_peptide'], # amino acid like
    'ster':  ['steroids'], # steroids 
    'nt':    ['pyrimidine-nucleotide', 'purine-nucleotide', # nucleot(z)ide like
              'pyrimidine-nucleozide', 'purine-nucleozide'],
    'sac':   ['aldose pyranose', 'ketose pyranose', 'pentose pyranose', # saccaride containing
              'ketose furanose', 'aldose furanose', 'pentose furanose',
              'desoxy-pentose furanose'],
    'macro': ['cycles with >7 members'], # macrocycles with ring > 7
    'eo':    ['at least 3 carbons + metal', 'element_organcs'], # organoelement
    'lip':   ['fatty acids/esters (>8 carbons chain)', 'triglyceride (ester)', # lipid like
              'triglyceride (ether)', 'phospholipid', 'lipide-like'],
    'cof':   ['hem-like', 'biotin-like', 'B6-like', 'flavin-like', 'FMN-like', # cofactors from cofactors.json
              'nicotin-like', 'quinone-like', 'glutathione-like'],
    'spiro': ['spiro'], # spiro
    'fused': ['condensed_system'] # condensed systems
}

# --- Precompute annotation sets ---
def getAnnotation(df, category):
    group_cols = [col for col in ligand_groups_dict[category] if col in df.columns]
    if not group_cols:
        return pd.Index([])
    return df[df[group_cols].fillna(0).gt(0).any(axis=1)].index

annotation_sets = {
    cat: set(getAnnotation(test_annotation, cat))
    for cat in ligand_groups_dict
}

# Cofactors = In cofactors.json or containing words cofactor, ubiquinone... in the ligand name
cofactor_ids = set(cofactors) | set(ligands_data[ligands_data['name'].str.contains('cofactor|ubiquinone|PORPHYRIN|PHEOPHYTIN|CHLOROPHYLL|nicotinamide.*nucleotide|flavin.*nucleotide', case=False)]['ligand_id'])
# Amino acids and peptide like 
aa_ids = set(modres_aa) | set(['ACE', 'NH2']) | set(ligands_data[ligands_data['type'] == 'PEPTIDE-LIKE']['ligand_id']) | \
         set([i.upper() for i in protein_letters_1to3_extended.values()])
# Nucleot(z)ide like
nt_ids = set(ligands_data[ligands_data['pdbx_type'] == 'nucleic acid']['ligand_id']) | \
         set(ligands_data[ligands_data['name'].str.contains('uridin.*phosphate|cytidin.*phosphate|adenin.*phosphate|thymidin.*phosphate|guanosin.*phosphate', case=False)]['ligand_id'])
# Saccaride like
sac_ids = set(ligands_data[(ligands_data['name'].str.contains('L-.*ose$|D-.*ose$', case=False)) & (ligands_data['mw'] < 300)]['ligand_id']) | set(saccharide_like)

def annotateLigands(ligand, uid):
    labels = []
    if ligand in cofactor_ids:
        labels.append('cof')
    if ligand in aa_ids or uid in annotation_sets['aa']:
        labels.append('aa')
    if ligand in nt_ids or uid in annotation_sets['nt']:
        labels.append('nt')
    if ligand in sac_ids:
        labels.append('sac')
    if uid in annotation_sets['ster']:
        labels.append('ster')
    if uid in annotation_sets['lip']:
        labels.append('lip')
    if uid in annotation_sets['macro']:
        labels.append('macro')
    if uid in annotation_sets['eo']:
        labels.append('eo')

    if not labels:
        labels.append('other')

    return tuple(labels) 

def getUniqueTypes(ligand_types):
    counter = Counter(ligand_types)
    if len(counter) == 1:
        return next(iter(counter))
    most_common = counter.most_common()
    return most_common[0][0] if most_common[0][1] > sum(v for _, v in most_common[1:]) else tuple(set(sum(ligand_types, () )))

# --- 4. ADD ANNOTATIONS (LIGAND CLASSES) ---
tests['ligand_types'] = tests.progress_apply(lambda x: tuple(annotateLigands(i, x['uid']) for i in x['ligand']), axis=1)
tests['ligand_types_unique'] = tests['ligand_types'].progress_apply(getUniqueTypes)

# --- 5. EXPLODE TEST ---
tests_exploded = tests.explode('ligand_types_unique', ignore_index=True)
# lipids, steroids and organoelement are sparse so they are united into "etc"
tests_exploded['ligand_types_unique_etc'] = tests_exploded.ligand_types_unique.map(lambda x: 'etc' if x in ['lip', 'ster', 'eo'] else x)
tests_exploded.ligand_types_unique.value_counts()

# --- 6. SAVE ANNOTATED TEST ---
#tests.to_csv('datasets/tests_annotated.tsv', sep='\t', index=False)
#tests_exploded.to_csv('datasets/tests_exploded_annotated.tsv', sep='\t', index=False)