### Environment preparation

###Requires:
### pymol to cut out pockets
#! conda install conda-forge::pymol-open-source -y
### java and javac for glosa
#! conda install bioconda::java-jdk -y
### g++ to compile glosa

### GLoSa homepage https://compbio.lehigh.edu/GLoSA/
# ! wget https://compbio.lehigh.edu/GLoSA/glosa_v2.2.tar.gz -O glosa_v2.2.tar.gz -q
# ! tar -xf glosa_v2.2.tar.gz
# ! cd glosa_v2.2 &&  g++ -c glosa.cpp
# ! cd glosa_v2.2 &&  g++ -o glosa glosa.o
# ! javac glosa_v2.2/AssignChemicalFeatures.java


import os
import subprocess
import pymol
from pymol import cmd
pymol.finish_launching(['pymol','-qc'])


def cut_bs(
    protein_path: str,
    ligand_path: str,
    output_path: str,
    cutoff: float = 4.5,
):

    cmd.delete("all")
    cmd.reinitialize()
    cmd.load(protein_path, "ref_protein")
    cmd.load(ligand_path, "ref_ligand")
    cmd.select("ref_protein_heavy", "ref_protein and not elem H")
    cmd.select("ref_ligand_heavy",  "ref_ligand and not elem H")
    cmd.select("binding_site", f"byres (ref_protein_heavy within {cutoff} of ref_ligand_heavy)")
    
    cmd.save(output_path, "binding_site")
    cmd.delete("all")
    
    
def prepare_binding_site(
    protein_path: str, 
    ligand_path: str, 
    bs_path: str,
):
    os.makedirs(os.path.dirname(bs_path), exist_ok = True)
    
    if 'pdb' in os.path.basename(bs_path).strip('.pdb'):
        print(f"Warning: glosa does not accept 'pdb' substring in filename, rename the file {bs_path}; 'PDB' is acceptable")
   
    cut_bs(protein_path, ligand_path, bs_path, 4.5) # 4.5 angstrom cutoff
    os.system(f"sed -i '/TER/d' {bs_path}") # remove all TER
    os.system(f"sed -i 's/^END$/TER\\nEND/' {bs_path}") ## add one TER
    os.system(f"sed -i '/^ANISOU/d' {bs_path}") # remove ANISOU
    
def prepare_chem_features(bs_path: str, glosa_dir: str):
    subprocess.run(['java', 'AssignChemicalFeatures', 
                   os.path.relpath(bs_path, glosa_dir,)], 
                   shell = False, 
                   cwd = glosa_dir)
    
def calculate_glosa_score(bs_path_1: str, bs_path_2: str, glosa_dir: str, timeout: float = 500):
    try:
        command_list = ['./glosa', 
                        '-s1', 
                        os.path.relpath(bs_path_1, glosa_dir),
                        '-s1cf', 
                        os.path.relpath(bs_path_1[:-4] + '-cf.pdb', glosa_dir),
                       '-s2', 
                        os.path.relpath(bs_path_2, glosa_dir),
                        '-s2cf', 
                        os.path.relpath(bs_path_1[:-4] + '-cf.pdb', glosa_dir,),
                       ]
        
        result = subprocess.run(command_list, shell = False, 
                        cwd = glosa_dir, 
                        capture_output = True, timeout = timeout)
    
    except Exception as e:
        print (f'glosa error for {bs_path_1} {bs_path_2} \n {e} \n')
        return
    
    try:
        if result.stdout.decode('utf-8') == '': return 
        score = result.stdout.decode('utf-8').split('\n')[4]
        score = score.split()[1]
        if score == '-nan': return 
        return float(score)
            
    except Exception as e:
        print (f'glosa error for {bs_path_1} {bs_path_2}\n{e}\n')
        return
    return 

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-output-file", type = str, help = 'file to write calculcated glosa scores csv table')
    parser.add_argument("-data-csv", type = str, help = 'filename for pd.dataframe to read')
    parser.add_argument("-protein-path", type = str, help = 'column name for protein path')
    parser.add_argument("-ligand-path", type = str,  help = 'column name for ligand path')
    parser.add_argument("-bs-dir",    type = str,  help = 'folder name to store binding sites')
    parser.add_argument("-bs-column",  required = False, type = str,  help = 'column name for binding site path')
    parser.add_argument("-glosa-dir", type = str,  help = 'folder with glosa programm')
    args = parser.parse_args()
    
    import pandas as pd
    from tqdm import tqdm
    import itertools
    
    df = pd.read_csv(args.data_csv)
    print(f'Read {args.data_csv} with {len(df)} rows')
    if args.bs_column is None:
        df['bs_path'] =  df[args.protein_path].apply(lambda x: 'bs/'+ os.path.basename(x))
    else:
        df['bs_path'] = df[args.bs_column]
   
    
    for _, row in tqdm(df.iterrows(), desc = 'cut out binding pockets', total = len(df)):
        if os.path.isfile(row.bs_path):
            continue
        prepare_binding_site(row[args.protein_path], row[args.ligand_path], row.bs_path)
        
    for _, row in tqdm(df.iterrows(), desc = 'prepare chemical features', total = len(df)):
        if os.path.isfile(row.bs_path[:-4] + '-cf.pdb'):
            continue
        prepare_chem_features(row.bs_path, args.glosa_dir)

    scores = []
    for bs1, bs2 in tqdm(
        itertools.combinations(df.bs_path.tolist(), 2), 
        ncols = 70, desc = 'calculate scores', 
        total = int(len(df)*(len(df)-1)/2) ):
        score = calculate_glosa_score(bs1, bs2, args.glosa_dir)
        scores.append([bs1, bs2, score])
        
    scores = pd.DataFrame(scores, columns = ['bs1', 'bs2', 'glosa_score'])
    scores.to_csv(args.output_file, index = False)
    print(f'Written {len(scores)} rows to {args.output_file}')

    