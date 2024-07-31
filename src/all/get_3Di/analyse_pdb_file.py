import os
import requests
from Bio.PDB import PDBParser, PPBuilder, is_aa
from Bio.SeqUtils import seq1

def download_pdb(pdb_id, output_dir):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    pdb_file_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    if not os.path.exists(pdb_file_path):
        print(f"Downloading {pdb_id} from {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        with open(pdb_file_path, 'w') as file:
            file.write(response.text)
    else:
        print(f"Using existing file {pdb_file_path}")
    
    return pdb_file_path

def print_pdb_sequences(pdb_file_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file_path)
    ppb = PPBuilder()
    
    for model in structure:
        print(f"Model {model.id}")
        for chain in model:
            print(f"  Chain {chain.id}")
            sequence = ''
            for pp in ppb.build_peptides(chain):
                seq = pp.get_sequence()
                seq = ''.join(seq1(residue.resname) if is_aa(residue) else 'X' for residue in pp)
                sequence += seq
            print(f"    Sequence: {sequence}")

# Example usage:
pdb_id = '1jfi'  # replace with your desired PDB ID
output_dir = './data/pdb_files/Test'
os.makedirs(output_dir, exist_ok=True)

pdb_file_path = download_pdb(pdb_id, output_dir)
print_pdb_sequences(pdb_file_path)
