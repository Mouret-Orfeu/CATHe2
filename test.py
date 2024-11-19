# libraries
import numpy as np
import pandas as pd 
import argparse
import subprocess
import os
import shutil
import sys
sys.path.append('./src')
from all.get_3Di.get_3Di_sequences import find_best_model, trim_pdb, TrimSelect
import glob


yellow = "\033[93m"
red = "\033[91m"
reset = "\033[0m"

def get_3di_sequences(pdb_folder_path, output_dir):
    """
    Extract 3Di sequences from all PDB files in a folder using Foldseek.
    Combine the 3Di sequences into a single FASTA file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a folder for trimmed PDB files inside pdb_folder_path
    trimmed_pdb_folder = os.path.join(pdb_folder_path, "trimmed_pdb_folder")
    os.makedirs(trimmed_pdb_folder, exist_ok=True)

    # Check for PDB files in the PDB folder
    pdb_file_names = [f for f in os.listdir(pdb_folder_path) if f.endswith(".pdb")]
    if not pdb_file_names:
        raise FileNotFoundError(
            f"No PDB files found. If the selected input_type is AA+3Di, provide PDB files at the folder path given ({pdb_folder_path}) from which 3Di sequences will be extracted."
        )
    
    # Load AA sequences
    dataset_path = "./src/cathe-predict/Dataset.csv"
    dataset_df = pd.read_csv(dataset_path)
    AA_sequences = dataset_df['Sequence'].tolist()

    # DEBUG
    print(f"{yellow} len AA_sequences: {len(AA_sequences)}")

    # trimmed_pdb_file_paths = []

    # Trim PDB files so that the only residues left correspond to the sequences in Sequences.fasta
    for pdb_file_name in pdb_file_names:

        seq_id = int(pdb_file_name.split('_')[0])
        sequence = AA_sequences[seq_id]
        pdb_file_path = os.path.join(pdb_folder_path, pdb_file_name)

        


        best_model_id, best_match_chain_id, _ = find_best_model(pdb_file_path, sequence)
        trimmed_pdb_file_name = f"{os.path.splitext(pdb_file_name)[0]}_trimmed.pdb"

    

        trimmed_pdb_file_path = os.path.join(trimmed_pdb_folder, trimmed_pdb_file_name)



        trim_pdb(pdb_file_path, sequence, best_match_chain_id, best_model_id, best_match_chain_id, trimmed_pdb_file_path)
        # trimmed_pdb_file_paths.append(trimmed_pdb_file_path)

    # FASTA file that will contain all 3Di sequences
    combined_fasta_output = os.path.join(output_dir, "combined_3di_sequences.fasta")

    query_db_path = f"{trimmed_pdb_folder}_queryDB"

    # Open the combined FASTA file in write mode
    with open(combined_fasta_output, 'w') as combined_fasta:

        # foldseek_path = './foldseek/bin/foldseek'

        # Run Foldseek commands to create a sequence database and extract the sequence
        try:

            subprocess.run(f"foldseek createdb {trimmed_pdb_folder} {query_db_path}", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            subprocess.run(f"foldseek lndb {query_db_path}_h {query_db_path}_ss_h", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Convert the created database into a sequence FASTA file (suppressing output)
            subprocess.run(f"foldseek convert2fasta {query_db_path}_ss {combined_fasta_output}", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Replace spaces with underscores in FASTA header lines and prepend "i_"
            # seq_id = int(pdb_file_name.split('_')[0])  # Extract seq_id for the header
            with open(combined_fasta_output, 'r') as fasta_file:
                fasta_content = fasta_file.readlines()

            with open(combined_fasta_output, 'w') as fasta_file:
                for line in fasta_content:
                    if line.startswith(">"):
                        # Prepend the header with "i_<seq_id>_"
                        # line = f">{seq_id}_{line[1:]}"  # Retain the rest of the original header
                        line = line.replace(" ", "_")  # Replace spaces with underscores
                    fasta_file.write(line)

            # Append the contents of this FASTA file to the combined FASTA file
            with open(combined_fasta_output, 'r') as fasta_file:
                combined_fasta.write(fasta_file.read())

            # Clean up temporary files created by foldseek
        
            # Find all files in the pdb_folder_path with "queryDB" in their filenames
            files_to_remove = glob.glob(os.path.join(pdb_folder_path, '*queryDB*'))

            # Remove the files
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

            print(f"3Di sequence for {pdb_file_name} extracted and added to combined file.")

        except subprocess.CalledProcessError as e:
            print(f"{red}An error occurred during 3Di computation for {pdb_file_name}: {e}")

    print(f"All 3Di sequences have been combined into {combined_fasta_output}.")



pdb_path = './src/cathe-predict/PDB_folder' 
output_dir = "./src/cathe-predict/3Di_sequence_folder"

get_3di_sequences(pdb_path, output_dir)