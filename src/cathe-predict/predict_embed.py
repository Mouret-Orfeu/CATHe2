# ANSI escape code for colored text
yellow = "\033[93m"
green = "\033[92m"
reset = "\033[0m"
red = "\033[91m"

import sys
import os
import argparse

parser = argparse.ArgumentParser(description="Run predictions pipeline with FASTA file")
parser.add_argument('--model', type=str, default='ProtT5', choices=['ProtT5', 'ProstT5'], help="Model to use: ProtT5 (original one) or ProstT5 (new one)")
parser.add_argument('--input_type', type=str, default='AA', choices=['AA', 'AA+3Di'], help="Input type: AA or AA+3Di (AA+3Di is only supported by ProstT5)")
args = parser.parse_args()

# Check if a virtual environment is active
if not hasattr(sys, 'base_prefix') or sys.base_prefix == sys.prefix:
    raise EnvironmentError(f"{red}No virtual environment is activated. Please activate the right venv first, see ReadMe for more details.{reset}")

# Get the name of the activated virtual environment
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path is None:
    raise EnvironmentError(f"{red}Error, venv path is none. Please activate the right venv first, see ReadMe for more details.{reset}")

venv_name = os.path.basename(venv_path)
if args.model == 'ProtT5' and venv_name != "venv_1":
    raise EnvironmentError(f"{red}The activated virtual environment is '{venv_name}', not 'venv_1'. If you want to use the ProtT5 model, venv_1 must be activated. See ReadMe for more details.{reset}")
if args.model == 'ProstT5' and venv_name != "venv_2":
    raise EnvironmentError(f"{red}The activated virtual environment is '{venv_name}', not 'venv_2'. If you want to use the ProstT5 model, venv_2 must be activated. See ReadMe for more details.{reset}")
if venv_name != "venv_1" and venv_name != "venv_2":
    raise EnvironmentError(f"{red}The activated virtual environment is '{venv_name}', but it should be 'venv_1' or 'venv_2'. See ReadMe for more details.{reset}")

# libraries
import numpy as np
import pandas as pd 
import subprocess
import shutil
sys.path.append('./src')
from all.get_3Di.get_3Di_sequences import find_best_model, trim_pdb, TrimSelect
import glob

def embed_sequence(model):

    if model == 'ProtT5':
        from bio_embeddings.embed import ProtTransT5BFDEmbedder

        print(f"{yellow}Loading ProtT5 model. (can take a few minutes){reset}")

        embedder = ProtTransT5BFDEmbedder()
        ds = pd.read_csv('./src/cathe-predict/Dataset.csv')

        sequences_Example = list(ds["Sequence"])
        num_seq = len(sequences_Example)

        i = 0
        length = 1000
        while i < num_seq:
            print("Doing", i, num_seq)
            start = i 
            end = i + length

            sequences = sequences_Example[start:end]

            embeddings = []
            for seq in sequences:
                embeddings.append(np.mean(np.asarray(embedder.embed(seq)), axis=0))

            s_no = start / length
            filename = './src/cathe-predict/Embeddings/' + 'ProtT5_' + str(s_no) + '.npz'

            embeddings = np.asarray(embeddings)
            np.savez_compressed(filename, embeddings)
            i += length
    
    if model == 'ProstT5':

        print("Embedding sequences with ProstT5")

        # Define the arguments for embed_all_models.py
        args = [
            'python3', './src/all/models/all_models/embed_all_models.py',
            '--model', 'ProstT5_full',
            '--is_3Di', '0',  
            '--seq_path', './src/cathe-predict/Dataset.csv',  
            '--embed_path', './src/cathe-predict/Embeddings/Embeddings_ProstT5_AA.npz',  # Path where embeddings will be saved
        ]

        # Run embed_all_models.py with the specified arguments
        subprocess.run(args)

        
        


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

    # trimmed_pdb_file_paths = []

    # Trim PDB files so that the only residues left correspond to the sequences in Sequences.fasta
    for pdb_file_name in pdb_file_names:

        seq_id = int(pdb_file_name.split('_')[0])
        sequence = AA_sequences[seq_id]
        pdb_file_path = os.path.join(pdb_folder_path, pdb_file_name)

        # DEBUG 
        print(f"{yellow} pdb_file_path: {pdb_file_path}")
        print(f"{yellow} pdb_file_name: {pdb_file_name}")


        best_model_id, best_match_chain_id, _ = find_best_model(pdb_file_path, sequence)
        trimmed_pdb_file_name = f"{os.path.splitext(pdb_file_name)[0]}_trimmed.pdb"

        # DEBUG
        print(f"{yellow} trimmed_pdb_file_name: {trimmed_pdb_file_name}")

        trimmed_pdb_file_path = os.path.join(trimmed_pdb_folder, trimmed_pdb_file_name)

        # DEBUG
        print(f"{yellow} trimmed_pdb_file_path: {trimmed_pdb_file_path}")

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


def embed_3Di(pdb_path):
    fasta_file_3Di = "./src/cathe-predict/3Di_sequence_folder/combined_3di_sequences.fasta"
    embed_path = "./src/cathe-predict/Embeddings/3Di_embeddings.npz"

    # Get the 3Di sequence
    # !!!!!!!!!!!!!!!!!!!
    # Uncomment this line and test the whole process not on google collab
    # get_3di_sequences(pdb_path, output_dir) !!!!!!!!!!!!!!!!

    try:
        subprocess.run(
            f"python ./src/all/models/all_models/embed_all_models.py --model ProstT5_full --is_3Di 1 --embed_path {embed_path} --seq_path {fasta_file_3Di} --dataset other",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,  
            stderr=subprocess.PIPE
        )
        print("Embedding 3Di sequence with ProstT5_full completed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred during 3Di embedding: {e}")


def main():

    # Create the folder for embeddings if it doesn't exist
    os.makedirs('./src/cathe-predict/Embeddings', exist_ok=True)

    if args.input_type == 'AA':
        
        embed_sequence(args.model)

    elif args.input_type == 'AA+3Di':

        pdb_path = './src/cathe-predict/PDB_folder' 

        if not os.listdir(pdb_path):
            raise FileNotFoundError(f"No files found in the folder {pdb_path}. Please provide PDB files for 3Di usage")

        if args.model == 'ProtT5':
            raise ValueError("ProtT5 model does not support 3Di embeddings. Please use ProstT5 if you want the input_type to be AA+3Di.")

        embed_sequence(args.model)
        embed_3Di(pdb_path)
    else:
        # raise error here

        raise ValueError("Invalid input_type. Please choose 'AA' or 'AA+3Di'.")


if __name__ == '__main__':
    main()
