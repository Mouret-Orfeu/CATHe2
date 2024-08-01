import pandas as pd
import requests
import subprocess
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from concurrent.futures import ThreadPoolExecutor, as_completed
from Bio.PDB import PDBParser, PDBIO, Select, is_aa
from Bio.SeqUtils import seq1
import warnings
from Bio import BiopythonWarning
from Bio.Align import PairwiseAligner
import argparse



warnings.simplefilter('ignore', BiopythonWarning)

# Function to remove 'X' from sequences
def clean_sequence(sequence):
    return sequence.replace('X', '')

# Function to find the model with the best matching chain sequence
def find_best_model(pdb_file_path, sequence, chain_id):
    parser = PDBParser()
    structure = parser.get_structure('structure', pdb_file_path)
    best_model_id = None
    best_match_score = float('inf')
    best_pdb_sequence = ''

    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                pdb_sequence = ''
                for residue in chain:
                    if residue.id[0] == ' ':  # Ensures only standard residues are considered
                        pdb_sequence += seq1(residue.resname)
                # Calculate match score (e.g., using Hamming distance or another metric)
                match_score = sum(1 for a, b in zip(sequence, pdb_sequence) if a != b) + abs(len(sequence) - len(pdb_sequence))
                if match_score < best_match_score:
                    best_model_id = model.id
                    best_match_score = match_score
                    best_pdb_sequence = pdb_sequence

    if best_model_id is None:
        raise ValueError(f"Chain {chain_id} not found in any model of PDB file {pdb_file_path}")

    return best_model_id, best_pdb_sequence

class TrimSelect(Select):
    def __init__(self, residues):
        self.residues = residues

    def accept_residue(self, residue):
        return residue in self.residues

def extract_global_plddt(pdb_file_path):
    """
    Extract the global pLDDT score from an AlphaFold PDB file.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file_path)
    plddt_scores = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    bfactor = residue['CA'].get_bfactor()
                    plddt_scores.append(bfactor)
    return plddt_scores

def save_plddt_scores(plddt_scores, output_path):
    """
    Save pLDDT scores to a file.
    """
    with open(output_path.replace('.pdb', '_plddt_scores.txt'), 'w') as f:
        for score in plddt_scores:
            f.write(f"{score}\n")

def plot_plddt_scores(plddt_scores, output_dir):
    """
    Create and save a boxplot of the aggregated pLDDT scores.
    """
    plt.boxplot(plddt_scores)
    plt.title('pLDDT Score Distribution')
    plt.ylabel('pLDDT Score')
    plt.savefig(os.path.join(output_dir, 'aggregated_plddt_boxplot.png'))
    plt.close()

def trim_pdb(pdb_file_path, sequence, chain_id, model_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file_path)

    # Extract sequence from PDB file for the specified chain and model
    pdb_sequence = ''
    residues = []
    for model in structure:
        if model.id == model_id:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        if is_aa(residue, standard=True):
                            pdb_sequence += seq1(residue.resname)
                            residues.append(residue)
                    break
            break

    if not pdb_sequence:
        raise ValueError(f"Chain {chain_id} in model {model_id} not found in PDB file {pdb_file_path}")

    # Perform sequence alignment using PairwiseAligner
    aligner = PairwiseAligner()
    alignments = aligner.align(sequence, pdb_sequence)
    best_alignment = alignments[0]

    aligned_seq1 = best_alignment.aligned[0]
    aligned_seq2 = best_alignment.aligned[1]

    # Extract aligned sequences and residue indexes
    pdb_residues_to_keep = []
    seq_index = 0
    trimmed_pdb_sequence = ''
    for i in range(len(aligned_seq1)):
        start1, end1 = aligned_seq1[i]
        start2, end2 = aligned_seq2[i]
        for j in range(start1, end1):
            if sequence[j] == pdb_sequence[start2 + (j - start1)]:
                pdb_residues_to_keep.append(residues[start2 + (j - start1)])
                trimmed_pdb_sequence += sequence[j]

    # DEBUG
    # Print sequences for verification and write to the file
    # example_path= './src/all/get_3Di/pdb_sequence_examples_train.txt'

    # with open(example_path, 'a') as file:
    #     file.write(f"CSV sequence:         {sequence}\n")
    #     file.write(f"Trimmed PDB sequence: {trimmed_pdb_sequence}\n")
    #     file.write(f"Untrimmed PDB sequence: {pdb_sequence}\n\n")

    # Write out the trimmed structure
    io = PDBIO()
    io.set_structure(structure)
    trimmed_pdb_file_path = pdb_file_path.replace('.pdb', '_trimmed.pdb')
    io.save(trimmed_pdb_file_path, select=TrimSelect(pdb_residues_to_keep))

    return trimmed_pdb_file_path



def download_and_trim_pdb(row, output_dir, process_training_set):
    sequence_id = row['Unnamed: 0']
    sequence = row['Sequence']
    plddt_scores = []  # Initialize plddt_scores

    if process_training_set:
        afdb_id = row['Domain']
        url = f"https://alphafold.ebi.ac.uk/files/AF-{afdb_id}-F1-model_v4.pdb"
        chain = 'A'
    else:
        domain = row['Domain']
        pdb_id = domain[:4]
        chain = domain[4]
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    try:
        response = requests.get(url)
        response.raise_for_status()
        pdb_file_path = os.path.join(output_dir, f"{sequence_id}_{os.path.basename(url)}")
        with open(pdb_file_path, 'w') as file:
            file.write(response.text)
        
        if process_training_set:
            plddt_scores = extract_global_plddt(pdb_file_path)
        
        best_model_id = 0  # Default to the first model if there's no function to find the best model
        trimmed_pdb_file_path = trim_pdb(pdb_file_path, sequence, chain, best_model_id)
        
        os.remove(pdb_file_path)
        
        return {"sequence_id": sequence_id, "pdb_file": trimmed_pdb_file_path}, plddt_scores
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP request failed: {http_err}")
        return {"sequence_id": sequence_id, "pdb_file": None}, plddt_scores
    except ValueError as val_err:
        print(val_err)
        return {"sequence_id": sequence_id, "pdb_file": None}, plddt_scores
    except Exception as err:
        print(f"Other error occurred: {err}")
        return {"sequence_id": sequence_id, "pdb_file": None}, plddt_scores


# Function to extract sequence from PDB file
def extract_sequence_from_pdb(pdb_file_path, chain_id, model_id):
    parser = PDBParser()
    structure = parser.get_structure('structure', pdb_file_path)
    sequence = ''
    chain_model_found = False
    for model in structure:
        if model.id == model_id:  # Check if model matches
            for chain in model:
                if chain.id == chain_id:  # Check if chain matches
                    chain_model_found = True
                    for residue in chain:
                        sequence += seq1(residue.resname)

    if not chain_model_found:
        raise ValueError(f"Chain {chain_id} in model {model_id} not found in PDB file {pdb_file_path}")

    return sequence

# Function to run a shell command and check for errors
def run_command(command):
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        print(result.stderr)
        raise Exception("Command failed")
    return result.stdout

# Create an ArgumentParser
def create_arg_parser():
    parser = argparse.ArgumentParser(description='Process dataset to extract 3Di sequences.')
    parser.add_argument('--dataset', type=str, choices=['all', 'test', 'train', 'validation'], default='all',
                        help="Dataset to process: 'all', 'test', 'train', 'validation'")
    return parser

def process_dataset(data, output_dir, query_db, query_db_ss_fasta, process_training_set, plddt_scores_list):
    os.makedirs(output_dir, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(download_and_trim_pdb, row, output_dir, process_training_set) for _, row in data.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result, plddt_scores = future.result()
            results.append(result)
            if process_training_set and plddt_scores:
                plddt_scores_list.extend(plddt_scores)

    results_df = pd.DataFrame(results)
    output_csv = os.path.join(output_dir, 'directly_saved_pdb_idx.csv')
    results_df.to_csv(output_csv, index=False)

    print(f"Download completed and results saved to {output_csv}")

    try:
        run_command(f"foldseek createdb {output_dir} {query_db}")
        run_command(f"foldseek lndb {query_db}_h {query_db}_ss_h")
        run_command(f"foldseek convert2fasta {query_db}_ss {query_db_ss_fasta}")
        print(f"FASTA file created at {query_db_ss_fasta}")

    except Exception as e:
        print(f"An error occurred: {e}")



def remove_intermediate_files(output_dir):
    pLDDT_plot_str = '_plddt_boxplot.png'

    # Remove all files in output_dir except 'directly_saved_pdb_idx.csv' and the pLDDT plot
    file_list = os.listdir(output_dir)
    for file_name in file_list:
        if file_name != 'directly_saved_pdb_idx.csv' and pLDDT_plot_str not in file_name:
            file_path = os.path.join(output_dir, file_name)
            os.remove(file_path)

def main():
    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    dataset_map = {
        'test': './data/Dataset/csv/Test.csv',
        'validation': './data/Dataset/csv/Val.csv',
        'train': './data/Dataset/csv/Train.csv'
    }

    datasets_to_process = dataset_map.values() if args.dataset == 'all' else [dataset_map[args.dataset]]

    for csv_file in datasets_to_process:
        #DEBUG
        data = pd.read_csv(csv_file, nrows=10)

        
        # data = pd.read_csv(csv_file)
        dataset_name = os.path.basename(csv_file).split('.')[0]

        if dataset_name == 'Train':
            process_training_set = True
            plddt_scores_list = []

            half_index = len(data) // 2
            data_first_half = data.iloc[:half_index]
            data_second_half = data.iloc[half_index:]

            output_dir_first = f'./data/pdb_files/{dataset_name}/{dataset_name}_first'
            query_db_first = f'./data/pdb_files/{dataset_name}/{dataset_name}_first/{dataset_name}_first_queryDB'
            query_db_ss_fasta_first = f'./data/Dataset/3Di/{dataset_name}_first.fasta'

            output_dir_second = f'./data/pdb_files/{dataset_name}/{dataset_name}_second'
            query_db_second = f'./data/pdb_files/{dataset_name}/{dataset_name}_second/{dataset_name}_second_queryDB'
            query_db_ss_fasta_second = f'./data/Dataset/3Di/{dataset_name}_second.fasta'

            process_dataset(data_first_half, output_dir_first, query_db_first, query_db_ss_fasta_first, process_training_set, plddt_scores_list)
            remove_intermediate_files(output_dir_first)

            process_dataset(data_second_half, output_dir_second, query_db_second, query_db_ss_fasta_second, process_training_set, plddt_scores_list)
            remove_intermediate_files(output_dir_second)

            # Plot and save aggregated pLDDT scores
            plot_plddt_scores(plddt_scores_list, f'./data/pdb_files/{dataset_name}')


            final_fasta_path = f'./data/Dataset/3Di/{dataset_name}.fasta'
            with open(final_fasta_path, 'w') as final_fasta:
                with open(query_db_ss_fasta_first, 'r') as first_fasta:
                    final_fasta.write(first_fasta.read())
                with open(query_db_ss_fasta_second, 'r') as second_fasta:
                    final_fasta.write(second_fasta.read())

            os.remove(query_db_ss_fasta_first)
            os.remove(query_db_ss_fasta_second)

            print(f"Merged FASTA file created at {final_fasta_path}")

        else:
            process_training_set = False
            plddt_scores_list = []

            output_dir = f'./data/pdb_files/{dataset_name}'
            query_db = f'./data/pdb_files/{dataset_name}/{dataset_name}_queryDB'
            query_db_ss_fasta = f'./data/Dataset/3Di/{dataset_name}.fasta'

            process_dataset(data, output_dir, query_db, query_db_ss_fasta, process_training_set, plddt_scores_list)

            remove_intermediate_files(output_dir)

            

if __name__ == '__main__':
    main()
