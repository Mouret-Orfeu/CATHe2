# This code is for extracting 3Di sequences from the CATHe Dataset, automatically downloading the corresponding PDB files and trimming them to match the sequences.

# This file includes foldseek commands found in the ProstT5 project ReadMe
# Author: Michael Heinzinger
# Source: https://github.com/mheinzinger/ProstT5?tab=readme-ov-file
# License: MIT License

# Foldseek:
# Author: Steinegger Lab 
# Source: https://github.com/steineggerlab/foldseek
# License: GNU GENERAL PUBLIC LICENSE

# Adjust to your CPU capacity
multi_threading_worker_nb = 100

# ANSI escape code for colored text
yellow = '\033[93m'
green = '\033[92m'
reset = '\033[0m'
red = '\033[91m'

print(f'{green}3Di computing code running (get_3Di_sequences.py){reset}')

import sys
import os

# Check if a virtual environment is active
if not hasattr(sys, 'base_prefix') or sys.base_prefix == sys.prefix:
    raise EnvironmentError(f'{red}No virtual environment is activated. Please activate the right venv_2 to run this code. See ReadMe for more details.{reset}')

# Get the name of the activated virtual environment
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path is None:
    raise EnvironmentError(f'{red}Error, venv path is none. Please activate the venv_2. See ReadMe for more details.{reset}')

# Check if the activated virtual environment is venv_2
venv_name = os.path.basename(venv_path)
if venv_name != 'venv_2':
    raise EnvironmentError(f'{red}The activated virtual environment is {venv_name}, not venv_2. However venv_2 must be activated to run this code. See ReadMe for more details.{reset}')


import pandas as pd
import requests
import subprocess
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

# Function to find the model with the best matching chain sequence
def find_best_model(pdb_file_path, sequence):
    '''
    Find the model with the best matching chain sequence in a pdb file, given a CATHe dataset AA sequence.

    Args:
        pdb_file_path (str): The path to the PDB file.
        sequence (str): The sequence to match.
    
    Returns:
        best_model_id (int): The model ID of the best matching model.
        best_match_chain_id (str): The chain ID of the best matching chain.
        best_pdb_sequence (str): The sequence of the best matching chain in the PDB file.
    
    '''

    parser = PDBParser()
    structure = parser.get_structure('structure', pdb_file_path)
    best_model_id = None
    best_match_score = float('inf')
    best_pdb_sequence = ''

    for model in structure:
        for chain in model:
            pdb_sequence = ''
            for residue in chain:
                if residue.id[0] == ' ':  # Ensures only standard residues are considered
                    pdb_sequence += seq1(residue.resname)
            # Calculate match score (using Hamming distance with penalty for length differences.)
            match_score = sum(1 for a, b in zip(sequence, pdb_sequence) if a != b) + abs(len(sequence) - len(pdb_sequence))
            if match_score < best_match_score:
                best_model_id = model.id
                best_match_chain_id = chain.id
                best_match_score = match_score
                best_pdb_sequence = pdb_sequence

    return best_model_id, best_match_chain_id, best_pdb_sequence 

class TrimSelect(Select):
    def __init__(self, residues):
        self.residues = residues

    def accept_residue(self, residue):
        return residue in self.residues

def extract_global_plddt(pdb_file_path):
    '''
    Extract the global pLDDT score from a PDB file.

    Args:
        pdb_file_path (str): The path to the PDB file.
    
    Returns:
        plddt_scores (list): The list of pLDDT scores corresponding to each residue of thechain considered.
    '''

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
    '''
    Save pLDDT scores to a file.

    Args:
        plddt_scores (list): The list of pLDDT scores to save.
        output_path (str): The path to save the scores to.

    Returns:
        None
    '''
    with open(output_path.replace('.pdb', '_plddt_scores.txt'), 'w') as f:
        for score in plddt_scores:
            f.write(f'{score}\n')

def plot_plddt_scores(plddt_scores, output_dir):
    '''
    Create and save a boxplot of the aggregated pLDDT scores.

    Args:
        plddt_scores (list): The list of pLDDT scores to plot.
        output_dir (str): The directory to save the plot to.
    
    Returns:
        None
    '''
    plt.boxplot(plddt_scores)
    plt.title('pLDDT Score Distribution')
    plt.ylabel('pLDDT Score')
    plt.savefig(os.path.join(output_dir, 'aggregated_plddt_boxplot.png'))
    plt.close()

def trim_pdb(pdb_file_path, sequence, best_chain_id, model_id, expected_chain_id, trimmed_pdb_file_path):
    '''
    Trim a PDB file to match a given sequence.

    Args:
        pdb_file_path (str): The path to the PDB file.
        sequence (str): The sequence to match.
        best_chain_id (str): The chain ID of the best matching chain.
        model_id (int): The model ID of the best matching model.
        expected_chain_id (str): The expected chain ID.
        trimmed_pdb_file_path (str): The path to save the trimmed PDB file to.
    
    Returns:
        None
    '''
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file_path)

   # Check if expected_chain_id exists in the model
    used_chain_id = expected_chain_id
    chain_ids = [chain.id for chain in structure[model_id]]
    if expected_chain_id not in chain_ids:
        used_chain_id = best_chain_id


    # Extract sequence from PDB file for the specified chain and model
    pdb_sequence = ''
    residues = []
    chain_model_found = False
    for model in structure:
        if model.id == model_id:
            for chain in model:
                if chain.id == used_chain_id:
                    chain_model_found = True
                    for residue in chain:
                        if is_aa(residue, standard=True):
                            pdb_sequence += seq1(residue.resname)
                            residues.append(residue)
                    break
            break

    
    #DEBUG
    # print(f'{red}get 3Di debug prints:')
    # print(f'{red}CSV sequence: {sequence}')
    # print(f'{red}pdb_sequence: {pdb_sequence}')
    # print(f'{red}used_chain_id: {used_chain_id}')
    # print(f'{red}model_id: {model_id}{reset}')
    


    if not chain_model_found:
        raise ValueError(f'Chain {used_chain_id} in model {model_id} not found in PDB file {pdb_file_path}')
    if len(pdb_sequence) == 0:
        raise ValueError(f'{red} No amino acids found in chain {used_chain_id} in model {model_id} in PDB file {pdb_file_path}')

    # Perform sequence alignment using PairwiseAligner
    aligner = PairwiseAligner()
    alignments = aligner.align(sequence, pdb_sequence)
    best_alignment = alignments[0]

    aligned_seq1 = best_alignment.aligned[0]
    aligned_seq2 = best_alignment.aligned[1]

    # Extract aligned sequences and residue indexes
    pdb_residues_to_keep = []
    trimmed_pdb_sequence = ''
    for i in range(len(aligned_seq1)):
        start1, end1 = aligned_seq1[i]
        start2, _ = aligned_seq2[i]
        for j in range(start1, end1):
            if sequence[j] == pdb_sequence[start2 + (j - start1)]:
                pdb_residues_to_keep.append(residues[start2 + (j - start1)])
                trimmed_pdb_sequence += sequence[j]

    # DEBUG
    # Print sequences for verification and write to the file
    # example_path= './src/model_building/get_3Di/pdb_sequence_examples_train.txt'

    # with open(example_path, 'a') as file:
    #     file.write(f'CSV sequence:         {sequence}\n')
    #     file.write(f'Trimmed PDB sequence: {trimmed_pdb_sequence}\n')
    #     file.write(f'Untrimmed PDB sequence: {pdb_sequence}\n\n')

    # Write out the trimmed structure
    io = PDBIO()
    io.set_structure(structure)
    io.save(trimmed_pdb_file_path, select=TrimSelect(pdb_residues_to_keep))




def download_and_trim_pdb(row, output_dir, process_training_set):
    '''
    Download the PDB file corresponding to a sequence and trim it to match the sequence.

    Args:
        row (pd.Series): The row of the dataset to process.
        output_dir (str): The directory to save the trimmed PDB file to.
        process_training_set (bool): Whether the dataset is the training set.
    
    Returns:
        dict: The sequence ID and path to the trimmed PDB file.
        pLDDT_scores (list): The list of pLDDT scores corresponding to each residue of the chain considered.
    '''
    sequence_id = row['Unnamed: 0']
    sequence = row['Sequence']
    plddt_scores = []  # Initialize plddt_scores

    if process_training_set:
        afdb_id = row['Domain']
        url = f'https://alphafold.ebi.ac.uk/files/AF-{afdb_id}-F1-model_v4.pdb'
        expected_chain = 'A'
    else:
        domain = row['Domain']
        pdb_id = domain[:4]
        expected_chain = domain[4]
        url = f'https://files.rcsb.org/download/{pdb_id}.pdb'

    try:

        # Download corresponding PDB file
        response = requests.get(url)
        response.raise_for_status()
        pdb_file_path = os.path.join(output_dir, f'{sequence_id}_{os.path.basename(url)}')

        with open(pdb_file_path, 'w') as file:
            file.write(response.text)

        # Find the best model and chain in the PDB file to match the CATH dataset sequence
        model, best_chain, _ = find_best_model(pdb_file_path, sequence)

        if process_training_set:
            plddt_scores = extract_global_plddt(pdb_file_path)
        
        # Trim the PDB file to only keep the residues that match the sequence, examples in pdb_sequence_examples.txt
        trimmed_pdb_file_path = pdb_file_path.replace('.pdb', '_trimmed.pdb')
        trim_pdb(pdb_file_path, sequence, best_chain, model, expected_chain, trimmed_pdb_file_path)
    
        os.remove(pdb_file_path)
        
        return {'sequence_id': sequence_id, 'pdb_file': trimmed_pdb_file_path}, plddt_scores
    except requests.exceptions.HTTPError as http_err:
        print(f'HTTP request failed: {http_err}')
        return {'sequence_id': sequence_id, 'pdb_file': None}, plddt_scores
    except ValueError as val_err:
        print(val_err)
        return {'sequence_id': sequence_id, 'pdb_file': None}, plddt_scores
    except Exception as err:
        print(f'Other error occurred: {err}')
        return {'sequence_id': sequence_id, 'pdb_file': None}, plddt_scores

   



#DEBUG
# Function to extract sequence from PDB file
def extract_sequence_from_pdb(pdb_file_path, chain_id, model_id):
    '''
    Debug function: extract the sequence of a chain in a PDB file.

    Args:
        pdb_file_path (str): The path to the PDB file.
        chain_id (str): The chain ID to extract.
        model_id (int): The model ID to extract.
    
    Returns:
        str: The sequence of the chain in the PDB file.
    '''

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
        raise ValueError(f'Chain {chain_id} in model {model_id} not found in PDB file {pdb_file_path}')

    return sequence

# Function to run a shell command and check for errors
def run_command(command):
    '''
    Run a shell command and check for errors.

    Args:
        command (str): The command to run.
    
    Returns:
        str: The output of the command.
    
    '''
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f'Error running command: {command}')
        print(result.stderr)
        raise Exception('Command failed')
    return result.stdout

def process_dataset(data, output_dir, query_db, query_db_ss_fasta, process_training_set, plddt_scores_list):
    os.makedirs(output_dir, exist_ok=True)

    '''
    Process a dataset with multiple threads to extract and save 3Di sequences

    Args:
        data (pd.DataFrame): The dataset to process.
        output_dir (str): The directory to save the trimmed PDB files to.
        query_db (str): The path to save the query database to.
        query_db_ss_fasta (str): The path to save the 3Di sequence FASTA file to.
        process_training_set (bool): Whether the dataset is the training set.
        plddt_scores_list (list): The list of global pLDDT scores for each 3Di seqeunce to fill.

    Returns:
        None
    
    '''

    results = []

    # Multi-threading the download and trimming process
    with ThreadPoolExecutor(max_workers=multi_threading_worker_nb) as executor:
        futures = [executor.submit(download_and_trim_pdb, row, output_dir, process_training_set) for _, row in data.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result, plddt_scores = future.result()
            results.append(result)
            if process_training_set and plddt_scores:
                plddt_scores_list.extend(plddt_scores)

    results_df = pd.DataFrame(results)
    output_csv = os.path.join(output_dir, 'directly_saved_pdb_idx.csv')
    results_df.to_csv(output_csv, index=False)

    print(f'Download completed and results saved to {output_csv}')

    foldseek_path = './foldseek/bin/foldseek' 
    
    # Foldseek uses all trimmed PDB files to create the 3Di sequence FASTA file
    try:
        run_command(f'{foldseek_path} createdb {output_dir} {query_db}')
        run_command(f'{foldseek_path} lndb {query_db}_h {query_db}_ss_h')
        run_command(f'{foldseek_path} convert2fasta {query_db}_ss {query_db_ss_fasta}')
        print(f'FASTA file created at {query_db_ss_fasta}')

    except Exception as e:
        print(f'An error occurred: {e}')



def remove_intermediate_files(output_dir):
    '''
    Remove all intermediate files in the output directory except the directly_saved_pdb_idx.csv file and the pLDDT plot.

    Args:
        output_dir (str): The directory to clean up.
    
    Returns:
        None
    
    '''
    pLDDT_plot_str = '_plddt_boxplot.png'

    # Remove all files in output_dir except 'directly_saved_pdb_idx.csv' and the pLDDT plot
    file_list = os.listdir(output_dir)
    for file_name in file_list:
        if file_name != 'directly_saved_pdb_idx.csv' and pLDDT_plot_str not in file_name:
            file_path = os.path.join(output_dir, file_name)
            os.remove(file_path)


def get_missing_domains():
    '''
    Create a dataset of missing domains from the training dataset for later processing.

    Args:
        None
    
    Returns:
        None

    '''
    Train_part_1 = './data/pdb_files/Train/Train_first/directly_saved_pdb_idx.csv'
    Train_part_2 = './data/pdb_files/Train/Train_second/directly_saved_pdb_idx.csv'
    Train_csv = './data/Dataset/csv/Train.csv'
    output_file = './data/pdb_files/Train/Train_missing_ones/missing_train_domains_id.csv'

    missing_domain_id_list = []
    with open(Train_part_1, 'r') as file:
        data = pd.read_csv(file)
        missing_domain_id_list.extend(data[data['pdb_file'].isnull()]['sequence_id'].tolist())

    with open(Train_part_2, 'r') as file:
        data = pd.read_csv(file)
        missing_domain_id_list.extend(data[data['pdb_file'].isnull()]['sequence_id'].tolist())

    missing_domain_id_list.sort()

    # Read the Train.csv file
    train_data = pd.read_csv(Train_csv)

    # Filter the rows in Train.csv that match the missing domain IDs
    missing_domains_data = train_data[train_data['Unnamed: 0'].isin(missing_domain_id_list)][['Unnamed: 0','Domain', 'Sequence']]

    # Save the filtered data to the output file
    missing_domains_data.to_csv(output_file, index=False)

    print(f'Missing domains data saved to {output_file}')

def read_fasta(file):
    '''
    Read a FASTA file and returns a list of tuples (id, header, sequence).
    
    Args:
        file (file): The file to read.
    
    Returns:
        list: A list of tuples (id, header, sequence).
    '''
    fasta_entries = []
    header = None
    sequence = []
    for line in file:
        line = line.strip()
        if line.startswith('>'):
            if header:
                fasta_entries.append((header.split('_')[0], header, ''.join(sequence)))
            header = line[1:]
            sequence = []
        else:
            sequence.append(line)
    if header:
        fasta_entries.append((header.split('_')[0], header, ''.join(sequence)))
    return fasta_entries

def write_fasta(file, fasta_entries):
    '''
    Write a list of tuples (id, header, sequence) to a FASTA file.
    
    Args:
        file (file): The file to write to.
        fasta_entries (list): A list of tuples (id, header, sequence).
    
    Returns:
        None
    
    '''
    for _, header, sequence in fasta_entries:
        file.write(f'>{header}\n')
        file.write(f'{sequence}\n')


def create_arg_parser():
    '''
    Create an ArgumentParser for the script.

    Args:
        None

    Returns:
        ArgumentParser: The ArgumentParser for the script.
    
    
    '''
    parser = argparse.ArgumentParser(description='Process dataset to extract 3Di sequences.')
    parser.add_argument('--dataset', type=str, choices=['all', 'test', 'train', 'validation', 'train_missing_ones'], default='all',
                        help='Dataset to process: all, test, train, validation or train_missing_ones. \
                        train_missing_ones is basically a re run of the train dataset to try creating missing 3Di sequences from the first run, due to failed requests for PDB files.\
                        all will process all datasets except train_missing_ones.')
    return parser

def main():

    # Parse arguments
    parser = create_arg_parser()
    args = parser.parse_args()

    dataset_map = {
    'test': './data/Dataset/csv/Test.csv',
    'validation': './data/Dataset/csv/Val.csv',
    'train': './data/Dataset/csv/Train.csv',
    'train_missing_ones': './data/pdb_files/Train/Train_missing_ones/missing_train_domains_id.csv',
    }

    if args.dataset == 'all':
        datasets_to_process = [value for key, value in dataset_map.items() if key != 'train_missing_ones']
    else:
        datasets_to_process = [dataset_map[args.dataset]]

    if len(datasets_to_process) == 1 and datasets_to_process[0] == dataset_map['train_missing_ones']:
        missing_train_domains_id_file = './data/pdb_files/Train/Train_missing_ones/missing_train_domains_id.csv'
        if not os.path.exists(missing_train_domains_id_file):
            get_missing_domains()

    for csv_file in datasets_to_process:
        
        data = pd.read_csv(csv_file)
        dataset_name = os.path.basename(csv_file).split('.')[0]

        if dataset_name == 'Train':
            # The training dataset will be processed in two parts to avoid memory issues

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

            final_fasta_path = f'./data/Dataset/3Di/{dataset_name}.fasta'
            
            with open(query_db_ss_fasta_first, 'r') as first_fasta, open(query_db_ss_fasta_second, 'r') as second_fasta:
                fasta_entries = read_fasta(first_fasta) + read_fasta(second_fasta)

            # Sort the entries by the extracted ID
            fasta_entries.sort(key=lambda x: int(x[0]))

            # Write the sorted entries to the final FASTA file
            with open(final_fasta_path, 'w') as final_fasta:
                write_fasta(final_fasta, fasta_entries)
                
            os.remove(query_db_ss_fasta_first)
            os.remove(query_db_ss_fasta_second)

            print(f'Merged FASTA file created at {final_fasta_path}')

        elif dataset_name == 'Test' or dataset_name == 'Val':
            process_training_set = False
            plddt_scores_list = []

            output_dir = f'./data/pdb_files/{dataset_name}'
            query_db = f'./data/pdb_files/{dataset_name}/{dataset_name}_queryDB'
            query_db_ss_fasta = f'./data/Dataset/3Di/{dataset_name}.fasta'

            process_dataset(data, output_dir, query_db, query_db_ss_fasta, process_training_set, plddt_scores_list)

            remove_intermediate_files(output_dir)
        
        elif dataset_name == 'missing_train_domains_id':

            process_training_set = True
            plddt_scores_list = []

            output_dir = f'./data/pdb_files/{dataset_name}'
            query_db = f'./data/pdb_files/{dataset_name}/{dataset_name}_queryDB'
            query_db_ss_fasta = f'./data/Dataset/3Di/{dataset_name}.fasta'

            process_dataset(data, output_dir, query_db, query_db_ss_fasta, process_training_set, plddt_scores_list)

            remove_intermediate_files(output_dir)

            final_fasta_path = './data/Dataset/3Di/Train.fasta'
            missing_train_fasta_path = f'./data/Dataset/3Di/{dataset_name}.fasta'

            with open(final_fasta_path, 'r') as Train_3Di, open(missing_train_fasta_path, 'r') as missing_train_3Di:
                Train_entries = read_fasta(Train_3Di)
                missing_3Di_found_during_search_rerun = read_fasta(missing_train_3Di)
                fasta_entries = Train_entries + missing_3Di_found_during_search_rerun

            # Sort the entries by the extracted ID
            fasta_entries.sort(key=lambda x: int(x[0]))

            number_of_missing_3Di_found_during_search_rerun = len(missing_3Di_found_during_search_rerun)

            # Write the sorted entries to the final FASTA file
            with open(final_fasta_path, 'w') as final_fasta:
                write_fasta(final_fasta, fasta_entries)
                
            print(f'{number_of_missing_3Di_found_during_search_rerun} Missing 3Di found during search rerun, added to Train.fasta')
                 
            os.remove(query_db_ss_fasta_first)
            os.remove(query_db_ss_fasta_second)

            print(f' {final_fasta_path}')
            

            

if __name__ == '__main__':
    main()
