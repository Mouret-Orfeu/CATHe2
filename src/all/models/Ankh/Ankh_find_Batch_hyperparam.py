# ANSI escape code for colored text
yellow = "\033[93m"
green = "\033[92m"
reset = "\033[0m"
red = "\033[91m"


import sys
import os

# Check if a virtual environment is active
if not hasattr(sys, 'base_prefix') or sys.base_prefix == sys.prefix:
    raise EnvironmentError(f"{red}No virtual environment is activated. Please activate the right venv_2 to run this code. See ReadMe for more details.{reset}")

# Get the name of the activated virtual environment
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path is None:
    raise EnvironmentError(f"{red}Error, venv path is none. Please activate the venv_2. See ReadMe for more details.{reset}")

venv_name = os.path.basename(venv_path)
if venv_name != "venv_2":
    raise EnvironmentError(f"{red}The activated virtual environment is '{venv_name}', not 'venv_2'. However venv_2 must be activated to run this code. See ReadMe for more details.{reset}")

import argparse
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from transformers import T5Tokenizer
from tqdm import tqdm
import ankh

if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Using device: {}".format(device))

def get_Ankh_model(model_type):
    print("Loading Ankh")

    if model_type == "large":
        model, tokenizer = ankh.load_large_model()
    elif model_type == "base":
        model, tokenizer = ankh.load_base_model()
    else:
        raise ValueError("Invalid model type. Choose 'large' or 'base'.")
    
    model.to(device)
    model.eval()
    return model, tokenizer

def read_csv(seq_path):
    '''
    Reads in CSV file containing sequences.
    Returns a dictionary of sequences with IDs as keys.
    '''
    sequences = {}
    df = pd.read_csv(seq_path)

    for _, row in df.iterrows():
        sequences[str(row['Unnamed: 0'])] = row['Sequence']  # Ensure keys are strings
    
    return sequences

def get_embeddings(seq_path, model_type,
                   max_residues, max_seq_len, max_batch, model, tokenizer):
    
    emb_dict = dict()

    # Read in CSV
    seq_dict = read_csv(seq_path)
    
    # model, tokenizer = get_Ankh_model(model_type)
    
    # sort sequences by length to trigger OOM at the beginning
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(kv[1]), reverse=True)
    
    
    start = time.time()
    batch = list()
    for seq_idx, (pdb_id, seq) in enumerate(tqdm(seq_dict, desc="Embedding sequences"), 1):
        # replace non-standard AAs
        seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')
        seq_len = len(seq)
        batch.append((pdb_id, seq, seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([s_len for _, _, s_len in batch])
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            # Split sequences into individual tokens
            tokenized_seqs = [list(seq) for seq in seqs]

            token_encoding = tokenizer.batch_encode_plus(
                tokenized_seqs, 
                add_special_tokens=True, 
                padding=True,  # Dynamic padding based on the longest sequence in the batch
                is_split_into_words=True,
                return_tensors='pt'
            ).to(device)
            
            try:
                with torch.no_grad():
                    embedding_repr = model(token_encoding.input_ids, attention_mask=token_encoding.attention_mask)
            except RuntimeError as e:
                print(f"RuntimeError during embedding for {pdb_id} (L={seq_len}): {e}")
                continue
            
            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                # account for prefix in offset
                emb = embedding_repr.last_hidden_state[batch_idx, 1:s_len+1]
                emb = emb.mean(dim=0)
                emb_dict[identifier] = emb.detach().cpu().numpy().squeeze()

    end = time.time()

    if len(emb_dict) != len(seq_dict):
        return None

    total_time = end - start
    return total_time

def find_best_params(seq_path, model_type, max_seq_len=3263):
    max_residues_values = [2**i for i in range(7, 19, 1)]  
    max_batch_values = [2**i for i in range(7, 19, 1)] 

    model, tokenizer = get_Ankh_model(model_type) 
    
    results = []
    
    for max_residues in tqdm(max_residues_values, desc="Max Residues testing"):
        for max_batch in tqdm(max_batch_values, desc="Max Batch testing"):
            try:
                total_time = get_embeddings(seq_path, model_type, max_residues, max_seq_len, max_batch, model, tokenizer)
                if total_time is None:
                    results.append((max_residues, max_batch, "Runtime Error"))
                    print(f"Runtime Error for max_residues={max_residues}, max_batch={max_batch}")
                    continue
                results.append((max_residues, max_batch, total_time))
                print(f"Tested max_residues={max_residues}, max_batch={max_batch}, time={total_time:.2f} seconds")
            except MemoryError:
                results.append((max_residues, max_batch, "Memory Error"))
                print(f"Memory Error for max_residues={max_residues}, max_batch={max_batch}")
            except Exception as e:
                results.append((max_residues, max_batch, f"Error: {e}"))
                print(f"Failed max_residues={max_residues}, max_batch={max_batch} with error: {e}")
    
    valid_results = [result for result in results if isinstance(result[2], (int, float))]
    if valid_results:
        best_params = min(valid_results, key=lambda x: x[2])
        print(f"Best parameters: max_residues={best_params[0]}, max_batch={best_params[1]} with time={best_params[2]:.2f} seconds")
    else:
        best_params = None
        print("No valid parameter combinations found.")
    
    return results, best_params

def main():
    seq_path = "./data/Dataset/csv/Val.csv"  

    for model_type in ["large", "base"]:
        results, best_params = find_best_params(seq_path, model_type=model_type)
        results_df = pd.DataFrame(results, columns=["max_residues", "max_batch", "time"])
        results_df.to_csv(f"./src/all/models/Ankh/embedding_time_results_{model_type}.csv", index=False)

if __name__ == '__main__':
    main()