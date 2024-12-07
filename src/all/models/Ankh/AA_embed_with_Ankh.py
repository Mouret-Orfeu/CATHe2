# -*- coding: utf-8 -*-
# part of the code from https://github.com/agemagician/Ankh/blob/main/README.md
# run with ```python ./src/all/models/Ankh/AA_embed_with_Ankh.py --input ./data/Dataset/csv/Val.csv --output ./data/Dataset/embeddings/Val_Ankh.npz --model large```


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
        sequences[int(row['Unnamed: 0'])] = row['Sequence']  # Ensure keys are strings
    
    return sequences

def get_embeddings(seq_path, emb_path, model_type,
                   max_residues,  max_batch, max_seq_len=3263,):
    
    emb_dict = dict()

    # Read in CSV
    seq_dict = read_csv(seq_path)
    
    model, tokenizer = get_Ankh_model(model_type)
    
    print('########################################')
    print('Total number of sequences: {}'.format(len(seq_dict)))

    avg_length = sum([len(seq) for seq in seq_dict.values()]) / len(seq_dict)
    n_long = sum([1 for seq in seq_dict.values() if len(seq) > max_seq_len])
    # sort sequences by length to trigger OOM at the beginning
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(kv[1]), reverse=True)
    
    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))
    
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
                padding="longest",  # Dynamic padding based on the longest sequence in the batch
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
                if len(emb_dict) == 1:
                    print("Example: embedded protein {} with length {} to emb. of shape: {}".format(identifier, s_len, emb.shape))

    end = time.time()

    # sort created embedding dict
    # Sort the keys in ascending order
    sorted_keys = sorted(emb_dict.keys())

    # Create a list of embeddings in the sorted order
    sorted_embeddings = [emb_dict[key] for key in tqdm(sorted_keys, desc="Sorting embeddings")]

    if len(sorted_embeddings) != len(seq_dict):
        print("Number of embeddings does not match number of sequences!")
        print('Total number of embeddings: {}'.format(len(sorted_embeddings)))
        sys.exit("Stopping execution due to mismatch.")
    
    np.savez(emb_path, sorted_embeddings)

    #DEBUG
    print("10 first keys: ",sorted_keys[:10], "\n 10 last keys: ", sorted_keys[-10:])

    print('\n############# STATS #############')
    print('Total number of embeddings: {}'.format(len(sorted_embeddings)))
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format(end-start, (end-start)/len(sorted_embeddings), avg_length))
    return True

def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=(
            'AA_embed_with_Ankh.py creates Ankh-Encoder embeddings for a given text ' +
            ' file containing sequence(s) in CSV-format.' +
            'Example: python ./src/all/models/Ankh/AA_embed_with_Ankh --input /path/to/some_sequences.csv --output /path/to/some_embeddings.npz'))

    # Required positional argument
    parser.add_argument('--model', required=False, type=str,
                        default="large",
                        help='large or base')

    return parser

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    
    model_type = args.model  # large or base Ankh model 

    if model_type not in ["large", "base"]:
        raise ValueError("Invalid model type. Choose 'large' or 'base'.")
    
    if model_type == "large":
        max_residues,  max_batch = 4096, 4096
    else:   
        max_residues,  max_batch = 8192, 8192

    # seq_path_Test = "./data/Dataset/csv/Test.csv"
    # emb_path_Test = f"./data/Dataset/embeddings/Test_Ankh_{model_type}.npz"

    # get_embeddings(
    #     seq_path_Test,
    #     emb_path_Test,
    #     model_type, 
    #     max_residues,
    #     max_batch
    
    # )

    # seq_path_Val = "./data/Dataset/csv/Val.csv"
    # emb_path_Val = f"./data/Dataset/embeddings/Val_Ankh_{model_type}.npz"

    # get_embeddings(
    #     seq_path_Val,
    #     emb_path_Val,
    #     model_type,
    #     max_residues,
    #     max_batch
    # )

    seq_path_Train = "./data/Dataset/csv/Train.csv"
    emb_path_Train = f"./data/Dataset/embeddings/Train_Ankh_{model_type}.npz"

    get_embeddings(
        seq_path_Train,
        emb_path_Train,
        model_type,
        max_residues,
        max_batch
    )
    

if __name__ == '__main__':
    main()
