import os
import sys

import argparse
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from transformers import T5EncoderModel, T5Tokenizer
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Using device: {}".format(device))


def get_T5_model(model_dir):
    print("Loading ProsT5 from: {}".format(model_dir))
    model = T5EncoderModel.from_pretrained(model_dir).to(device)
    model = model.eval()
    tokenizer = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False)
    return model, tokenizer


def read_csv(seq_path):
    '''
        Reads in CSV file containing sequences.
        Returns a dictionary of sequences with IDs as keys.
    '''
    sequences = {}
    df = pd.read_csv(seq_path)
    for _, row in df.iterrows():
        sequences[int(row['Unnamed: 0'])] = row['Sequence']
    return sequences


def get_embeddings(seq_path, emb_path, model_dir, half_precision, is_3Di,
                   max_residues=4096, max_seq_len=3263, max_batch=34359738368):
    
    emb_dict = dict()

    # Read in CSV
    seq_dict = read_csv(seq_path)
    prefix = "<fold2AA>" if is_3Di else "<AA2fold>"
    
    model, tokenizer = get_T5_model(model_dir)
    if half_precision:
        model = model.half()
        print("Using model in half-precision!")

    print('########################################')
    print(f"Input is 3Di: {is_3Di}")
    print('Example sequence: {}\n{}'.format(next(iter(seq_dict.keys())), next(iter(seq_dict.values()))))
    print('########################################')
    print('Total number of sequences: {}'.format(len(seq_dict)))

    avg_length = sum([len(seq) for seq in seq_dict.values()]) / len(seq_dict)
    n_long = sum([1 for seq in seq_dict.values() if len(seq) > max_seq_len])
    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))

    # sort sequences by length to trigger OOM at the beginning
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(kv[1]), reverse=True)
    
    start = time.time()
    batch = list()
    processed_sequences = 0

    for seq_idx, (pdb_id, seq) in enumerate(tqdm(seq_dict, desc="Embedding sequences"), 1):
        # replace non-standard AAs
        seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')
        seq_len = len(seq)
        seq = prefix + ' ' + ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len 
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = tokenizer.batch_encode_plus(seqs, 
                                                     add_special_tokens=True, 
                                                     padding="longest", 
                                                     return_tensors='pt'
                                                     ).to(device)
            try:
                with torch.no_grad():
                    embedding_repr = model(token_encoding.input_ids, 
                                           attention_mask=token_encoding.attention_mask)
            except RuntimeError:
                print("RuntimeError during embedding for {} (L={})".format(pdb_id, seq_len))
                sys.exit("Stopping execution due to RuntimeError.")
                
            
            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                # account for prefix in offset
                emb = embedding_repr.last_hidden_state[batch_idx, 1:s_len+1]
                
                emb = emb.mean(dim=0)
                emb_dict[identifier] = emb.detach().cpu().numpy().squeeze()
                processed_sequences += 1

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
    
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format(end-start, (end-start)/processed_sequences, avg_length))
    return True


def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=(
            'AA_embed_with_ProstT5.py creates ProstT5-Encoder embeddings for a given text ' +
            ' file containing sequence(s).'))
    
      
    parser.add_argument('--half', type=int, 
                        default=1,
                        help="Whether to use half_precision or not. Default: 0 (full-precision)")
    
    parser.add_argument('--is_3Di', type=int,
                        default=0,
                        help="1 if you want to embed 3Di, 0 if you want to embed AA sequences. Default: 0")
    
    return parser

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    
    model_dir = "Rostlab/ProstT5"  # path/repo_link to checkpoint

    half_precision = False if int(args.half) == 0 else True
    is_3Di = False if int(args.is_3Di) == 0 else True

    if half_precision:
        model_type = "half"
    else:
        model_type = "full"

    seq_path_Test = "./data/Dataset/csv/Test.csv"
    emb_path_Test = f"./data/Dataset/embeddings/Test_ProstT5_{model_type}.npz"

    get_embeddings(
        seq_path_Test,
        emb_path_Test,
        model_dir,
        half_precision,
        is_3Di
    
    )

    seq_path_Val = "./data/Dataset/csv/Val.csv"
    emb_path_Val = f"./data/Dataset/embeddings/Val_ProstT5_{model_type}.npz"

    get_embeddings(
        seq_path_Val,
        emb_path_Val,
        model_dir,
        half_precision,
        is_3Di
    )

    seq_path_Train = "./data/Dataset/csv/Train.csv"
    emb_path_Train = f"./data/Dataset/embeddings/Train_ProsT5_{model_type}.npz"

    get_embeddings(
        seq_path_Train,
        emb_path_Train,
        model_dir,
        half_precision,
        is_3Di
    )
    


if __name__ == '__main__':
    main()
