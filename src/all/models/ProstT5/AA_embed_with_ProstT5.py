# -*- coding: utf-8 -*-
# code in large part from https://github.com/mheinzinger/ProstT5/blob/main/scripts/embed.py

# run with ```python ./src/all/models/ProstT5/AA_embed_with_ProstT5.py --input ./data/CATHe\ Dataset/csv/Test.csv --output ./data/CATHe\ Dataset/embeddings/Test_ProstT5.npz --model Rostlab/ProstT5 --per_protein 1 --half 1 --is_3Di 0```


"""
Created on Fri Jun 16 14:27:44 2023

@author: mheinzinger
"""

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
    print("Loading T5 from: {}".format(model_dir))
    model = T5EncoderModel.from_pretrained(model_dir).to(device)
    model = model.eval()
    vocab = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False)
    return model, vocab


def read_csv(seq_path):
    '''
        Reads in CSV file containing sequences.
        Returns a dictionary of sequences with IDs as keys.
    '''
    sequences = {}
    df = pd.read_csv(seq_path)
    for index, row in df.iterrows():
        sequences[str(row['Unnamed: 0'])] = row['Sequence']  # Ensure keys are strings
    return sequences


def get_embeddings(seq_path, emb_path, model_dir, per_protein, half_precision, is_3Di,
                   max_residues=4000, max_seq_len=3263, max_batch=100):
    
    emb_dict = dict()

    # Read in CSV
    seq_dict = read_csv(seq_path)
    prefix = "<fold2AA>" if is_3Di else "<AA2fold>"
    
    model, vocab = get_T5_model(model_dir)
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
        seq = prefix + ' ' + ' '.join(list(seq))
        batch.append((pdb_id, seq, seq_len))

        # count residues in current batch and add the last sequence length to
        # avoid that batches with (n_res_batch > max_residues) get processed 
        n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len 
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
            pdb_ids, seqs, seq_lens = zip(*batch)
            batch = list()

            token_encoding = vocab.batch_encode_plus(seqs, 
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
                continue
            
            # batch-size x seq_len x embedding_dim
            # extra token is added at the end of the seq
            for batch_idx, identifier in enumerate(pdb_ids):
                s_len = seq_lens[batch_idx]
                # account for prefix in offset
                emb = embedding_repr.last_hidden_state[batch_idx, 1:s_len+1]
                
                if per_protein:
                    emb = emb.mean(dim=0)
                emb_dict[identifier] = emb.detach().cpu().numpy().squeeze()
                if len(emb_dict) == 1:
                    print("Example: embedded protein {} with length {} to emb. of shape: {}".format(identifier, s_len, emb.shape))

    end = time.time()

    # sort created embedding dict
    

    # # Sort the keys in ascending order
    # sorted_keys = sorted(embeddings_dict.keys())

    # # Create a list of embeddings in the sorted order
    # sorted_embeddings = [embeddings_dict[key] for key in tqdm(sorted_keys, desc="Sorting embeddings")]

    # # Convert the list to a dictionary with string keys to save as NPZ
    # sorted_embeddings_dict = {str(key): value for key, value in tqdm(zip(sorted_keys, sorted_embeddings), desc="Creating sorted dictionary")}
    
    np.savez(emb_path, **emb_dict)

    print('\n############# STATS #############')
    print('Total number of embeddings: {}'.format(len(emb_dict)))
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format(end-start, (end-start)/len(emb_dict), avg_length))
    return True


def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=(
            'embed.py creates ProstT5-Encoder embeddings for a given text ' +
            ' file containing sequence(s) in CSV-format.' +
            'Example: python embed.py --input /path/to/some_sequences.csv --output /path/to/some_embeddings.npz --half 1 --is_3Di 0 --per_protein 1'))
    
    # Required positional argument
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='A path to a CSV-formatted text file containing protein sequence(s).')

    # Optional positional argument
    parser.add_argument('-o', '--output', required=True, type=str, 
                        help='A path for saving the created embeddings as NPZ file.')

    # Required positional argument
    parser.add_argument('--model', required=False, type=str,
                        default="Rostlab/ProstT5",
                        help='Either a path to a directory holding the checkpoint for a pre-trained model or a huggingface repository link.')

    # Optional argument
    parser.add_argument('--per_protein', type=int, 
                        default=0,
                        help="Whether to return per-residue embeddings (0: default) or the mean-pooled per-protein representation (1).")
        
    parser.add_argument('--half', type=int, 
                        default=0,
                        help="Whether to use half_precision or not. Default: 0 (full-precision)")
    
    parser.add_argument('--is_3Di', type=int, 
                        default=0,
                        help="Whether to create embeddings for 3Di or AA file. Default: 0 (generate AA-embeddings)")
    
    return parser

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    
    seq_path = Path(args.input)  # path to input CSV
    emb_path = Path(args.output)  # path where embeddings should be stored
    model_dir = args.model  # path/repo_link to checkpoint

    per_protein = False if int(args.per_protein) == 0 else True
    half_precision = False if int(args.half) == 0 else True
    is_3Di = False if int(args.is_3Di) == 0 else True

    get_embeddings(
        seq_path,
        emb_path,
        model_dir,
        per_protein=per_protein,
        half_precision=half_precision,
        is_3Di=is_3Di
    )


if __name__ == '__main__':
    main()

