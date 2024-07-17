# -*- coding: utf-8 -*-
# part of the code from https://github.com/facebookresearch/esm/blob/main/README.md#bulk_fasta
# run with: python ./src/all/models/ProstT5/AA_embed_with_ESM2.py --input "./data/Dataset/csv/Test.csv" --output "./data/Dataset/embeddings/Test_ESM2.npz" --batch_size 32

import argparse
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# Check for available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def get_ESM2_model():
    """
    Load the ESM2 model and tokenizer from Hugging Face.
    """
    print("Loading ESM2 model")
    model_name = "facebook/esm2_t33_650M_UR50D"
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, tokenizer

def read_csv(seq_path):
    """
    Read sequences from a CSV file.
    """
    df = pd.read_csv(seq_path)
    sequences = df['Sequence'].tolist()
    ids = df['Unnamed: 0'].astype(int).tolist()
    return dict(zip(ids, sequences))

def replace_non_standard_aa(seq):
    """
    Replace non-standard amino acids in the sequence with 'X'.
    """
    return seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')

def compute_embeddings(sequences_dict, model, tokenizer, batch_size=32):
    """
    Compute embeddings for the given sequences using the ESM2 model in batches.
    """
    embeddings = {}
    sequence_ids = list(sequences_dict.keys())
    sequence_values = list(sequences_dict.values())
    
    for i in tqdm(range(0, len(sequence_values), batch_size), desc="Computing embeddings"):
        batch_ids = sequence_ids[i:i+batch_size]
        batch_sequences = [replace_non_standard_aa(seq) for seq in sequence_values[i:i+batch_size]]
        inputs = tokenizer(batch_sequences, return_tensors='pt', padding="longest", truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        for batch_idx, seq_id in enumerate(batch_ids):
            s_len = (inputs['attention_mask'][batch_idx] == 1).sum().item()
            emb = outputs.last_hidden_state[batch_idx, 1:s_len+1]  # Exclude padding and special tokens
            emb = emb.mean(dim=0)  # Average pooling
            embeddings[seq_id] = emb.detach().cpu().numpy().squeeze()

    return embeddings

def save_embeddings(embeddings, output_path):
    """
    Save the computed embeddings to an NPZ file.
    """
    np.savez(output_path, **embeddings)

def get_embeddings(seq_path, emb_path, batch_size=32):
    sequences = read_csv(seq_path)
    model, tokenizer = get_ESM2_model()
    embeddings = compute_embeddings(sequences, model, tokenizer, batch_size)
    save_embeddings(embeddings, emb_path)

def create_arg_parser():
    """
    Creates and returns the ArgumentParser object.
    """
    parser = argparse.ArgumentParser(description=(
        'Compute ESM2 embeddings for protein sequences.'))
    
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='Path to the input CSV file containing sequences.')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='Path to save the output NPZ file with embeddings.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing sequences.')
    return parser

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    
    seq_path = Path(args.input)  # path to input CSV
    emb_path = Path(args.output)  # path where embeddings should be stored

    get_embeddings(seq_path, emb_path, args.batch_size)

if __name__ == '__main__':
    main()
