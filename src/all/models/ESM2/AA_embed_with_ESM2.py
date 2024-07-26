# -*- coding: utf-8 -*-
# part of the code from 
# run with ```python ./src/all/models/ProstT5/AA_embed_with_ESM2.py --input ./data/CATHe\ Dataset/csv/Test.csv --output ./data/CATHe\ Dataset/embeddings/Test_ESM2.npz --model large```



import time
import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using device: {}".format(device))


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
    '''
        Reads in CSV file containing sequences.
        Returns a dictionary of sequences with IDs as keys.
    '''
    sequences = {}
    df = pd.read_csv(seq_path)

    for _, row in df.iterrows():
        sequences[int(row['Unnamed: 0'])] = row['Sequence']  # Ensure keys are strings 
    
    return sequences


def get_embeddings(seq_path, emb_path,
                   max_residues=4096, max_seq_len=3263, max_batch=4096):
    
    emb_dict = dict()

    # Read in CSV
    seq_dict = read_csv(seq_path)
    
    model, tokenizer = get_ESM2_model()
    

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

            token_encoding = tokenizer.batch_encode_plus(seqs, 
                                                     add_special_tokens=True, 
                                                     padding="longest", 
                                                     is_split_into_words=True,
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
    
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format(end-start, (end-start)/len(sorted_embeddings), avg_length))
    return True


def main():

    seq_path_Test = "./data/Dataset/csv/Test.csv"
    emb_path_Test = f"./data/Dataset/embeddings/Test_ESM2.npz"

    get_embeddings(
        seq_path_Test,
        emb_path_Test
    
    )

    seq_path_Val = "./data/Dataset/csv/Val.csv"
    emb_path_Val = f"./data/Dataset/embeddings/Val_ESM2.npz"

    get_embeddings(
        seq_path_Val,
        emb_path_Val
    )

    seq_path_Train = "./data/Dataset/csv/Train.csv"
    emb_path_Train = f"./data/Dataset/embeddings/Train_ESM2.npz"

    get_embeddings(
        seq_path_Train,
        emb_path_Train
    )


if __name__ == '__main__':
    main()

