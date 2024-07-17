import os
os.chdir('/home/ku76797/Documents/internship/Work/CATHe')

import time
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


def get_embeddings(seq_path, model_dir, half_precision, is_3Di,
                   max_residues, max_seq_len, max_batch):
    
    emb_dict = dict()

    # Read in CSV
    seq_dict = read_csv(seq_path)
    # seq_dict = dict(list(seq_dict.items())[0:3500]) # Limit to 4000 sequences for testing
    prefix = "<fold2AA>" if is_3Di else "<AA2fold>"
    
    model, tokenizer = get_T5_model(model_dir)
    if half_precision:
        model = model.half()
   
    # sort sequences by length to trigger OOM at the beginning
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(kv[1]), reverse=True)
    
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
                return None

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

def find_best_params(seq_path, model_dir, half_precision=True, is_3Di=False, max_seq_len=3263):
    max_residues_values = [2**i for i in range(12, 18, 1)]  
    max_batch_values = [2**i for i in range(7, 37, 2)] 
    
    results = []
    
    for max_residues in max_residues_values:
        for max_batch in max_batch_values:
            try:
                total_time = get_embeddings(seq_path, model_dir, half_precision, is_3Di, max_residues, max_seq_len, max_batch)
                if total_time is None:
                    results.append((max_residues, max_batch, "Runtime Error"))
                    print(f"Runtime Error for max_residues={max_residues}, max_batch={max_batch}")
                    continue  # Skip to the next iteration if total_time is None
                results.append((max_residues, max_batch, total_time))
                print(f"Tested max_residues={max_residues}, max_batch={max_batch}, time={total_time:.2f} seconds")
            except MemoryError:
                results.append((max_residues, max_batch, "Memory Error"))
                print(f"Memory Error for max_residues={max_residues}, max_batch={max_batch}")
            except Exception as e:
                results.append((max_residues, max_batch, f"Error: {e}"))
                print(f"Failed max_residues={max_residues}, max_batch={max_batch} with error: {e}")
    
    # Find the best parameters
    valid_results = [result for result in results if isinstance(result[2], (int, float))]
    if valid_results:
        best_params = min(valid_results, key=lambda x: x[2])
        print(f"Best parameters: max_residues={best_params[0]}, max_batch={best_params[1]} with time={best_params[2]:.2f} seconds")
    else:
        print("No valid parameter combinations found.")
    
    return results, best_params if valid_results else None

    
def main():
    seq_path = "./data/Dataset/csv/Val.csv"

    
    results, best_params = find_best_params(seq_path, half_precision=True, model_dir="Rostlab/ProstT5", is_3Di=False)

    print("Results: ", results,"\nBest parameters: ", best_params)
    
    # Optionally, save the results to a file for later analysis
    results_df = pd.DataFrame(results, columns=["max_residues", "max_batch", "time"])
    results_df.to_csv(f"./src/all/models/ProstT5/embedding_time_results.csv", index=False)
    
if __name__ == '__main__':
    main()