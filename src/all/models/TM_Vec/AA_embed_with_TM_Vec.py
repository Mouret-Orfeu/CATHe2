# run with python ./src/all/models/TM_Vec/AA_embed_with_TM_Vec.py
import time
import torch
import numpy as np
import pandas as pd
from transformers import T5EncoderModel, T5Tokenizer
from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
from tqdm import tqdm
import re
import gc
import os
import sys
os.chdir('/home/ku76797/Documents/internship/Work/CATHe')



if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Using device: {}".format(device))

def load_T5_model():
    print("loading model")
    tokeniser = T5Tokenizer.from_pretrained("./data/Dataset/weights/ProtT5/prot_t5_xl_uniref50", do_lower_case=False )
    model = T5EncoderModel.from_pretrained("./data/Dataset/weights/ProtT5/prot_t5_xl_uniref50")
    gc.collect()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    
    return model, tokeniser

def read_csv(seq_path):
    '''
        Reads in CSV file containing sequences.
        Returns a dictionary of sequences with IDs as keys.
    '''
    sequences = {}
    df = pd.read_csv(seq_path)
    for _ , row in df.iterrows():
        sequences[int(row['Unnamed: 0'])] = row['Sequence']  # Ensure keys are integers
    return sequences

# Function to extract ProtTrans embedding for a sequence
def featurize_prottrans(sequences, model, tokenizer, device):
    sequences = [(" ".join(seq)) for seq in sequences]
    sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding="longest",)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    try:
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
    
    except RuntimeError:
                print("RuntimeError during ProtT5 embedding  (nb sequences in batch={} /n (length of sequences in the batch ={}))".format(len(sequences), [len(seq) for seq in sequences]))
                sys.exit("Stopping execution due to RuntimeError.")
    
    embedding = embedding.last_hidden_state.cpu().numpy()

    features = []
    for seq_num in range(len(sequences)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len - 1]
        features.append(seq_emd)

    prottrans_embedding = torch.tensor(features[0])
    prottrans_embedding = torch.unsqueeze(prottrans_embedding, 0).to(device)

    return prottrans_embedding

# Embed a protein using tm_vec (takes as input a prottrans embedding)
def embed_tm_vec(prottrans_embedding, model_deep, device, seq):
    padding = torch.zeros(prottrans_embedding.shape[0:2]).type(torch.BoolTensor).to(device)

    try:
        tm_vec_embedding = model_deep(prottrans_embedding, src_mask=None, src_key_padding_mask=padding)
    
    except RuntimeError:
        print("RuntimeError during TM_Vec embedding sequence {}".format(seq))
        sys.exit("Stopping execution due to RuntimeError.")

    return tm_vec_embedding.cpu().detach().numpy()

def encode(sequences, model_deep, model, tokenizer, device):
    embed_all_sequences = []
    for seq in tqdm(sequences, desc="Batch encoding"):
        protrans_sequence = featurize_prottrans([seq], model, tokenizer, device)
        if protrans_sequence is None:
            sys.exit()
        embedded_sequence = embed_tm_vec(protrans_sequence, model_deep, device, seq)
        embed_all_sequences.append(embedded_sequence)
    return np.concatenate(embed_all_sequences, axis=0)

def get_embeddings(seq_path, emb_path,
                   max_residues=4096, max_seq_len=3263, max_batch=64):

    emb_dict = dict()

    # Read in CSV
    sequences_dict = read_csv(seq_path)
    sequences = list(sequences_dict.values())
    # sequences = sorted(sequences_dict.items(), key=lambda x: x[0])
    sequence_keys = list(sequences_dict.keys())
    
    model, tokeniser = load_T5_model()

    # TM-Vec model paths
    tm_vec_model_cpnt = "./data/Dataset/weights/TM_Vec/tm_vec_cath_model.ckpt"
    tm_vec_model_config = "./data/Dataset/weights/TM_Vec/tm_vec_cath_model_params.json"

    # Load the TM-Vec model
    tm_vec_model_config = trans_basic_block_Config.from_json(tm_vec_model_config)
    model_deep = trans_basic_block.load_from_checkpoint(tm_vec_model_cpnt, config=tm_vec_model_config)
    model_deep = model_deep.to(device)
    model_deep = model_deep.eval()

    print('########################################')
    print('Total number of sequences: {}'.format(len(sequences)))

    avg_length = sum([len(seq) for seq in sequences]) / len(sequences)
    n_long = sum([1 for seq in sequences if len(seq) > max_seq_len])
    
    # Making (key,sequences) tuple, with sequences sorted by length to trigger OOM at the beginning
    sorted_sequences_tuple = sorted(zip(sequence_keys, sequences), key=lambda x: len(x[1]), reverse=True)
    
    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))
    
    start = time.time()

    batch = []
    batch_keys = []
    for seq_idx, (seq_key, seq) in enumerate(tqdm(sorted_sequences_tuple, desc="Embedding sequences"), 1):
        seq_len = len(seq)
        batch.append(seq)
        batch_keys.append(seq_key)

        n_res_batch = sum([len(s) for s in batch]) + seq_len
        if len(batch) >= max_batch or n_res_batch >= max_residues or seq_idx == len(sorted_sequences_tuple) or seq_len > max_seq_len:
            embedded_batch = encode(batch, model_deep, model, tokeniser, device)
            for i, seq_key in enumerate(batch_keys):
                emb_dict[seq_key] = embedded_batch[i]
            batch = []
            batch_keys = []

    end = time.time()

    # sort created embedding dict
    # Sort the keys in ascending order
    sorted_keys = sorted(emb_dict.keys())

    # Create a list of embeddings in the sorted order
    sorted_embeddings = [emb_dict[key] for key in tqdm(sorted_keys, desc="Sorting embeddings")]

    #DEBUG
    print("10 first keys: ",sorted_keys[:10], "\n 10 last keys: ", sorted_keys[-10:])

    if len(sorted_embeddings) != len(sequences):
        print("Number of embeddings does not match number of sequences!")
        print('Total number of embeddings: {}'.format(len(sorted_embeddings)))
        print('Total number of processed sequences: {}'.format(sequences))
        sys.exit("Stopping execution due to mismatch.")
    
    np.savez(emb_path, sorted_embeddings)

    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format(end-start, (end-start)/len(sorted_embeddings), avg_length))
    return True

def main():

    seq_path_Test = "./data/Dataset/csv/Test.csv"
    emb_path_Test = "./data/Dataset/embeddings/Test_TM_Vec.npz"

    get_embeddings(
        seq_path_Test,
        emb_path_Test
    
    )

    seq_path_Val = "./data/Dataset/csv/Val.csv"
    emb_path_Val = "./data/Dataset/embeddings/Val_TM_Vec.npz"

    get_embeddings(
        seq_path_Val,
        emb_path_Val
    )

    seq_path_Train = "./data/Dataset/csv/Train.csv"
    emb_path_Train = "./data/Dataset/embeddings/Train_TM_Vec.npz"

    get_embeddings(
        seq_path_Train,
        emb_path_Train
    )
    
if __name__ == '__main__':
    main()
