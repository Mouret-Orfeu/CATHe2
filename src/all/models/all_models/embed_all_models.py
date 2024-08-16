# -*- coding: utf-8 -*-
# part of the code from 
# run with ```python ./src/all/models/ProstT5/AA_embed_with_ESM2.py --input ./data/CATHe\ Dataset/csv/Test.csv --output ./data/CATHe\ Dataset/embeddings/Test_ESM2.npz --model large```


import argparse
import time
import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer, T5Tokenizer, T5EncoderModel
from tqdm import tqdm
import sys
import ankh
from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
import re
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using device: {}".format(device))

# TM_Vec functions ##########################################################################

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

# all_models functions ##########################################################################

def get_model(model_name):

    print(f"Loading {model_name}")

    if model_name == 'ProtT5_new':
        tokenizer = T5Tokenizer.from_pretrained("./data/Dataset/weights/ProtT5/prot_t5_xl_uniref50", do_lower_case=False )
        model = T5EncoderModel.from_pretrained("./data/Dataset/weights/ProtT5/prot_t5_xl_uniref50")
        gc.collect()

    elif model_name == 'ESM2':
        model_path = "facebook/esm2_t33_650M_UR50D"
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_deep = None

    elif model_name == 'Ankh_large':
        model, tokenizer = ankh.load_large_model()
        model_deep = None
        
    elif model_name == 'Ankh_base':
        model, tokenizer = ankh.load_base_model()
        model_deep = None


    elif model_name in ['ProstT5_full', 'ProstT5_half']:
        model_path = "Rostlab/ProstT5"
        print("Loading ProstT5 from: {}".format(model_path))
        model = T5EncoderModel.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
        model_deep = None
        
    elif model_name == 'TM_Vec':
        tokenizer = T5Tokenizer.from_pretrained("./data/Dataset/weights/ProtT5/prot_t5_xl_uniref50", do_lower_case=False)
        model = T5EncoderModel.from_pretrained("./data/Dataset/weights/ProtT5/prot_t5_xl_uniref50")
        gc.collect()

        # TM-Vec model paths
        tm_vec_model_cpnt = "./data/Dataset/weights/TM_Vec/tm_vec_cath_model.ckpt"
        tm_vec_model_config = "./data/Dataset/weights/TM_Vec/tm_vec_cath_model_params.json"

        # Load the TM-Vec model
        tm_vec_model_config = trans_basic_block_Config.from_json(tm_vec_model_config)
        model_deep = trans_basic_block.load_from_checkpoint(tm_vec_model_cpnt, config=tm_vec_model_config)
        model_deep = model_deep.to(device)
        model_deep = model_deep.eval()
                
    else:
        
        sys.exit(f"Stopping execution due to model '{model_name}' not found. Choose from: ESM2, Ankh_large, Ankh_base, ProstT5_full, ProstT5_half, TM_Vec.")
    
    model.to(device)
    model.eval()
        
    return model_deep, model, tokenizer


def read_fasta(file):
    """Reads a FASTA file and returns a list of tuples (id, header, sequence)."""
    fasta_entries = []
    header = None
    sequence = []
    for line in file:
        line = line.strip()
        if line.startswith(">"):
            if header:
                fasta_entries.append((header.split('_')[0], header, ''.join(sequence)))
            header = line[1:]
            sequence = []
        else:
            sequence.append(line)
    if header:
        fasta_entries.append((header.split('_')[0], header, ''.join(sequence)))
    return fasta_entries

def get_sequences(seq_path, dataset, is_3Di):

    print("Reading sequences")

    sequences = {}

    if is_3Di:
        # Determine the correct CSV file path based on the dataset
        usage_csv_path = f'./data/Dataset/csv/{dataset}_ids_for_3Di_usage_0.csv'

        # Load the IDs that should be kept
        df_domains_for_3Di_usage = pd.read_csv(usage_csv_path)
        sequence_ids_to_use = set(df_domains_for_3Di_usage['Domain_id'])

        # Read the FASTA file and filter based on sequence_ids_to_use
        with open(seq_path, 'r') as fasta_file:
            fasta_entries = read_fasta(fasta_file)
            fasta_entries.sort(key=lambda entry: int(entry[0]))
        for entry in fasta_entries:
            if int(entry[0]) in sequence_ids_to_use:
                sequences[int(entry[0])] = entry[2]
        
        # 3Di-sequences need to be lower-case
        for key in sequences.keys():
            sequences[key] = sequences[key].lower()
        
    else:
        # If not 3Di, simply load the sequences from the CSV
        df = pd.read_csv(seq_path)
        for _, row in df.iterrows():
            sequences[int(row['Unnamed: 0'])] = row['Sequence']  
    
    return sequences



def embedding_set_up(seq_path, model_name, is_3Di, dataset, max_seq_len=3263):
    emb_dict = dict()
    seq_dict = get_sequences(seq_path, dataset, is_3Di)
    model_deep, model, tokenizer = get_model(model_name)

    if model_name == 'ProstT5_half':
        model = model.half()
    if model_name in ['ProstT5_full', 'ProstT5_half']:
        prefix = "<fold2AA>" if is_3Di else "<AA2fold>"
        print(f"Input is 3Di: {is_3Di}")
    else:
        prefix = None

    print('########################################')
    print('Total number of sequences: {}'.format(len(seq_dict)))

    avg_length = sum([len(seq) for seq in seq_dict.values()]) / len(seq_dict)
    n_long = sum([1 for seq in seq_dict.values() if len(seq) > max_seq_len])
    # sort sequences by length to trigger OOM at the beginning
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(kv[1]), reverse=True)
    
    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))

    return emb_dict, seq_dict, model_deep, model, tokenizer, avg_length, prefix
    


def get_embeddings(seq_path, emb_path, model_name, is_3Di, dataset,
                   max_residues=4096, max_seq_len=3263, nb_seq_max_in_batch=4096):
                                                                                           
    emb_dict, seq_dict, model_deep, model, tokenizer, avg_length, prefix = embedding_set_up(seq_path, model_name, is_3Di, dataset, max_seq_len)

    if model_name == 'TM_Vec':
        start = time.time()
        batch = []
        batch_keys = []
        for seq_idx, (seq_key, seq) in enumerate(tqdm(seq_dict, desc="Embedding sequences"), 1):
            seq_len = len(seq)
            batch.append(seq)
            batch_keys.append(seq_key)

            n_res_batch = sum([len(s) for s in batch])
            if len(batch) >= nb_seq_max_in_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
                embedded_batch = encode(batch, model_deep, model, tokenizer, device)
                for i, seq_key in enumerate(batch_keys):
                    emb_dict[seq_key] = embedded_batch[i]
                batch = []
                batch_keys = []

    else:
        start = time.time()
        batch = list()
        processed_sequences = 0
        for seq_idx, (pdb_id, seq) in enumerate(tqdm(seq_dict, desc="Embedding sequences"), 1):
            if model_name == 'ProtT5_new':
                # add a spaces between AA
                seq = " ".join(seq)

            # replace non-standard AAs
            seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')
            seq_len = len(seq)
            if model_name in ['ProstT5_full', 'ProstT5_half']:
                seq = prefix + ' ' + ' '.join(list(seq))
            batch.append((pdb_id, seq, seq_len))

            # count residues in current batch and add the last sequence length to
            # avoid that batches with (n_res_batch > max_residues) get processed 
            n_res_batch = sum([s_len for _, _, s_len in batch])
            if len(batch) >= nb_seq_max_in_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict) or seq_len > max_seq_len:
                pdb_ids, seqs, seq_lens = zip(*batch)
                batch = list()



                if model_name in ['Ankh_large', 'Ankh_base']:
                    # Split sequences into individual tokens
                    seqs = [list(seq) for seq in seqs]
                    
            
                token_encoding = tokenizer.batch_encode_plus(seqs, 
                                                        add_special_tokens=True, 
                                                        padding="longest", 
                                                        is_split_into_words =(model_name in ['ESM2','Ankh_base','Ankh_large']),
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
                    processed_sequences += 1
                    
                    # DEBUG
                    # if len(emb_dict) == 1:
                    #     print("Example: embedded protein {} with length {} to emb. of shape: {}".format(identifier, s_len, emb.shape))

    end = time.time()

    # sort created embedding dict
    # Sort the keys in ascending order
    sorted_keys = sorted(emb_dict.keys())

    # Create a list of embeddings in the sorted order
    sorted_embeddings = [emb_dict[key] for key in tqdm(sorted_keys, desc="Sorting embeddings")]

    if len(sorted_embeddings) != len(seq_dict):
        print("Number of embeddings does not match number of sequences!")
        print('Total number of embeddings: {}'.format(len(sorted_embeddings)))
        raise ValueError(f"Stopping execution due to mismatch. processed_sequences: {processed_sequences}, sequence to be processed: {len(seq_dict)}")
    
    np.savez(emb_path, sorted_embeddings)

    #DEBUG
    print("10 first keys: ",sorted_keys[:10], "\n 10 last keys: ", sorted_keys[-10:])
    
    print('Total number of embeddings: {}'.format(len(sorted_embeddings)))
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format(end-start, (end-start)/len(sorted_embeddings), avg_length))

    return True


def create_arg_parser():
    """Creates and returns the ArgumentParser object."""

    parser = argparse.ArgumentParser(description=
                        'Compute embeddings with one or all pLMs')
    
    parser.add_argument('--model', type=str, 
                        default='all', 
                        help="What model to use between ProtT5_new, ESM2, Ankh_large, Ankh_base, ProstT5_full, ProstT5_half, TM_Vec, or all")
    
    
    parser.add_argument('--is_3Di', type=int,
                        default=0,
                        help="1 if you want to embed 3Di, 0 if you want to embed AA sequences. Default: 0")
    
    return parser


def process_datasets(model_name, is_3Di):
    print(f"Embedding with {model_name}")

    datasets = ["Test", "Val", "Train"]
    for dataset in datasets:
        if is_3Di:
            seq_path = f"./data/Dataset/3Di/{dataset}.fasta"
            emb_path = f"./data/Dataset/embeddings/{dataset}_{model_name}_per_protein_3Di.npz"

        else:
            seq_path = f"./data/Dataset/csv/{dataset}.csv"
            emb_path = f"./data/Dataset/embeddings/{dataset}_{model_name}_per_protein.npz"
        

        get_embeddings(
            seq_path,
            emb_path,
            model_name,
            is_3Di,
            dataset
        )

def main():

    parser = create_arg_parser()
    args = parser.parse_args()

    model_name = args.model
    is_3Di = False if int(args.is_3Di) == 0 else True

    if is_3Di:
        if model_name not in ['ProstT5_full', 'ProstT5_half']:
            raise ValueError("For 3Di sequences, the model should be 'ProstT5_full' or 'ProstT5_half'")

    if model_name == 'all':
        
        print("Embedding with all models")
        model_names = ['ProtT5_new', 'ESM2', 'Ankh_large', 'Ankh_base', 'ProstT5_full', 'ProstT5_half', 'TM_Vec']
        for model in model_names:
            process_datasets(model, is_3Di)
    else:
        
        print(f"Embedding with {model_name}")
        process_datasets(model_name, is_3Di)



if __name__ == '__main__':
    main()

