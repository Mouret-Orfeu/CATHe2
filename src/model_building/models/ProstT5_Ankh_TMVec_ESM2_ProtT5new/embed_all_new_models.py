# This code is used to embed AA or 3Di sequences. It can do so with all new models: Ankh_base, Ankh_large, TMVec, ProstT5_full, ProstT5_half, ESM2 and ProtT5_new (which is just prot_t5_xl_uniref50, the 'new' is just a remainder that CATH datasets embeddings were already computed in the previous version of CATHe).
# (For 3Di, only ProstT5_full or ProstT5_half are available) 

# Part of the code from:
# Author: Tymor Hamamsy
# Source: https://github.com/tymor22/tm-vec/blob/master/ipynb/repo_EMBED.ipynb, https://github.com/tymor22/tm-vec/blob/master/tm_vec/tm_vec_utils.py
# License: BSD 3-Clause License, Copyright (c) 2022, Tymor Hamamsy. All rights reserved.
# License available at: https://github.com/tymor22/tm-vec/blob/master/LICENSE

# Part of the code from:
# Author: Ahmed Elnaggar 
# Source: https://github.com/agemagician/Ankh/blob/main/README.md
# License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International

# Part of the code from:
# Author: Martin Heinzinger
# Source: https://github.com/mheinzinger/ProstT5
# License: MIT License

# Part of the code from:
# Author: Meta Research 
# Source: https://github.com/facebookresearch/esm
# License: MIT License

# ANSI escape code for colored text
yellow = '\033[93m'
green = '\033[92m'
reset_color = '\033[0m'
red = '\033[91m'
orange_color = '\033[33m'

print(f'{green}embedding code running (embed_all_new_models.py){reset_color}')

print(f'{green}library imports in progress, it may take a long time (~40 min if you start from a fresh system, depending on your connexion speed){reset_color}')

# max_res_per_batch is the maximum number of residues per batch,
# Adjust to the GPU memory you have, (for a 40 GB GPU, max_res_per_batch = 40000 is close to the max you can use for a heavy model like Ankh_large)
# If max_res_per_batch is too hight you might get an error saying not all sequences were enbedded
# nb_seq_max_per_batch is the maximum number of sequences per batch, just put the same value as max_res_per_batch, it worked well for me
max_res_per_batch = 4096
nb_seq_max_per_batch = 4096

import sys
import os

# Check if a virtual environment is active
if not hasattr(sys, 'base_prefix') or sys.base_prefix == sys.prefix:
    raise EnvironmentError(f'{red}No virtual environment is activated. Please activate venv_2 to run this code. See ReadMe for more details.{reset_color}')

# Get the name of the activated virtual environment
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path is None:
    raise EnvironmentError(f'{red}Error, venv path is none. Please activate the venv_2. See ReadMe for more details.{reset_color}')

# Check if the activated virtual environment is venv_2
venv_name = os.path.basename(venv_path)
if venv_name != 'venv_2':
    raise EnvironmentError(f'{red}The activated virtual environment is {venv_name}, not venv_2. However venv_2 must be activated to run this code. See ReadMe for more details.{reset_color}')


import time
import argparse
import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer, T5Tokenizer, T5EncoderModel
import ankh
from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
import re
import gc
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# import argparse
# import time
# import torch
# import numpy as np
# import pandas as pd
# from transformers import AutoModel, AutoTokenizer, T5Tokenizer, T5EncoderModel


# import ankh
# from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
# import re
# import gc
# import os


# GPU management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using device: {}'.format(device))

# Print in orange if a GPU is used or not
gpu_status = 'GPU is being used!' if torch.cuda.is_available() else 'No GPU is available.'


print(f'{orange_color}{gpu_status}{reset_color}')

# TM_Vec functions ##########################################################################


def featurize_prottrans(sequences, model, tokenizer, device):
    '''
    Extract ProtT5 embedding for later TM_Vec embedding

    Args:
        sequences: list of sequences
        model: ProtT5 model
        tokenizer: ProtT5 tokenizer
        device: device to use for computation (CPU or GPU)
    
    
    '''


    sequences = [(' '.join(seq)) for seq in sequences]
    sequences = [re.sub(r'[UZOB]', 'X', sequence) for sequence in sequences]
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding='longest',)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    try:
        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)
    
    except RuntimeError:
                print('RuntimeError during ProtT5 embedding  (nb sequences in batch={} /n (length of sequences in the batch ={}))'.format(len(sequences), [len(seq) for seq in sequences]))
                sys.exit('Stopping execution due to RuntimeError.')


    # Takes the final hidden states, moves the tensor to the CPU and converts it to a NumPy array for easier manipulation.
    embedding = embedding.last_hidden_state.cpu().numpy()

    features = []
    for seq_num in range(len(sequences)):

        # Uses the 0-1 mask of valid tokens to determine the length of the sequence 
        seq_len = (attention_mask[seq_num] == 1).sum()

        # Slices out the embeddings corresponding to valid tokens, excluding the final special token
        seq_emd = embedding[seq_num][:seq_len - 1]

        # Appends the processed embedding to the features list.
        features.append(seq_emd)

    # Converts the NumPy embedding of the first sequence (features[0]) back into a PyTorch tensor.
    prottrans_embedding = torch.tensor(features[0])

    # Add a Batch Dimension for later processing
    prottrans_embedding = torch.unsqueeze(prottrans_embedding, 0).to(device)

    return prottrans_embedding


# Embed a protein using tm_vec (takes as input a ProtT5 embedding)
def embed_tm_vec(prottrans_embedding, model_deep, device, seq):
    '''
    Embed using TM_Vec

    Args:
        prottrans_embedding: ProtT5 embedding
        model_deep: TM_Vec model
        device: device to use for computation (CPU or GPU)
        seq: the sequence beeing embedded

    Returns:
        tm_vec_embedding: TM_Vec embedding

    '''


    padding = torch.zeros(prottrans_embedding.shape[0:2]).type(torch.BoolTensor).to(device)

    try:
        tm_vec_embedding = model_deep(prottrans_embedding, src_mask=None, src_key_padding_mask=padding)
    
    except RuntimeError:
        print('RuntimeError during TM_Vec embedding sequence {}'.format(seq))
        sys.exit('Stopping execution due to RuntimeError.')

    return tm_vec_embedding.cpu().detach().numpy()


def encode(sequences, model_deep, model, tokenizer, device):
    '''
    Run the whole process to embed AA sequences using TM_Vec

    Args:
        sequences: list of sequences
        model_deep: TM_Vec model
        model: ProtT5 model
        tokenizer: ProtT5 tokenizer
        device: device to use for computation (CPU or GPU)
    
    Returns:
        embed_all_sequences: list of embeddings

    '''
    embed_all_sequences = []
    for seq in tqdm(sequences, desc='Batch encoding'):
        protrans_sequence = featurize_prottrans([seq], model, tokenizer, device)
        if protrans_sequence is None:
            sys.exit()
        embedded_sequence = embed_tm_vec(protrans_sequence, model_deep, device, seq)
        embed_all_sequences.append(embedded_sequence)
    return np.concatenate(embed_all_sequences, axis=0)

# all_models functions ##########################################################################

def get_model(model_name):
    '''
    Load the model and tokenizer based on the model name

    Args:
        model_name: the name of the model to load (ProtT5_new, ESM2, Ankh_large, Ankh_base, ProstT5_full, ProstT5_half, TM_Vec)

    Returns:
        model_deep: TM_Vec extra model (if needed)
        model: the model
        tokenizer: the tokenizer
    
    '''

    print(f'Loading {model_name}')

    if model_name == 'ProtT5_new':
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
        model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc').to(device)
        model_deep = None


    elif model_name == 'ESM2':
        model_path = 'facebook/esm2_t33_650M_UR50D'
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
        model_path = 'Rostlab/ProstT5'
        print('Loading ProstT5 from: {}'.format(model_path))
        model = T5EncoderModel.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
        model_deep = None
        
    elif model_name == 'TM_Vec':
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)
        model = T5EncoderModel.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc').to(device)
        gc.collect()

        # TM-Vec model paths
        tm_vec_model_cpnt = './data/Dataset/weights/TM_Vec/tm_vec_cath_model.ckpt'
        tm_vec_model_config = './data/Dataset/weights/TM_Vec/tm_vec_cath_model_params.json'

        # Load the TM-Vec model
        tm_vec_model_config = trans_basic_block_Config.from_json(tm_vec_model_config)
        model_deep = trans_basic_block.load_from_checkpoint(tm_vec_model_cpnt, config=tm_vec_model_config)
        model_deep = model_deep.to(device)
        model_deep = model_deep.eval()
                
    else:
        
        sys.exit(f'Stopping execution due to model {model_name} not found. Choose from: ESM2, Ankh_large, Ankh_base, ProstT5_full, ProstT5_half, TM_Vec.')
    
    model.to(device)
    model.eval()
        
    return model_deep, model, tokenizer


def read_fasta(file):
    '''
    Reads a FASTA file and returns a list of tuples (id, header, sequence).

    Args:
        file: the FASTA file to read
       
    Returns:
        fasta_entries: list of tuples (id, header, sequence) corresponding to the FASTA entries
    
    
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


def get_sequences(seq_path, dataset, is_3Di):
    '''
    Read and return the AA or 3Di sequences from the CSV or FASTA file

    Args:
        seq_path: the path of the CSV or FASTA file
        dataset: the corresponding dataset (Val, Test, Train, all, other)
        is_3Di: 1 if the sequences are 3Di, 0 if the sequences are AA
    
    Returns:
        sequences: dictionary of sequences (key: sequence ID, value: sequence)
    
    '''

    print('Reading sequences')

    sequences = {}

    if is_3Di:

        # dataset='other' means the programme is used to embed a custom dataset, not the CATHe datasets, so no filtering is needed
        if dataset != 'other':
            # Determine the correct CSV file path based on the dataset
            usage_csv_path = f'./data/Dataset/csv/{dataset}_ids_for_3Di_usage_0.csv'

            if not os.path.exists(usage_csv_path):
                raise FileNotFoundError(f'CSV file not found: {usage_csv_path}')

            # Load the IDs that should be kept
            df_domains_for_3Di_usage = pd.read_csv(usage_csv_path)
            sequence_ids_to_use = set(df_domains_for_3Di_usage['Domain_id'])

            
            if not sequence_ids_to_use:
                print(f'{red}No sequence IDs found in the CSV file: {usage_csv_path}{reset_color}')
                raise ValueError('No sequence IDs found in the CSV file.')

        

        # Read the FASTA file and filter based on sequence_ids_to_use
        with open(seq_path, 'r') as fasta_file:
            fasta_entries = read_fasta(fasta_file)
            fasta_entries.sort(key=lambda entry: int(entry[0]))
        for entry in fasta_entries:
            if dataset != 'other':
                if int(entry[0]) in sequence_ids_to_use:
                    sequences[int(entry[0])] = entry[2]
            else:
                sequences[int(entry[0])] = entry[2]
        
        # 3Di-sequences need to be lower-case
        for key in sequences.keys():
            sequences[key] = sequences[key].lower()

        print(f'{yellow}Processing FASTA file: {seq_path}{reset_color}')

        if not sequences:
            print(f'{red}No sequences found in the FASTA file: {fasta_file}{reset_color}')
            raise ValueError('No sequences found in the FASTA file.')
        
    else:
        # If not 3Di, simply load the sequences from the CSV
        df = pd.read_csv(seq_path)
        for _, row in df.iterrows():
            sequences[int(row['Unnamed: 0'])] = row['Sequence']  
    
    return sequences



def embedding_set_up(seq_path, model_name, is_3Di, dataset):
    '''
    Set up the embedding process by loading the sequences and the model

    Args:
        seq_path: the path of the CSV or FASTA file
        model_name: the name of the model to load (ProtT5_new, ESM2, Ankh_large, Ankh_base, ProstT5_full, ProstT5_half, TM_Vec)
        is_3Di: 1 if the sequences are 3Di, 0 if the sequences are AA
        dataset: the corresponding dataset (Val, Test, Train, all, other)
    
    Returns:
        emb_dict: empty dictionary of embeddings (key: sequence ID, value: embedding)
        seq_dict: dictionary of sequences (key: sequence ID, value: sequence)

    
    
    '''


    emb_dict = dict()
    seq_dict = get_sequences(seq_path, dataset, is_3Di)
    model_deep, model, tokenizer = get_model(model_name)

    if model_name == 'ProstT5_half':
        model = model.half()
    if model_name in ['ProstT5_full', 'ProstT5_half']:
        prefix = '<fold2AA>' if is_3Di else '<AA2fold>'
        print(f'Input is 3Di: {is_3Di}')
    else:
        prefix = None

    print('########################################')
    print('Total number of sequences: {}'.format(len(seq_dict)))

    avg_length = sum([len(seq) for seq in seq_dict.values()]) / len(seq_dict)
    # sort sequences by length to trigger OOM (out of memory) at the beginning
    seq_dict = sorted(seq_dict.items(), key=lambda kv: len(kv[1]), reverse=True)

    return emb_dict, seq_dict, model_deep, model, tokenizer, avg_length, prefix
    


def get_embeddings(seq_path, emb_path, model_name, is_3Di, dataset,
                   max_residues=max_res_per_batch, nb_seq_max_in_batch=nb_seq_max_per_batch):

    '''
    Embed the sequences and save the embeddings

    Args:
        seq_path: the path of the CSV or FASTA file
        emb_path: the path where to save the embeddings
        model_name: the name of the model to load (ProtT5_new, ESM2, Ankh_large, Ankh_base, ProstT5_full, ProstT5_half, TM_Vec)
        is_3Di: 1 if the sequences are 3Di, 0 if the sequences are AA
        dataset: the corresponding dataset (Val, Test, Train, all, other)
        max_residues: maximum number of residues per batch
        nb_seq_max_in_batch: maximum number of sequences per batch
    
        
    Returns:
        True if the embeddings were computed and saved successfully
    
    
    '''
                                                                                           
    emb_dict, seq_dict, model_deep, model, tokenizer, avg_length, prefix = embedding_set_up(seq_path, model_name, is_3Di, dataset)

    if model_name == 'TM_Vec':
        start = time.time()
        batch = []
        batch_keys = []
        for seq_idx, (seq_key, seq) in enumerate(tqdm(seq_dict, desc='Embedding sequences'), 1):
            seq_len = len(seq)
            batch.append(seq)
            batch_keys.append(seq_key)

            n_res_batch = sum([len(s) for s in batch])
            if len(batch) >= nb_seq_max_in_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict):
                embedded_batch = encode(batch, model_deep, model, tokenizer, device)
                for i, seq_key in enumerate(batch_keys):
                    emb_dict[seq_key] = embedded_batch[i]
                batch = []
                batch_keys = []

    else:
        start = time.time()
        batch = list()
        processed_sequences = 0
        for seq_idx, (seq_key, seq) in enumerate(tqdm(seq_dict, desc='Embedding sequences'), 1):
            if model_name == 'ProtT5_new':
                # add a spaces between AA
                seq = ' '.join(seq)

            # replace non-standard AAs
            seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')
            seq_len = len(seq)
            if model_name in ['ProstT5_full', 'ProstT5_half']:
                seq = prefix + ' ' + ' '.join(list(seq))
            batch.append((seq_key, seq, seq_len))

            # count residues in current batch and add the last sequence length to
            # avoid that batches with (n_res_batch > max_residues) get processed 
            n_res_batch = sum([s_len for _, _, s_len in batch])
            if len(batch) >= nb_seq_max_in_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict):
                seq_keys, seqs, seq_lens = zip(*batch)
                batch = list()



                if model_name in ['Ankh_large', 'Ankh_base']:
                    # Split sequences into individual tokens
                    seqs = [list(seq) for seq in seqs]
                    
            
                token_encoding = tokenizer.batch_encode_plus(seqs, 
                                                        add_special_tokens=True, 
                                                        padding='longest', 
                                                        is_split_into_words =(model_name in ['ESM2','Ankh_base','Ankh_large']),
                                                        return_tensors='pt'
                                                        ).to(device)

                try:
                    with torch.no_grad():
                        embedding_repr = model(token_encoding.input_ids, 
                                            attention_mask=token_encoding.attention_mask)
                except RuntimeError:
                    print('RuntimeError during embedding for {} (L={})'.format(seq_key, seq_len))
                    continue
                
                # batch-size x seq_len x embedding_dim
                # extra token is added at the end of the seq
                for batch_idx, domaine_id in enumerate(seq_keys):
                    s_len = seq_lens[batch_idx]
                    # account for prefix in offset
                    emb = embedding_repr.last_hidden_state[batch_idx, 1:s_len+1]
                    
                    
                    
                    
                    emb = emb.mean(dim=0)
                    

                    emb_dict[domaine_id] = emb.detach().cpu().numpy().squeeze()
                    processed_sequences += 1
                    
                    

    end = time.time()

    # Sort the keys in ascending order
    sorted_keys = sorted(emb_dict.keys())

    keys = np.array(sorted_keys)  # Convert sorted_keys to a NumPy array
    embeddings = np.array([emb_dict[key] for key in sorted_keys])  # Create the embeddings array


    if len(embeddings) != len(seq_dict):
        print('Number of embeddings does not match number of sequences!')
        print('Total number of embeddings: {}'.format(len(embeddings)))
        raise ValueError(f'Stopping execution due to mismatch. processed_sequences: {processed_sequences}, sequence to be processed: {len(seq_dict)}')
    

    # Save the keys and embeddings separately
    np.savez(emb_path, keys=keys, embeddings=embeddings)
    
    
    print('Total number of embeddings: {}'.format(len(embeddings)))
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format(end-start, (end-start)/len(embeddings), avg_length))

    return True


def create_arg_parser():
    '''
    Creates and returns the ArgumentParser object.
    
    Args:
        None

    Returns:
        parser: the ArgumentParser object
    
    '''

    parser = argparse.ArgumentParser(description=
                        'Compute embeddings with one or all pLMs')
    
    parser.add_argument('--model', type=str, 
                        default='ProstT5_full', 
                        help='What model to use between ProtT5_new, ESM2, Ankh_large, Ankh_base, ProstT5_full, ProstT5_half, TM_Vec')
    
    
    parser.add_argument('--is_3Di', type=int,
                        default=0,
                        help='1 if you want to embed 3Di, 0 if you want to embed AA sequences. Default: 0')
    
    parser.add_argument('--seq_path', type=str,
                        default='default',
                        help='''If is_3Di==0: This argument contains the path of a personalized CSV file with sequences to embed 
                            (this CSV must have the same structure as the CATHe datasets). 
                            By default, the script will embed the sequences from the CATHe datasets.

                            If is_3Di==1: This argument contains the path of a FASTA file with 3Di sequences to embed 
                            (this FASTA must have the same structure as the ones produced by get_3Di_sequences.py).
                            ''')
    
    parser.add_argument('--embed_path', type=str,
                        default='default',
                        help='This argument contain the path where to put the computed embeddings')
    
    parser.add_argument('--dataset', type=str,
                        default='Val',
                        help='The dataset to embed (Val, Test, Train, other). other is used to embed a custom dataset, not the CATHe ones. In this last case you might want to adjust the seq_path and embed_path arguments.')
    
    return parser


def process_datasets(model_name, is_3Di, embed_path, seq_path, dataset):

    
    if is_3Di:
        if embed_path == 'default':
            embed_path = f'./data/Dataset/embeddings/{dataset}_{model_name}_per_protein_3Di.npz'
        if seq_path == 'default':  
            seq_path = f'./data/Dataset/3Di/{dataset}.fasta'
        

    else:
        if embed_path == 'default':
            embed_path = f'./data/Dataset/embeddings/{dataset}_{model_name}_per_protein.npz'
        if seq_path == 'default':
            seq_path = f'./data/Dataset/csv/{dataset}.csv'
        
    

    get_embeddings(
        seq_path,
        embed_path,
        model_name,
        is_3Di,
        dataset
    )

def main():

    parser = create_arg_parser()
    args = parser.parse_args()

    model_name = args.model
    is_3Di = False if int(args.is_3Di) == 0 else True
    embed_path = args.embed_path
    seq_path = args.seq_path

    dataset = args.dataset

    if dataset not in ['Val', 'Test', 'Train', 'all', 'other']:
        raise ValueError('The dataset should be Val, Test, Train, all or other')

    if is_3Di:
        if model_name not in ['ProstT5_full', 'ProstT5_half']:
            raise ValueError('For 3Di sequences, the model should be ProstT5_full or ProstT5_half')
        
    print(f'Embedding with {model_name}')
    process_datasets(model_name, is_3Di, embed_path, seq_path, dataset)



if __name__ == '__main__':
    main()

