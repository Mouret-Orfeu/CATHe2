import argparse
import time
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from transformers import T5EncoderModel, T5Tokenizer
from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
from tqdm import tqdm
import re
import gc
from multiprocessing import Pool, cpu_count
import h5py

if torch.cuda.is_available():
    device = torch.device('cuda:0')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print("Using device: {}".format(device))

def load_T5_model():
    print("loading model")
    vocab = T5Tokenizer.from_pretrained("./data/Dataset/weights/ProtT5/prot_t5_xl_uniref50", do_lower_case=False)
    model = T5EncoderModel.from_pretrained("./data/Dataset/weights/ProtT5/prot_t5_xl_uniref50")
    gc.collect()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model = model.eval()
    
    return model, vocab

def read_csv(seq_path):
    df = pd.read_csv(seq_path)
    sequences = df['Sequence'].tolist()
    return sequences

def pad_sequences(sequences, max_length=None, padding_value=0.0):
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    embedding_dim = sequences[0].shape[1]
    padded_sequences = np.full((len(sequences), max_length, embedding_dim), padding_value, dtype=np.float32)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq), :] = seq
    return padded_sequences

def featurize_prottrans(sequences, model, tokenizer, device):
    sequences = [(" ".join(seq)) for seq in sequences]
    sequences = [re.sub(r"[UZOB]", "X", seq) for seq in sequences]
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)

    embedding = embedding.last_hidden_state.cpu().numpy()

    features = []
    for seq_num in range(len(sequences)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len - 1]
        features.append(seq_emd)

    padded_features = pad_sequences(features)
    prottrans_embedding = torch.tensor(padded_features).to(device)

    return prottrans_embedding

def embed_tm_vec(prottrans_embedding, model_deep, device):
    padding = torch.zeros(prottrans_embedding.shape[0:2]).type(torch.BoolTensor).to(device)
    tm_vec_embedding = model_deep(prottrans_embedding, src_mask=None, src_key_padding_mask=padding)
    return tm_vec_embedding.cpu().detach().numpy()

def encode_gen(sequences, model_deep, model, tokenizer, device):
    i = 0
    while i < len(sequences):
        protrans_sequence = featurize_prottrans(sequences[i:i+1], model, tokenizer, device)
        embedded_sequence = embed_tm_vec(protrans_sequence, model_deep, device)
        i = i + 1
        yield embedded_sequence

def save_embeds(names, embeds, sequences, file_name):
    names, embeds, sequences = iter(names), iter(embeds), iter(sequences)
    with h5py.File(file_name, 'a') as h5file:
        if "embedding" in h5file:
            seq_embed = h5file["embedding"]
            n = seq_embed.shape[0]
            for name, embed, seq in zip(names, embeds, sequences):
                if isinstance(embed, torch.Tensor):
                    embed = embed.cpu().detach().numpy()
                seq_embed.resize(n + 1, axis=0)
                seq_embed[n] = embed
                seq_embed.attrs['sequence'] = seq_embed.attrs['sequence'] + [seq]
                seq_embed.attrs['id'] = seq_embed.attrs['id'] + [name]
                n += 1
        else:
            name, embed, seq = next(names), next(embeds), next(sequences)
            if isinstance(embed, torch.Tensor):
                embed = embed.cpu().detach().numpy()
            embed = embed.reshape(1, embed.shape[0], embed.shape[1])
            seq_embed = h5file.create_dataset(
                "embedding", data=embed,
                maxshape=(None, embed.shape[1], embed.shape[2]),
                chunks=True)
            seq_embed.attrs['sequence'] = [seq]
            seq_embed.attrs['id'] = [name]
            n = 1
            for name, embed, seq in zip(names, embeds, sequences):
                n += 1
                if isinstance(embed, torch.Tensor):
                    embed = embed.cpu().detach().numpy()
                seq_embed.resize(n, axis=0)
                seq_embed[-1] = embed
                seq_embed.attrs['sequence'] = seq_embed.attrs['sequence'] + [seq]
                seq_embed.attrs['id'] = seq_embed.attrs['id'] + [name]

def get_embeddings(seq_path, emb_path, batch_size=16, max_residues=100000, max_seq_len=3263):
    sequences = read_csv(seq_path)
    
    model, vocab = load_T5_model()

    tm_vec_model_cpnt = "./data/Dataset/weights/TM_Vec/tm_vec_cath_model.ckpt"
    tm_vec_model_config = "./data/Dataset/weights/TM_Vec/tm_vec_cath_model_params.json"

    tm_vec_model_config = trans_basic_block_Config.from_json(tm_vec_model_config)
    model_deep = trans_basic_block.load_from_checkpoint(tm_vec_model_cpnt, config=tm_vec_model_config)
    model_deep = model_deep.to(device)
    model_deep = model_deep.eval()

    print('########################################')
    print('Total number of sequences: {}'.format(len(sequences)))

    avg_length = sum([len(seq) for seq in sequences]) / len(sequences)
    n_long = sum([1 for seq in sequences if len(seq) > max_seq_len])
    sequences = sorted(sequences, key=lambda x: len(x), reverse=True)
    
    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))
    
    start = time.time()

    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing batches"):
        batch_sequences = sequences[i:i + batch_size]
        batch_headers = [f"seq_{i+j}" for j in range(len(batch_sequences))]
        query_gen = encode_gen(batch_sequences, model_deep, model, vocab, device)
        save_embeds(batch_headers, query_gen, batch_sequences, emb_path)

    end = time.time()
    print('\n############# STATS #############')
    print('Total time: {:.2f}[s]; time/prot: {:.4f}[s]; avg. len= {:.2f}'.format(end-start, (end-start)/len(sequences), avg_length))
    return True

def create_arg_parser():
    parser = argparse.ArgumentParser(description=(
            'AA_embed_with_TM_Vec creates TM_Vec-Encoder embeddings for a given text ' +
            ' file containing sequence(s) in CSV-format.' +
            'Example: python ./src/all/models/TM_Vec/AA_embed_with_TM_Vec.py --input /path/to/some_sequences.csv --output /path/to/some_embeddings.npz'))
    
    parser.add_argument('-i', '--input', required=True, type=str, help='A path to a CSV-formatted text file containing protein sequence(s).')
    parser.add_argument('-o', '--output', required=True, type=str, help='A path for saving the created embeddings as HDF5 file.')
    
    return parser

def main():
    parser = create_arg_parser()
    args = parser.parse_args()
    
    seq_path = Path(args.input)
    emb_path = Path(args.output)

    get_embeddings(seq_path, emb_path)

if __name__ == '__main__':
    main()
