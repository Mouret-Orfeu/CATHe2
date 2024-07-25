import numpy as np
import pandas as pd
import h5py

def load_labels(labels_path):
    # Load labels from CSV
    labels_df = pd.read_csv(labels_path)
    return labels_df

def load_cath_domain_list(cath_path):
    # Load CATH domain list and create a mapping of CATH label to indices
    cath_mapping = {}
    with open(cath_path, 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue
            parts = line.split()
            index = parts[0]
            cath_label = '.'.join(parts[1:5])
            if cath_label not in cath_mapping:
                cath_mapping[cath_label] = []
            cath_mapping[cath_label].append(index)
    return cath_mapping

def load_embeddings(embeddings_path):
    # Load embeddings from NPZ file
    embeddings_npz = np.load(embeddings_path)
    embeddings = embeddings_npz['arr_0']  # Assuming embeddings are stored under 'arr_0'
    return embeddings

def create_embedding_dict(labels_df, cath_mapping, embeddings):
    embedding_dict = {}
    for i, row in labels_df.iterrows():
        sf_label = row['SF']
        if sf_label in cath_mapping:
            indices = cath_mapping[sf_label]
            for index in indices:
                embedding_dict[index] = embeddings[i]
    return embedding_dict

def save_embeddings_to_h5(embedding_dict, output_path):
    with h5py.File(output_path, 'w') as h5_f:
        for index, embedding in embedding_dict.items():
            h5_f.create_dataset(index, data=embedding)

def main():
    labels_path = '/data/Dataset/annotations/Y_Train_SF.csv'
    cath_path = '/data/Dataset/annotations/cath-domain-list.txt'
    embeddings_path = '/data/Dataset/embeddings/Train_ProstT5_full.npz'
    output_path = './ProstT5_full_CATHe.h5'

    labels_df = load_labels(labels_path)
    cath_mapping = load_cath_domain_list(cath_path)
    embeddings = load_embeddings(embeddings_path)
    embedding_dict = create_embedding_dict(labels_df, cath_mapping, embeddings)
    save_embeddings_to_h5(embedding_dict, output_path)

if __name__ == '__main__':
    main()
