import numpy as np
import pandas as pd

Dataset = "Val"

# Load the .npz file
embeddings_3Di = np.load(f"./data/Dataset/embeddings/{Dataset}_ProstT5_full_per_protein_3Di.npz")
embeddings_seq = np.load(f"./data/Dataset/embeddings/{Dataset}_ProstT5_full_per_protein.npz")

treshold_list = [0, 4, 14, 24, 34, 44, 54, 64, 74, 84]

filter_csv_thresh_0 = f'./data/Dataset/csv/{Dataset}_ids_for_3Di_usage_0.csv'
filter_csv_thresh_4 = f'./data/Dataset/csv/{Dataset}_ids_for_3Di_usage_4.csv'
filter_csv_thresh_14 = f'./data/Dataset/csv/{Dataset}_ids_for_3Di_usage_14.csv'
filter_csv_thresh_24 = f'./data/Dataset/csv/{Dataset}_ids_for_3Di_usage_24.csv'
filter_csv_thresh_34 = f'./data/Dataset/csv/{Dataset}_ids_for_3Di_usage_34.csv'
filter_csv_thresh_44 = f'./data/Dataset/csv/{Dataset}_ids_for_3Di_usage_44.csv'
filter_csv_thresh_54 = f'./data/Dataset/csv/{Dataset}_ids_for_3Di_usage_54.csv'
filter_csv_thresh_64 = f'./data/Dataset/csv/{Dataset}_ids_for_3Di_usage_64.csv'
filter_csv_thresh_74 = f'./data/Dataset/csv/{Dataset}_ids_for_3Di_usage_74.csv'
filter_csv_thresh_84 = f'./data/Dataset/csv/{Dataset}_ids_for_3Di_usage_84.csv'

# Extract keys and embeddings
embedding_keys_3Di = embeddings_3Di['keys']
embeddings_3Di= embeddings_3Di['embeddings']

embedding_keys_seq = embeddings_seq['keys']
embeddings_seq = embeddings_seq['embeddings']

embedding_dict_3Di = dict(zip(embedding_keys_3Di, embeddings_3Di))
embedding_dict_seq = dict(zip(embedding_keys_seq, embeddings_seq))

# print(f"embedding_keys first 10: {embedding_keys[:10]}")
# print(f"embedding_keys last 10: {embedding_keys[-10:]}")



# Load the IDs that should be kept
df_domains_for_3Di_usage = pd.read_csv(filter_csv_thresh_0)
ids_to_keep = df_domains_for_3Di_usage['Domain_id'].values

print("len ids_to_keep: embedding_keys", len(ids_to_keep))
print("len embedding_keys: ", len(embedding_keys))

if len(ids_to_keep) != len(embedding_keys):
    print("Length of ids_to_keep and embedding_keys do not match")
    exit()


print("Compare id embeddings 3Di")
# Use enumerate to get both the index and the key
for k, key in enumerate(embedding_keys):
    if key != ids_to_keep[k]:
        print(f"Key {key} does not match ids_to_keep[{k}] {ids_to_keep[k]}")
        exit()

print("All keys match!")

print("\n")
print("Compare filtered id embeddings seq, for each pLDDT threshold")

for tresh in treshold_list:
    if tresh == 0:
        filter_csv = filter_csv_thresh_0
    elif tresh == 4:
        filter_csv = filter_csv_thresh_4
    elif tresh == 14:
        filter_csv = filter_csv_thresh_14
    elif tresh == 24:
        filter_csv = filter_csv_thresh_24
    elif tresh == 34:
        filter_csv = filter_csv_thresh_34
    elif tresh == 44:
        filter_csv = filter_csv_thresh_44
    elif tresh == 54:
        filter_csv = filter_csv_thresh_54
    elif tresh == 64:
        filter_csv = filter_csv_thresh_64
    elif tresh == 74:
        filter_csv = filter_csv_thresh_74
    elif tresh == 84:
        filter_csv = filter_csv_thresh_84

    df_domains_for_3Di_usage = pd.read_csv(filter_csv)
    ids_to_keep = df_domains_for_3Di_usage['Domain_id'].values

    embedding_keys_set = set(embedding_keys)

    # Verify all ids_to_keep are in embedding_keys
    for domain_id in ids_to_keep:
        if domain_id not in embedding_keys_set:
            print(f"Domain ID {domain_id} not found in embedding_keys")
            exit()
    
    filtered_embedding_keys = [key for key in embedding_keys if key in ids_to_keep]

    # print("len ids_to_keep: embedding_keys", len(ids_to_keep))
    # print("len embedding_keys: ", len(embedding_keys))

    if len(ids_to_keep) != len(filtered_embedding_keys):
        print("Length of ids_to_keep and embedding_keys do not match")
        exit()

    print(f"Compare filtered id embeddings seq, for each pLDDT threshold {tresh}")
    # Use enumerate to get both the index and the key
    for k, key in enumerate(filtered_embedding_keys):
        if key != ids_to_keep[k]:
            print(f"Key {key} does not match ids_to_keep[{k}] {ids_to_keep[k]}")
            exit()

    print("All keys match!")
    print("\n")
