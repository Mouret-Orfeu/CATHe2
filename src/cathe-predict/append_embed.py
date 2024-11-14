import numpy as np
import os

# Set the embedding path
embedding_path = './src/cathe-predict/Embeddings'

# Load the first embedding file
filename = os.path.join(embedding_path, 'ProtT5_0.0.npz')
pb_arr = np.load(filename)['arr_0']

# Loop through the files and load them if they exist
for i in range(1, 1000000):
    print(i, pb_arr.shape)
    try:
        filename = os.path.join(embedding_path, f'ProtT5_{i}.0.npz')
        arr = np.load(filename)['arr_0']
        pb_arr = np.append(pb_arr, arr, axis=0)
    except FileNotFoundError:
        print(f"File {filename} not found, stopping.")
        break

# Save the concatenated array
np.savez_compressed('./src/cathe-predict/Embeddings/Embeddings_ProtT5.npz', pb_arr)
