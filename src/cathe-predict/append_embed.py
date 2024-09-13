import numpy as np
import os

# Set the embedding path
embedding_path = './src/cathe-predict/Embeddings'

# Load the first embedding file
filename = os.path.join(embedding_path, 'T5_0.0.npz')
pb_arr = np.load(filename)['arr_0']

# Loop through the files and load them if they exist
for i in range(1, 1000000):
    print(i, pb_arr.shape)
    try:
        filename = os.path.join(embedding_path, f'T5_{i}.0.npz')
        arr = np.load(filename)['arr_0']
        pb_arr = np.append(pb_arr, arr, axis=0)
    except FileNotFoundError:
        print(f"File {filename} not found, stopping.")
        break

# Save the concatenated array in the current directory
np.savez_compressed('./Embeddings_T5.npz', pb_arr)
