import numpy as np
import h5py

def npz_to_h5_embedding_converter(npz_file_path, h5_file_path):
    # Load the .npz file
    npz_data = np.load(npz_file_path)
    
    # Create a new .h5 file
    with h5py.File(h5_file_path, 'w') as h5_file:
        for key in npz_data:
            # Save each dataset from the .npz file to the .h5 file
            h5_file.create_dataset(key, data=npz_data[key])

    print(f"Converted {npz_file_path} to {h5_file_path}")

# Example usage
npz_file_path = './data/Dataset/embeddings/Train_Test_Val_ProstT5_full_concat.npz'
h5_file_path = './data/Dataset/embeddings/Train_Test_Val_ProstT5_full_concat.h5'
npz_to_h5_embedding_converter(npz_file_path, h5_file_path)
