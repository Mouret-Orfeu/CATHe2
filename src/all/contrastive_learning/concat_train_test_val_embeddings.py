import numpy as np

# Paths
train_npz_path = './data/Dataset/embeddings/Train_ProstT5_full.npz'
test_npz_path = './data/Dataset/embeddings/Test_ProstT5_full.npz'
val_npz_path = './data/Dataset/embeddings/Val_ProstT5_full.npz'
output_npz_path = './data/Dataset/embeddings/Train_Test_Val_ProstT5_full_concat.npz'

# Load the npz files
train_npz = np.load(train_npz_path)
test_npz = np.load(test_npz_path)
val_npz = np.load(val_npz_path)

# Extract the arrays
train_embeddings = train_npz['arr_0']
test_embeddings = test_npz['arr_0']
val_embeddings = val_npz['arr_0']

# Concatenate the arrays
concat_embeddings = np.concatenate((train_embeddings, test_embeddings, val_embeddings), axis=0)

# Save the concatenated array into a new npz file
np.savez(output_npz_path, concat_embeddings)

print(f"Concatenated NPZ file created at {output_npz_path}")
