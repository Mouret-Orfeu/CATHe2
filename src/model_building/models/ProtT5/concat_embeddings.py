import numpy as np

def load_npz(file_path):
    """Load an .npz file and return its contents as a dictionary."""
    return dict(np.load(file_path))

def save_npz(file_path, data):
    """Save a dictionary to an .npz file."""
    np.savez(file_path, **data)

def concat_npz_files(file_paths, output_file_path):
    """Concatenate multiple .npz files and save the result."""
    concatenated_data = {}
    
    for file_path in file_paths:
        data = load_npz(file_path)
        for key, value in data.items():
            if key in concatenated_data:
                concatenated_data[key] = np.concatenate((concatenated_data[key], value), axis=0)
            else:
                concatenated_data[key] = value
    
    save_npz(output_file_path, concatenated_data)

# Example usage for Test, Train, and Val datasets
base_path = './data/Dataset/embeddings/'
datasets = ['Test', 'Train', 'Val']

for dataset in datasets:
    file1 = f'{base_path}Other Class/Other_{dataset}.npz'
    file2 = f'{base_path}SF_{dataset}_ProtT5.npz'
    output_file = f'{base_path}{dataset}_ProtT5.npz'
    
    concat_npz_files([file1, file2], output_file)
    print(f"Concatenated files {file1} and {file2} into {output_file}")
