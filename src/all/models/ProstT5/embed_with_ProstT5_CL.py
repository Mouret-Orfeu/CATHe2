# ANSI escape code for colored text
yellow = "\033[93m"
green = "\033[92m"
reset = "\033[0m"
red = "\033[91m"


import sys
import os

# Check if a virtual environment is active
if not hasattr(sys, 'base_prefix') or sys.base_prefix == sys.prefix:
    raise EnvironmentError(f"{red}No virtual environment is activated. Please activate the right venv_2 to run this code. See ReadMe for more details.{reset}")

# Get the name of the activated virtual environment
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path is None:
    raise EnvironmentError(f"{red}Error, venv path is none. Please activate the venv_2. See ReadMe for more details.{reset}")

venv_name = os.path.basename(venv_path)
if venv_name != "venv_2":
    raise EnvironmentError(f"{red}The activated virtual environment is '{venv_name}', not 'venv_2'. However venv_2 must be activated to run this code. See ReadMe for more details.{reset}")

import torch
import h5py
import numpy as np
import torch.nn as nn

class ProtTucker(nn.Module):
    def __init__(self):
        super(ProtTucker, self).__init__()

        self.protTucker = nn.Sequential(
            nn.Linear(1024, 1024),  # Adjust the input size if different
            nn.Tanh(),
            nn.Linear(1024, 1024),  # Adjust the output size if different
        )

    def single_pass(self, X):
        X = X.float()
        return self.protTucker(X)

    def forward(self, X):
        anchor = self.single_pass(X[:, 0, :])
        pos = self.single_pass(X[:, 1, :])
        neg = self.single_pass(X[:, 2, :])
        return (anchor, pos, neg)

def load_model(model_path):
    model = ProtTucker()
    with h5py.File(model_path, 'r') as f:
        for name, param in model.named_parameters():
            param.data.copy_(torch.tensor(f[name][:], requires_grad=False))
    model.eval()
    return model

def load_embeddings(file_path):
    npz_file = np.load(file_path)
    embeddings_array = npz_file['arr_0']  # Assuming the array is saved as 'arr_0'
    print(f"Loaded embeddings from {file_path}, shape: {embeddings_array.shape}")
    return embeddings_array

def compute_new_embeddings(model, embeddings, device):
    model.eval()
    with torch.no_grad():
        embeddings_tensor = torch.tensor(embeddings, device=device)
        print(f"Computing new embeddings for tensor of shape: {embeddings_tensor.shape}")
        new_embeddings = model.single_pass(embeddings_tensor).cpu().numpy()
    print(f"New embeddings computed, shape: {new_embeddings.shape}")
    return new_embeddings

def save_new_embeddings(new_embeddings, output_path):
    np.savez(output_path, arr_0=new_embeddings)
    print(f"Saved new embeddings to {output_path}")

# Load model
model_path = './saved_models/ann_ProstT5_full_CL.h5'
model = load_model(model_path)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Load existing embeddings
train_embeddings_path = './data/Dataset/embeddings/Train_ProstT5_full.npz'
val_embeddings_path = './data/Dataset/embeddings/Val_ProstT5_full.npz'
test_embeddings_path = './data/Dataset/embeddings/Test_ProstT5_full.npz'

train_embeddings = load_embeddings(train_embeddings_path)
val_embeddings = load_embeddings(val_embeddings_path)
test_embeddings = load_embeddings(test_embeddings_path)

# Compute new embeddings
new_train_embeddings = compute_new_embeddings(model, train_embeddings, device)
new_val_embeddings = compute_new_embeddings(model, val_embeddings, device)
new_test_embeddings = compute_new_embeddings(model, test_embeddings, device)

# Save new embeddings
new_train_embeddings_path = './data/Dataset/embeddings/Train_ProstT5_full_CL.npz'
new_val_embeddings_path = './data/Dataset/embeddings/Val_ProstT5_full_CL.npz'
new_test_embeddings_path = './data/Dataset/embeddings/Test_ProstT5_full_CL.npz'

save_new_embeddings(new_train_embeddings, new_train_embeddings_path)
save_new_embeddings(new_val_embeddings, new_val_embeddings_path)
save_new_embeddings(new_test_embeddings, new_test_embeddings_path)
