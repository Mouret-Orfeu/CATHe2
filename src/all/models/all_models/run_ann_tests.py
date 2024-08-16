import os
import itertools
from tqdm import tqdm

# Define the possible values for each parameter
dropout_values = [0.3]
layer_size_values = [2048]
nb_layer_block_values = ['three']
input_type = ['AA+3Di']
pLDDT_threshold = [0, 4, 14, 24, 34, 44, 54, 64, 74, 84]
script_path = './src/all/models/all_models/ann_all_models.py'

# Create all possible combinations of parameters
combinations = list(itertools.product(dropout_values, layer_size_values, nb_layer_block_values, input_type, pLDDT_threshold))




# Iterate over each combination and run the script with a progress bar
for dropout, layer_size, nb_layer_block, input_type, pLDDT_threshold in tqdm(combinations, desc="Running configurations"):

    # # Skip the specified combinations
    # if (dropout, layer_size, nb_layer_block) in combinations_to_skip:
    #     continue

    command = f"python {script_path} --classifier_input {input_type} --dropout {dropout} --layer_size {layer_size} --nb_layer_block {nb_layer_block} --pLDDT_threshold {pLDDT_threshold}"
    print(f"Running: {command}")
    os.system(command)