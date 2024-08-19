import os
import itertools
from tqdm import tqdm

# Define the possible values for each parameter
dropout_values = [0, 0.2, 0.3]
layer_size_values = [512, 1024]
nb_layer_block_values = ['one','two', 'three']
input_type = ['3Di']
pLDDT_threshold = [0, 14, 24, 34, 44, 54, 64]
model = ['ProstT5_full']
script_path = './src/all/models/all_models/ann_all_models.py'

# Create all possible combinations of parameters
combinations = list(itertools.product(dropout_values, layer_size_values, nb_layer_block_values, input_type, pLDDT_threshold, model))




# Iterate over each combination and run the script with a progress bar
for dropout, layer_size, nb_layer_block, input_type, pLDDT_threshold, model in tqdm(combinations, desc="Running configurations"):

    # # Skip the specified combinations
    # if (dropout, layer_size, nb_layer_block) in combinations_to_skip:
    #     continue

    command = f"python {script_path} --classifier_input {input_type} --dropout {dropout} --layer_size {layer_size} --nb_layer_block {nb_layer_block} --pLDDT_threshold {pLDDT_threshold} --model {model}"
    print(f"Running: {command}")
    os.system(command)