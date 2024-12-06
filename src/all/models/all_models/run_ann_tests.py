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

print(f"{green}test code running (run_ann_tests.py), make sure you set up and activated venv_2{reset}")

import itertools
from tqdm import tqdm



def run_script_with_combinations(script_path, dropout_values, layer_size_values, nb_layer_block_values, input_type_values, pLDDT_threshold_values, model_values, do_training, only_50_largest_SF, support_threshold, combinations_to_skip=None):
    """
    Function to create all possible combinations of parameters and run a script for each combination.

    :param script_path: Path to the Python script to be run.
    :param dropout_values: List of possible dropout values.
    :param layer_size_values: List of possible layer size values.
    :param nb_layer_block_values: List of possible Nb_Layer_Block values.
    :param input_type_values: List of possible Input_Type values.
    :param pLDDT_threshold_values: List of possible pLDDT_threshold values.
    :param model_values: List of model names to use.
    :param do_training: Boolean to decide if training is necessary (1), or if the goal is to evaluate a already trained model (0)
    :param only_50_largest_SF: Boolean to decide if the model should be trained on the 50 largest SF (1), or on all SF (0)
    :param combinations_to_skip: List of tuples representing parameter combinations to skip (optional).
    :param support_threshold: Integer representing the support filter value to use.
    """
    if combinations_to_skip is None:
        combinations_to_skip = []

    # Create all possible combinations of parameters
    combinations = list(itertools.product(dropout_values, layer_size_values, nb_layer_block_values, input_type_values, pLDDT_threshold_values, model_values, do_training, only_50_largest_SF, support_threshold))

    # Iterate over each combination and run the script with a progress bar
    for dropout, layer_size, nb_layer_block, input_type, pLDDT_threshold, model, do_training, only_50_largest_SF, support_threshold in tqdm(combinations, desc="Running configurations"):
        
        # Skip the specified combinations
        if (dropout, layer_size, nb_layer_block, input_type, pLDDT_threshold, model, only_50_largest_SF, support_threshold) in combinations_to_skip:
            continue

        # Create the command to execute
        command = (f"python {script_path} --classifier_input {input_type} --dropout {dropout} "
                   f"--layer_size {layer_size} --nb_layer_block {nb_layer_block} "
                   f"--pLDDT_threshold {pLDDT_threshold} --model {model} "
                   f"--do_training {do_training} "
                   f"--only_50_largest_SF {only_50_largest_SF} "
                   f"--support_threshold {support_threshold}")
        
        # Print the command (optional)
        print(f"Running: {command}")
        
        # Execute the command
        os.system(command)


script_path = './src/all/models/all_models/ann_all_models.py'


# # Initialize the progress bar
# total_runs = 3  # Number of different configurations/runs we have
# progress_bar = tqdm(total=total_runs, desc="Overall Progress")

# Run each configuration setup and update the progress bar
# 1. Dropout analysis setup
dropout_values = [0.3]
layer_size_values = [2048]
nb_layer_block_values = ['two']
input_type = ['AA+3Di']
pLDDT_threshold = [4, 14, 24, 34, 44, 54, 64, 74, 84]
model = ['ProstT5_full']
do_training = [1]
only_50_largest_SF = [0]
support_threshold = [10]

run_script_with_combinations(script_path, dropout_values, layer_size_values, nb_layer_block_values, input_type, pLDDT_threshold, model, do_training, only_50_largest_SF, support_threshold)
# progress_bar.update(1)
# print(f"{progress_bar.n} run(s) have terminated.")

# # 2. Layer size analysis setup
# dropout_values = [0, 0.7]
# layer_size_values = [1024]
# nb_layer_block_values = ['two']
# input_type = ['AA']
# pLDDT_threshold = [0]
# model = ['ProstT5_full']

# run_script_with_combinations(script_path, dropout_values, layer_size_values, nb_layer_block_values, input_type, pLDDT_threshold, model)
# progress_bar.update(1)
# print(f"{progress_bar.n} run(s) have terminated.")

# # 3. Nb layer block analysis setup
# dropout_values = [0,0.1, 0.2]
# layer_size_values = [1024]
# nb_layer_block_values = ['two']
# input_type = ['AA+3Di']
# pLDDT_threshold = [0]
# model = ['ProstT5_full']

# run_script_with_combinations(script_path, dropout_values, layer_size_values, nb_layer_block_values, input_type, pLDDT_threshold, model)
# progress_bar.update(1)
# print(f"{progress_bar.n} run(s) have terminated.")

# # Close the progress bar after all runs
# progress_bar.close()



# # Create all possible combinations of parameters
# combinations = list(itertools.product(dropout_values, layer_size_values, nb_layer_block_values, input_type, pLDDT_threshold, model))


# # Iterate over each combination and run the script with a progress bar
# for dropout, layer_size, nb_layer_block, input_type, pLDDT_threshold, model in tqdm(combinations, desc="Running configurations"):

#     # # Skip the specified combinations
#     # if (dropout, layer_size, nb_layer_block) in combinations_to_skip:
#     #     continue

#     command = f"python {script_path} --classifier_input {input_type} --dropout {dropout} --layer_size {layer_size} --nb_layer_block {nb_layer_block} --pLDDT_threshold {pLDDT_threshold} --model {model}"
#     print(f"Running: {command}")
#     os.system(command)