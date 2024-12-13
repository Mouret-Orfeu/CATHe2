# This code is used to run the CATHe classifier trainaing and evaluation script with different hyperparameter combinations. 

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

# Check if the activated virtual environment is 'venv_2'
venv_name = os.path.basename(venv_path)
if venv_name != "venv_2":
    raise EnvironmentError(f"{red}The activated virtual environment is '{venv_name}', not 'venv_2'. However venv_2 must be activated to run this code. See ReadMe for more details.{reset}")

print(f"{green}test code running (run_ann_tests.py), make sure you set up and activated venv_2{reset}")

import itertools
from tqdm import tqdm



def run_script_with_combinations(script_path, dropout_values, layer_size_values, nb_layer_block_values, input_type_values, pLDDT_threshold_values, model_values, do_training, only_50_largest_SF, support_threshold, combinations_to_skip=None):
    '''
    Function to create all possible combinations of hyperparameters and run the ann training and evaluation script for each combination.

    Args:
        script_path (str): The path to the script to run.
        dropout_values (list): List of dropout values to test.
        layer_size_values (list): List of layer size values to test.
        nb_layer_block_values (list): List of number of layer block values to test.
        input_type_values (list): List of input type values to test.
        pLDDT_threshold_values (list): List of pLDDT threshold values to test.
        model_values (list): List of model to test.
        do_training (list): List of do_training values to test.
        only_50_largest_SF (list): List of only_50_largest_SF values to test.
        support_threshold (list): List of support threshold values to test.
        combinations_to_skip (list): List of combinations to skip. Each combination should be a tuple with the right order og hyperparameters.

    Returns:
        None
    '''

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


script_path = './src/model_building/models/ProstT5_Ankh_TMVec_ESM2_ProtT5new/ann_all_new_models.py'

# example of usage: here the function will train a model with all possible combinations of the following hyperparameters:
dropout_values = [0.3]
layer_size_values = [2048]
nb_layer_block_values = [2]
input_type = ['AA+3Di']
pLDDT_threshold = [4, 14, 24, 34, 44, 54, 64, 74, 84]
model = ['ProstT5_full']
do_training = [1]
only_50_largest_SF = [0]
support_threshold = [10]

run_script_with_combinations(script_path, dropout_values, layer_size_values, nb_layer_block_values, input_type, pLDDT_threshold, model, do_training, only_50_largest_SF, support_threshold)
