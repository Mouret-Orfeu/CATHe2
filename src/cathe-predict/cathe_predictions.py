# This code run the entire process for CATH annotation estimation based on a given model (ProtT5 or ProstT5) and input type (Amino Acid sequences in a FASTA file and 3Di sequences computed with PDB files).

# ANSI escape code for colored text
yellow = '\033[93m'
green = '\033[92m'
reset = '\033[0m'
red = '\033[91m'

import sys
import os

# Check if a virtual environment is active
if not hasattr(sys, 'base_prefix') or sys.base_prefix == sys.prefix:
    raise EnvironmentError(f'{red}No virtual environment is activated. Please activate venv_2 or venv_1 to run this code. See ReadMe for more details.{reset}')

print(f'{green}cathe_prediction running, library imports in progress, may take a long time{reset}')

import argparse
import tensorflow as tf

# Suppress TensorFlow useless logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Parse command-line arguments for the model and the input type
parser = argparse.ArgumentParser(description='Run predictions pipeline with FASTA file')
parser.add_argument('--model', type=str,default='ProtT5', choices=['ProtT5', 'ProstT5'], help='Model to use: ProtT5 (original one) or ProstT5 (new one)')
parser.add_argument('--input_type', type=str,default='AA', choices=['AA', 'AA+3Di'], help='Input type: AA or AA+3Di (AA+3Di is only supported by ProstT5). If you select AA+3Di, ensure to provide pdb files in ./src/cathe-predict/PDB_folder, from which 3Di sequences will be extracted.')
args = parser.parse_args()


# Check if a virtual environment is active
if not hasattr(sys, 'base_prefix') or sys.base_prefix == sys.prefix:
    raise EnvironmentError(f'{red}No virtual environment is activated. Please activate the right venv first, see ReadMe for more details.{reset}')

# Get the name of the activated virtual environment
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path is None:
    raise EnvironmentError(f'{red}Error, venv path is none. Please activate the right venv first, see ReadMe for more details.{reset}')

# Check if the activated virtual environment is the right one depending on the model
venv_name = os.path.basename(venv_path)
if args.model == 'ProtT5' and venv_name != 'venv_1':
    raise EnvironmentError(f'{red}The activated virtual environment is {venv_name}, not venv_1. If you want to use the ProtT5 model, venv_1 must be activated. See ReadMe for more details.{reset}')
if args.model == 'ProstT5' and venv_name != 'venv_2':
    raise EnvironmentError(f'{red}The activated virtual environment is {venv_name}, not venv_2. If you want to use the ProstT5 model, venv_2 must be activated. See ReadMe for more details.{reset}')
if venv_name != 'venv_1' and venv_name != 'venv_2':
    raise EnvironmentError(f'{red}The activated virtual environment is {venv_name}, but it should be venv_1 or venv_2. See ReadMe for more details.{reset}')


# Check and set up GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f'{yellow}Num GPUs Available: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs{reset}')
    except RuntimeError as e:
        print(e)
else:
    print(f'{yellow}No GPUs available. Running on CPU.{reset}')



# Validate the arguments
if args.model == 'ProtT5' and args.input_type == 'AA+3Di':
    raise ValueError(f'{red}Error: Model ProtT5 does not support input type AA+3Di, please select ProstT5 for AA+3Di usage{reset}')

print(f'{yellow}Model: {args.model}')
print(f'Input Type: {args.input_type}{reset}')

# Create Embeddings directory if not already present
cmd = 'mkdir -p ./src/cathe-predict/Embeddings'
os.system(cmd)

# Converts a FASTA file containing protein sequences into a CSV dataset
cmd = f'python3 ./src/cathe-predict/fasta_to_ds.py --model {args.model}'
os.system(cmd)

# Computes the embeddings used to make the CATH annotation prediction
cmd = f'python3 ./src/cathe-predict/predict_embed.py --model {args.model} --input_type {args.input_type}'
os.system(cmd)

if args.model == 'ProtT5':
    # Concatenates all individual embedding files into a single file (only useful for ProtT5)
    cmd = f'python3 ./src/cathe-predict/append_embed.py'
    os.system(cmd)

# Uses the selected model to make the prediction
cmd = f'python3 ./src/cathe-predict/make_predictions.py --model {args.model} --input_type {args.input_type}'
os.system(cmd)
