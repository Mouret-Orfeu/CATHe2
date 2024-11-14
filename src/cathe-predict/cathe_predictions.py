import os
import argparse
import tensorflow as tf

# ANSI escape code for colored text
yellow = "\033[93m"
reset = "\033[0m"
red = "\033[91m"

# Check and set up GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{yellow}Num GPUs Available: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs{reset}")
    except RuntimeError as e:
        print(e)
else:
    print(f"{yellow}No GPUs available. Running on CPU.{reset}")

# Parse command-line arguments for the model and the input type
parser = argparse.ArgumentParser(description="Run predictions pipeline with FASTA file")
parser.add_argument('--model', type=str,default='ProtT5', choices=['ProtT5', 'ProstT5'], help="Model to use: ProtT5 (original one) or ProstT5 (new one)")
parser.add_argument('--input_type', type=str,default='AA', choices=['AA', 'AA+3Di'], help="Input type: AA or AA+3Di (AA+3Di is only supported by ProstT5). If you select AA+3Di, ensure to provide pdb files in ./src/cathe-predict/pdb_folder, from which 3Di sequences will be extracted.")
args = parser.parse_args()

# Validate the arguments
if args.model == 'ProtT5' and args.input_type == 'AA+3Di':
    raise ValueError(f"{red}Error: Model ProtT5 does not support input type AA+3Di, please select ProstT5 for AA+3Di{reset}")

if args.input_type == 'AA+3Di' and not args.pdb_path:
    raise ValueError(f"{red}Error: --pdb_path is required when input_type is AA+3Di, please provide the path to the input PDB file if you want to use AA+3Di{reset}")

# DEBUG
print(f"{yellow}Model: {args.model}")
print(f"Input Type: {args.input_type}{reset}")

# Create Embeddings directory if not already present
cmd = 'mkdir -p ./src/cathe-predict/Embeddings'
os.system(cmd)

# Converts a FASTA file containing protein sequences into a CSV dataset
cmd = f'python3 ./src/cathe-predict/fasta_to_ds.py'
os.system(cmd)

# Pass the model and input_type to predict_embed.py
cmd = f'python3 ./src/cathe-predict/predict_embed.py --model {args.model} --input_type {args.input_type}'
os.system(cmd)

if args.model == 'ProtT5':
    # Concatenates all individual embedding files into a single file
    cmd = f'python3 ./src/cathe-predict/append_embed.py'
    os.system(cmd)

# Pass the model, input_type, and pdb_path arguments to make_predictions.py
cmd = f'python3 ./src/cathe-predict/make_predictions.py --model {args.model} --input_type {args.input_type}'
os.system(cmd)
