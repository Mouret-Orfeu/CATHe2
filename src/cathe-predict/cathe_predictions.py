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

# Parse command-line arguments for the FASTA file path, model, input type, and pdb_path
parser = argparse.ArgumentParser(description="Run predictions pipeline with FASTA file")
parser.add_argument('--model', type=str, required=True, choices=['ProtT5', 'ProstT5'], help="Model to use: ProtT5 (original one) or ProstT5 (new one)")
parser.add_argument('--input_type', type=str, required=True, choices=['AA', 'AA+3Di'], help="Input type: AA or AA+3Di (AA+3Di is only supported by ProstT5)")
parser.add_argument('--pdb_path', type=str, default=None, help="Path to the input PDB file with protein structure (required if input_type=AA+3Di)")
args = parser.parse_args()

# Validate the arguments
if args.model == 'ProtT5' and args.input_type == 'AA+3Di':
    raise ValueError(f"{red}Error: Model ProtT5 does not support input type AA+3Di, please select ProstT5 for AA+3Di{reset}")

if args.input_type == 'AA+3Di' and not args.pdb_path:
    raise ValueError(f"{red}Error: --pdb_path is required when input_type is AA+3Di, please provide the path to the input PDB file if you want to use AA+3Di{reset}")

# Create Embeddings directory if not already present
cmd = 'mkdir -p Embeddings'
os.system(cmd)

# Pass the FASTA path, model, input_type, and pdb_path (if applicable) to fasta_to_ds.py
cmd = f'python3 fasta_to_ds.py'
os.system(cmd)

# Pass the model, input_type, and pdb_path arguments to predict_embed.py
cmd = f'python3 predict_embed.py --model {args.model} --input_type {args.input_type}'
if args.pdb_path:
    cmd += f' --pdb_path {args.pdb_path}'
os.system(cmd)

# Pass the model argument to append_embed.py (input_type and pdb_path not needed for this step)
cmd = f'python3 append_embed.py --model {args.model}'
os.system(cmd)

# Pass the model, input_type, and pdb_path arguments to make_predictions.py
cmd = f'python3 make_predictions.py --model {args.model} --input_type {args.input_type}'
if args.pdb_path:
    cmd += f' --pdb_path {args.pdb_path}'
os.system(cmd)
