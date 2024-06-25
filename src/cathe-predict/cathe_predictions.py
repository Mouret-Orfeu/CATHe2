import os 

import tensorflow as tf

# ANSI escape code for yellow text
yellow = "\033[93m"
reset = "\033[0m"

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

cmd = 'mkdir Embeddings'
os.system(cmd)

cmd = 'python3 fasta_to_ds.py'
os.system(cmd)

cmd = 'python3 predict_embed.py'
os.system(cmd)

cmd = 'python3 append_embed.py'
os.system(cmd)

cmd = 'python3 make_predictions.py'
os.system(cmd)

