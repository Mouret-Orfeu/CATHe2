# libraries
import numpy as np
import pandas as pd 
import argparse
import subprocess
import os
import shutil

yellow = "\033[93m"
reset = "\033[0m"

def embed_sequence(model):

    if model == 'ProtT5':
        from bio_embeddings.embed import ProtTransT5BFDEmbedder

        print(f"{yellow}Loading ProtT5 model. (can take a few minutes){reset}")

        embedder = ProtTransT5BFDEmbedder()
        ds = pd.read_csv('./src/cathe-predict/Dataset.csv')

        sequences_Example = list(ds["Sequence"])
        num_seq = len(sequences_Example)

        i = 0
        length = 1000
        while i < num_seq:
            print("Doing", i, num_seq)
            start = i 
            end = i + length

            sequences = sequences_Example[start:end]

            embeddings = []
            for seq in sequences:
                embeddings.append(np.mean(np.asarray(embedder.embed(seq)), axis=0))

            s_no = start / length
            filename = './src/cathe-predict/Embeddings/' + 'ProtT5_' + str(s_no) + '.npz'

            embeddings = np.asarray(embeddings)
            np.savez_compressed(filename, embeddings)
            i += length
    
    if model == 'ProstT5':

        print("Embedding sequences with ProstT5")

        # Define the arguments for embed_all_models.py
        args = [
            'python3', './src/all/models/all_models/embed_all_models.py',
            '--model', 'ProstT5_full',
            '--is_3Di', '0',  
            '--seq_path', './src/cathe-predict/Dataset.csv',  
            '--embed_path', './src/cathe-predict/Embeddings/Embeddings_ProstT5_AA.npz',  # Path where embeddings will be saved
        ]

        # Run embed_all_models.py with the specified arguments
        subprocess.run(args)

        
        


def get_3di_sequences(pdb_folder_path, output_dir):
    """
    Extract 3Di sequences from all PDB files in a folder using Foldseek.
    Combine the 3Di sequences into a single FASTA file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check for PDB files in the folder
    pdb_files = [f for f in os.listdir(pdb_folder_path) if f.endswith(".pdb")]
    if not pdb_files:
        raise FileNotFoundError(
            f"No PDB files found. If the selected input_type is AA+3Di, provide PDB files at the folder path given ({pdb_folder_path}) from which 3Di sequences will be extracted."
        )

    # Output FASTA file that will contain all 3Di sequences
    combined_fasta_output = os.path.join(output_dir, "combined_3di_sequences.fasta")

    # Open the combined FASTA file in write mode
    with open(combined_fasta_output, 'w') as combined_fasta:

        # Loop through all PDB files in the folder
        for pdb_filename in pdb_files:
            pdb_file_path = os.path.join(pdb_folder_path, pdb_filename)

            # Replace the file extension with '.fasta' for individual 3Di sequences
            file_name_3Di_sequence = os.path.splitext(pdb_filename)[0] + ".fasta"
            fasta_output = os.path.join(output_dir, file_name_3Di_sequence)
            db_output = os.path.join(output_dir, "pdb_db")

            # Run Foldseek commands to create a sequence database and extract the sequence
            try:
                # Create database from PDB file (suppressing output)
                subprocess.run(f"foldseek createdb {pdb_file_path} {db_output}", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Convert the created database into a sequence FASTA file (suppressing output)
                subprocess.run(f"foldseek convert2fasta {db_output} {fasta_output}", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # Replace spaces with underscores in FASTA header lines
                with open(fasta_output, 'r') as fasta_file:
                    fasta_content = fasta_file.readlines()

                with open(fasta_output, 'w') as fasta_file:
                    for line in fasta_content:
                        if line.startswith(">"):
                            line = line.replace(" ", "_")  # Replace spaces with underscores in headers
                        fasta_file.write(line)

                # Append the contents of this FASTA file to the combined FASTA file
                with open(fasta_output, 'r') as fasta_file:
                    combined_fasta.write(fasta_file.read())

                # Clean up temporary files (remove individual FASTA files and databases)
                os.remove(fasta_output)
                shutil.rmtree(db_output, ignore_errors=True)

                print(f"3Di sequence for {pdb_filename} extracted and added to combined file.")

            except subprocess.CalledProcessError as e:
                print(f"An error occurred during 3Di computation for {pdb_filename}: {e}")

    print(f"All 3Di sequences have been combined into {combined_fasta_output}.")


def embed_3Di(pdb_path):
    output_dir = "./src/cathe-predict/3Di_sequence_folder"

    # Get the 3Di sequence
    get_3di_sequences(pdb_path, output_dir)

    try:
        subprocess.run(
            f"python ./src/all/models/all_models/embed_all_models.py --model ProstT5_full --is_3Di 1 --embed_path ./src/cathe-predict/Embeddings/3Di_embeddings.npz --seq_path {output_dir}",
            shell=True,
            check=True,
            stdout=subprocess.PIPE,  
            stderr=subprocess.PIPE
        )
        print("Embedding 3Di sequence with ProstT5_full completed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred during 3Di embedding: {e}")


def main():

    parser = argparse.ArgumentParser(description="Run predictions pipeline with FASTA file")
    parser.add_argument('--model', type=str, default='ProtT5', choices=['ProtT5', 'ProstT5'], help="Model to use: ProtT5 (original one) or ProstT5 (new one)")
    parser.add_argument('--input_type', type=str, default='AA', choices=['AA', 'AA+3Di'], help="Input type: AA or AA+3Di (AA+3Di is only supported by ProstT5)")
    args = parser.parse_args()

    # Create the folder for embeddings if it doesn't exist
    os.makedirs('./src/cathe-predict/Embeddings', exist_ok=True)

    if args.input_type == 'AA':
        
        embed_sequence(args.model)

    elif args.input_type == 'AA+3Di':

        pdb_path = 'path_to_folder_with_pdb_files' 

        if pdb_path == 'path_to_folder_with_pdb_files':
            raise ValueError("pdb_path must be changed to a folder path containing the pdb files for the domains to compute 3Di embeddings with.")

        if args.model == 'ProtT5':
            raise ValueError("ProtT5 model does not support 3Di embeddings. Please use ProstT5 if you want the input_type to be AA+3Di.")

        embed_sequence(args.model)
        embed_3Di(pdb_path)
    else:
        # raise error here

        raise ValueError("Invalid input_type. Please choose 'AA' or 'AA+3Di'.")


if __name__ == '__main__':
    main()
