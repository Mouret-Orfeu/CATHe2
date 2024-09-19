import os
import subprocess

def compute_3di(pdb_file_path, output_dir):
    """
    Extract 3Di sequence from a PDB file using Foldseek.
    The function will use Foldseek to create a 3Di sequence in FASTA format.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the output paths
    db_output = os.path.join(output_dir, "pdb_db")
    fasta_output = os.path.join(output_dir, "1i5p_3di.fasta")

    foldseek_path = './foldseek/bin/foldseek' 

    try:
        # Step 1: Create a Foldseek database from the PDB file
        print(f"Creating database from {pdb_file_path}")
        subprocess.run(f"{foldseek_path} createdb {pdb_file_path} {db_output}", shell=True, check=True)

        # Step 2: Convert the database into a 3Di sequence FASTA file
        print("Converting the database to FASTA format...")
        subprocess.run(f"{foldseek_path} convert2fasta {db_output} {fasta_output}", shell=True, check=True)

        print(f"3Di sequence has been extracted and saved to {fasta_output}")

    except subprocess.CalledProcessError as e:
        print(f"Error while running Foldseek commands: {e}")
    
    # Optionally return the path of the 3Di sequence file
    return fasta_output

# Example usage
pdb_file = './1i5p.pdb'  # Replace with your PDB file path
output_directory = './3di_output_test'  # Directory to store the output

# Call the function to compute the 3Di sequence
compute_3di(pdb_file, output_directory)
