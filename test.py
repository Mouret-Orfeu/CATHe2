import subprocess
import os

def get_3di_sequence(pdb_file_path, output_dir):
    """
    Extract 3Di sequence from a PDB file using Foldseek.
    The function will use Foldseek to create a 3Di sequence in FASTA format.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output database and sequence paths
    db_output = os.path.join(output_dir, "pdb_db")
    fasta_output = os.path.join(output_dir, "pdb_3di.fasta")

    # Run Foldseek commands to create a sequence database and extract the sequence
    try:
        # Create database from PDB file (suppressing output)
        subprocess.run(f"foldseek createdb {pdb_file_path} {db_output}", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Convert the created database into a sequence FASTA file (suppressing output)
        subprocess.run(f"foldseek convert2fasta {db_output} {fasta_output}", shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Read the output FASTA file to get the 3Di sequence
        with open(fasta_output, 'r') as f:
            fasta_content = f.read()
        
        print("3Di sequence extracted successfully.")
        return fasta_content

    except subprocess.CalledProcessError as e:
        print(f"An error occurred during 3Di computation: {e}")
        return None


# Example usage
pdb_file_path = "./2vsq.pdb"  # Replace with your actual PDB file path
output_dir = "./output"  # Directory to store the output

# Get the 3Di sequence
fasta_sequence = get_3di_sequence(pdb_file_path, output_dir)

if fasta_sequence:
    # Optionally print the extracted sequence (can be commented out)
    # print(f"Extracted 3Di Sequence:\n{fasta_sequence}")
    pass
