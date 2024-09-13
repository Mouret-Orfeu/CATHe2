import subprocess
import os
import shutil

def get_3di_sequence(pdb_file_path, output_dir):
    """
    Extract 3Di sequence from a PDB file using Foldseek.
    The function will use Foldseek to create a 3Di sequence in FASTA format.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the last part of pdb_file_path (filename without directories)
    pdb_filename = os.path.basename(pdb_file_path)
    
    # Replace the file extension with '.fasta'
    file_name_3Di_sequence = os.path.splitext(pdb_filename)[0] + ".fasta"

    # Output database and sequence paths
    db_output = os.path.join(output_dir, "pdb_db")
    fasta_output = os.path.join(output_dir, file_name_3Di_sequence)

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

        # Remove all files in output_dir except file_name_3Di_sequence
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if filename != file_name_3Di_sequence:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove directories
                else:
                    os.remove(file_path)  # Remove files

        print("3Di sequence extracted successfully.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred during 3Di computation: {e}")

        


# Example usage
pdb_file_path = "./2vsq.pdb"  # Replace with your actual PDB file path
output_dir = "./3Di_sequence_folder"  # Directory to store the output

# Get the 3Di sequence
fasta_sequence = get_3di_sequence(pdb_file_path, output_dir)


