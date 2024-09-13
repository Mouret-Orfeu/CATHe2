import os
import shutil
import zipfile
import requests
from tqdm import tqdm
import sys

# Optional: Install termcolor for colored output (if you don't want to use ANSI escape codes)
try:
    from termcolor import colored
except ImportError:
    # Install via pip if not installed
    print("Installing termcolor for colored output...")
    os.system("pip install termcolor")
    from termcolor import colored

# Define source and destination paths
source_zip = './CATHe Dataset.zip'
destination_dir = './data'
old_folder_name = os.path.join(destination_dir, 'CATHe Dataset')
new_folder_name = os.path.join(destination_dir, 'Dataset')
source_h5 = os.path.join(new_folder_name, 'weights', 'CATHe.h5')
destination_h5 = os.path.join(new_folder_name, 'weights', 'CATHe', 'CATHe.h5')

# Step 1: Check if 'CATHe Dataset.zip' exists at the root of the project
if not os.path.exists(source_zip):
    error_message = f"'{source_zip}' not found! Please download 'CATHe Dataset.zip' at the root of the project first."
    print(colored(error_message, 'red'))  # Print error message in red
    sys.exit(1)  # Exit the program without traceback

# Step 2: Create the './data' directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Step 3: Extract the ZIP file in './data' without moving it
if os.path.exists(source_zip):
    with zipfile.ZipFile(source_zip, 'r') as zip_ref:
        zip_ref.extractall(destination_dir)
    print(f"Extracted '{source_zip}' in '{destination_dir}'.")
else:
    print(f"File '{source_zip}' not found!")

# Step 4: Rename 'CATHe Dataset' to 'Dataset' if 'Dataset' does not already exist
if os.path.exists(old_folder_name):
    if not os.path.exists(new_folder_name):
        os.rename(old_folder_name, new_folder_name)
        print(f"Renamed '{old_folder_name}' to '{new_folder_name}'.")
    else:
        print(f"Directory '{new_folder_name}' already exists. Skipping renaming.")
else:
    print(f"Folder '{old_folder_name}' not found!")

# Step 5: Move the CATHe.h5 file to the correct location
# Ensure the destination directory exists before moving the file
os.makedirs(os.path.join(new_folder_name, 'weights', 'CATHe'), exist_ok=True)

if os.path.exists(source_h5):
    shutil.move(source_h5, destination_h5)
    print(f"Moved '{source_h5}' to '{destination_h5}'.")
else:
    print(f"File '{source_h5}' not found!")

# List of directories to create (if missing)
directories = [
    './data/Dataset/3Di',
    './data/Dataset/annotations',
    './data/Dataset/csv',
    './data/Dataset/embeddings/Other Class',
    './data/Dataset/weights/ProtT5/prot_t5_xl_uniref50',
    './data/Dataset/weights/TM_Vec',
    './data/pdb_files/missing_train_domains_id',
    './data/pdb_files/Test',
    './data/pdb_files/Train/Train_first',
    './data/pdb_files/Train/Train_missing_ones',
    './data/pdb_files/Train/Train_second',
    './data/pdb_files/Val',
    './data/test_data/foldseek/bin'
]

# Create directories if they do not exist
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Step 6: Ask the user if they want to download the ProtTrans model (default: No)
download_prompt = (
    "Do you want to download the ProtT5 model (5G)?\n"
    "Note: You can choose not to if you are not planning to re-compute embeddings, which you can already download following the README.\n"
    "Download now? [y/N]: "
)
user_input = input(download_prompt).strip().lower()

# Step 7: If the user says yes, download 'prot_t5_xl_uniref50.zip' using requests with a progress bar
if user_input == 'y':
    prot_t5_dir = './data/Dataset/weights/ProtT5'
    os.makedirs(prot_t5_dir, exist_ok=True)  # Ensure the directory exists

    url = 'https://zenodo.org/record/4644188/files/prot_t5_xl_uniref50.zip'
    zip_filename = 'prot_t5_xl_uniref50.zip'

    def download_file_with_progress(url, output_path):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))  # Total size in bytes
        block_size = 1024  # 1 Kilobyte
        t = tqdm(total=total_size, unit='iB', unit_scale=True)

        with open(output_path, 'wb') as file:
            for data in response.iter_content(block_size):
                t.update(len(data))
                file.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            print("ERROR: Something went wrong during the download.")
        else:
            print(f"Downloaded {output_path} successfully.")

    if not os.path.exists(os.path.join(prot_t5_dir, zip_filename)):
        print(f"Downloading {zip_filename}...")
        download_file_with_progress(url, os.path.join(prot_t5_dir, zip_filename))
    else:
        print(f"{zip_filename} already exists in {prot_t5_dir}.")

    # Step 8: Unzip the file
    if os.path.exists(os.path.join(prot_t5_dir, zip_filename)):
        print(f"Extracting {zip_filename}...")
        with zipfile.ZipFile(os.path.join(prot_t5_dir, zip_filename), 'r') as zip_ref:
            zip_ref.extractall(prot_t5_dir)
        print(f"Extracted {zip_filename}.")
    else:
        print(f"{zip_filename} not found.")
else:
    print("Skipping ProtTrans model download. Follow the README to use precomputed embeddings.")

# Step 9: Change back to the root (4 levels up)
os.chdir('../../../..')  # Go up 4 levels in the directory tree

print("Data directory created and file structure is completed.")
