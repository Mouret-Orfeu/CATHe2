import os
import csv
import requests
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

# Paths
fasta_file = './data/Dataset/3Di/Train.fasta'
output_csv = './data/Dataset/csv/Train_pLDDT.csv'
max_workers = 100  # We will create 100 workers to process the FASTA file in parallel

# For manual progress tracking
progress = 0
progress_lock = Lock()

# Function to download CIF file
def download_cif(database_code):
    url = f'https://alphafold.ebi.ac.uk/files/{database_code}-model_v4.cif'
    response = requests.get(url)
    
    if response.status_code == 200:
        cif_filename = f"{database_code}-model_v4.cif"
        with open(cif_filename, 'wb') as f:
            f.write(response.content)
        return cif_filename
    else:
        print(f"Failed to download CIF file for {database_code}")
        return None

# Function to extract global pLDDT score from CIF file
def extract_global_plddt(cif_file, expected_database_code):
    try:
        with open(cif_file, 'r') as file:
            lines = file.readlines()

            in_metric_global_section = False
            metric_value = None
            global_plddt_id = None

            for line in lines:
                line = line.strip()

                # Check the database code
                if line.startswith('_database_2.database_code'):
                    found_database_code = line.split()[-1]
                    cleaned_found_database_code = found_database_code.replace('-model_v4', '')

                    if cleaned_found_database_code != expected_database_code:
                        print(f"Database code mismatch: {cleaned_found_database_code} != {expected_database_code}")
                        return None

                # Capture the metric ID for global pLDDT
                if line.startswith("1 global pLDDT"):
                    global_plddt_id = line.split()[0]

                # Start collecting metric global section
                if line.startswith('_ma_qa_metric_global.metric_id'):
                    if global_plddt_id is None:
                        print("Error: Global pLDDT ID not found in the CIF file.")
                        return None

                    metric_id = line.split()[-1]

                    # Ensure this is the global pLDDT metric
                    if metric_id == global_plddt_id:
                        in_metric_global_section = True
                    else:
                        in_metric_global_section = False
                    continue

                # Extract global pLDDT value
                if in_metric_global_section and line.startswith('_ma_qa_metric_global.metric_value'):
                    metric_value = float(line.split()[-1])
                    break

            if metric_value is None:
                print("Error: Global pLDDT value could not be extracted.")
            return metric_value
    except FileNotFoundError:
        print(f"Error: CIF file {cif_file} not found.")
        return None

# Function to process each entry (download CIF, extract pLDDT, cleanup)
def process_records_chunk(records_chunk, total_records):
    global progress
    results = []
    
    for record in records_chunk:
        header = record.id
        parts = header.split('_')

        protein_seq_id = parts[0]
        afdb_id = parts[1]
        database_code = '-'.join(parts[1].split('-')[0:3])

        # Download CIF file
        cif_file = download_cif(database_code)
        if cif_file and os.path.exists(cif_file):  # Check if the file exists
            # Extract global pLDDT
            global_plddt = extract_global_plddt(cif_file, database_code)
            if global_plddt is not None:
                results.append((protein_seq_id, global_plddt))
            os.remove(cif_file)  # Clean up CIF file

        # Update progress
        with progress_lock:
            progress += 1
            print(f"Processing: {progress}/{total_records} records completed.", end='\r')

    return results

# Function to split the records into evenly-sized chunks
def chunk_records(records, num_chunks):
    avg = len(records) / float(num_chunks)
    chunks = []
    last = 0.0

    while last < len(records):
        chunks.append(records[int(last):int(last + avg)])
        last += avg

    return chunks

def get_pLDDT():
    # Parse the FASTA file and split into 100 chunks
    all_records = list(SeqIO.parse(fasta_file, 'fasta'))
    record_chunks = chunk_records(all_records, max_workers)

    # Count total number of records
    total_records = len(all_records)

    # Use ThreadPoolExecutor for multi-threading
    final_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_records_chunk, chunk, total_records) for chunk in record_chunks]

        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    final_results.extend(result)
            except Exception as e:
                print(f"An error occurred: {e}")

    # Sort results by protein sequence ID
    final_results.sort(key=lambda x: int(x[0]))

    # Write results to CSV, sorted by ID
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['id', 'pLDDT'])
        for row in final_results:
            csv_writer.writerow(row)


def verify_pLDDT_and_get_missing_ones(output_csv, fasta_file):
    # Load the CSV file and store existing entries
    with open(output_csv, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)  # Skip header if present
        rows = list(csv_reader)

    # Get all existing pLDDT IDs from the CSV
    pLDDT_ids = [row[0] for row in rows]

    # Parse the FASTA file and extract IDs and database codes
    fasta_entries_list = list(SeqIO.parse(fasta_file, 'fasta'))
    fasta_ids = []
    fasta_database_code = []

    for record in tqdm(fasta_entries_list, desc="Parsing FASTA Entries"):
        header = record.id
        parts = header.split('_')
        protein_seq_id = parts[0]
        fasta_ids.append(protein_seq_id)

        database_code = '-'.join(parts[1].split('-')[0:3])
        fasta_database_code.append(database_code)

    # Create a dictionary mapping fasta_ids to their database code
    fasta_dict = dict(zip(fasta_ids, fasta_database_code))

    # Convert IDs to sets for comparison
    pLDDT_ids = set(pLDDT_ids)
    fasta_ids = set(fasta_ids)

    # Find the missing IDs that are in the FASTA file but not in the CSV
    missing_ids = list(fasta_ids - pLDDT_ids)

    # Debug output to check the number of missing IDs
    print("Number of missing IDs:", len(missing_ids))

    added_missing_ids = 0
    new_rows = []

    # Process missing IDs and add pLDDT values to the CSV
    for missing_id in tqdm(missing_ids, desc="Processing Missing IDs"):
        database_code = fasta_dict.get(missing_id)

        # Download the CIF file and extract the pLDDT value
        try:
            cif_file = download_cif(database_code)
            if cif_file and os.path.exists(cif_file):  # Check if the CIF file was successfully downloaded
                global_plddt = extract_global_plddt(cif_file, database_code)
                if global_plddt is not None:
                    # Store the missing ID and pLDDT in a list to add later
                    new_rows.append([missing_id, global_plddt])
                    added_missing_ids += 1
                os.remove(cif_file)  # Clean up the CIF file after processing
        except Exception as e:
            print(f"Error processing {missing_id}: {e}")

    # Add the new rows to the original data and sort by ID
    rows.extend(new_rows)
    rows = sorted(rows, key=lambda x: int(x[0]))  # Sort rows by the first column (ID)

    # Write the combined and sorted rows back to the CSV
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(['ID', 'pLDDT'])
        # Write sorted data
        csv_writer.writerows(rows)

    print(f"Added missing pLDDT values: {added_missing_ids}/{len(missing_ids)}")


def sort_csv_by_id_as_int(input_csv, output_csv):
    # Read the CSV file
    with open(input_csv, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # Read the header
        rows = list(csv_reader)    # Read the rest of the data

    # Sort the rows by the first column (ID) as integers
    rows_sorted = sorted(rows, key=lambda x: int(x[0]))

    # Write the sorted data back to a new CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)  # Write the header
        csv_writer.writerows(rows_sorted)  # Write the sorted rows

    
def main():
    # get_pLDDT()
    # verify_pLDDT_and_get_missing_ones(output_csv, fasta_file)
    sort_csv_by_id_as_int(output_csv, output_csv)
    

if __name__ == '__main__':
    main()


    


