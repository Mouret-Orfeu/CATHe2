import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load the CSV file
csv_file = './data/Dataset/csv/Test.csv'
data = pd.read_csv(csv_file)

# Function to search sequence in RCSB PDB
def search_pdb(sequence):
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    headers = {'Content-Type': 'application/json'}
    query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "target": "pdb_protein_sequence",
                "value": sequence,
                "identity_cutoff": 1.0,
                "sequence_type": "protein"
            }
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": [
                "computational",
                "experimental"
            ],
            "paginate": {
                "start": 0,
                "rows": 25
            }
        }
    }
    
    try:
        response = requests.post(url, json=query, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        # Check if the response is in JSON format
        if response.headers.get('Content-Type') == 'application/json':
            return response.json()
        else:
            print(f"Unexpected response format: {response.text}")
            return None
    except requests.exceptions.HTTPError as http_err:
        #print(f"HTTP request failed: {http_err}")
        return None
    except Exception as err:
        #print(f"Other error occurred: {err}")
        return None

# Prepare a list to collect results
results = []

# Function to process each row and search for sequences
def process_row(index, row):
    sequence_id = row['Unnamed: 0']
    sequence = row['Sequence']
    result = search_pdb(sequence)
    
    if result:
        identifiers = [entry['identifier'] for entry in result['result_set']]
        return {"id": sequence_id, "sequence": sequence, "identifiers": identifiers}
    else:
        return {"id": sequence_id, "sequence": sequence, "identifiers": None}

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=100) as executor:
    future_to_row = {executor.submit(process_row, index, row): index for index, row in data.iterrows()}
    for future in tqdm(as_completed(future_to_row), total=len(data)):
        try:
            result = future.result()
            results.append(result)
        except Exception as exc:
            print(f"Generated an exception: {exc}")

# Create a DataFrame from the results and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('./data/Dataset/csv/sequence_search_results.csv', index=False)

print("Search completed and results saved to sequence_search_results.csv")
