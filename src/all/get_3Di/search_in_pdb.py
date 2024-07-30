import pandas as pd
import requests

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
        print(f"HTTP request failed: {http_err}")
        return None
    except Exception as err:
        print(f"Other error occurred: {err}")
        return None

# Extract the first sequence
first_sequence = data['Sequence'].iloc[1]

# Search for the first sequence
result = search_pdb(first_sequence)

# Print the result
if result:
    print("Result found in PDB:", result)
else:
    print("Sequence not found in PDB:", first_sequence)
