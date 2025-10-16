# This code snippet is used to parse the AA sequence fasta file and save it as a CSV file

import pandas as pd 
import argparse
import os

from Bio import SeqIO

# Parse command-line arguments for the model 
parser = argparse.ArgumentParser(description='Run predictions pipeline with FASTA file')
parser.add_argument('--model', type=str, default='ProtT5', choices=['ProtT5', 'ProstT5'], help='Model to use: ProtT5 (original one) or ProstT5 (new one)')
args = parser.parse_args()

fasta_file_path = './src/cathe-predict/Sequences.fasta'
output_csv_path = './src/cathe-predict/Dataset.csv'

if args.model == 'ProtT5':
    
    seq = []
    desc = []

    with open(fasta_file_path) as handle:
        for record in SeqIO.parse(handle, 'fasta'):
            seq.append(str(record.seq))
            desc.append(record.description)

    df = pd.DataFrame(list(zip(desc, seq)),
                columns =['Record', 'Sequence'])

    print(df)
    df.to_csv(output_csv_path)

    if df.empty or (not os.path.isfile(output_csv_path)) or os.path.getsize(output_csv_path) == 0:
        raise RuntimeError(f"Dataset CSV was not created or is empty: {output_csv_path}")

    print(f'Dataset saved to {output_csv_path}')

elif args.model == 'ProstT5':

    # Initialize lists to store parsed data
    indices = []
    domains = []
    sequences = []
    records = []

    # Parse the fasta file
    for i, record in enumerate(SeqIO.parse(fasta_file_path, 'fasta')):
        # Extract the ID, sequence, and full description
        description = record.description
        sequence = str(record.seq)
        
        # Extract domain (ID) as the first element after '>'
        domain = description.split('|')[0].replace('>', '')
        
        # Append data to lists
        indices.append(i)
        domains.append(domain)
        sequences.append(sequence)
        records.append(description)  # Add full description to the records list

    # Create DataFrame with the new 'Record' column
    df = pd.DataFrame({
        'Unnamed: 0': indices,
        'Domain': domains,
        'Sequence': sequences,
        'Record': records  # Add the Record column
    })

    # Save DataFrame to CSV
    df.to_csv(output_csv_path, index=False)

    if df.empty or (not os.path.isfile(output_csv_path)) or os.path.getsize(output_csv_path) == 0:
        raise RuntimeError(f"Dataset CSV was not created or is empty: {output_csv_path}")

    print(f'Dataset saved to {output_csv_path}')
