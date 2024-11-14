import pandas as pd 
import argparse

from Bio import SeqIO

parser = argparse.ArgumentParser(description="Run predictions pipeline with FASTA file")
parser.add_argument('--model', type=str, default='ProtT5', choices=['ProtT5', 'ProstT5'], help="Model to use: ProtT5 (original one) or ProstT5 (new one)")
args = parser.parse_args()

fasta_file_path = '/home/orfeu/Documents/documents/important/travail/stage/stage/stage_2023_2024/Recherche_scientifique_Londre/Auto_Prot_function_detection_and_classification/work/code/CATHe/src/cathe-predict/Sequences.fasta'

if args.model == 'ProtT5':
    
    seq = []
    desc = []

    with open(fasta_file_path) as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq.append(str(record.seq))
            desc.append(record.description)

    df = pd.DataFrame(list(zip(desc, seq)),
                columns =['Record', 'Sequence'])

    print(df)
    df.to_csv('./src/cathe-predict/Dataset.csv')

elif args.model == 'ProstT5':

    output_csv_path = "./src/cathe-predict/Dataset.csv"

    # Initialize lists to store parsed data
    indices = []
    domains = []
    sequences = []
    records = []

    # Parse the fasta file
    for i, record in enumerate(SeqIO.parse(fasta_file_path, "fasta")):
        # Extract the ID, sequence, and full description
        description = record.description
        sequence = str(record.seq)
        
        # Extract domain (ID) as the first element after '>'
        domain = description.split("|")[0].replace(">", "")
        
        # Append data to lists
        indices.append(i)
        domains.append(domain)
        sequences.append(sequence)
        records.append(description)  # Add full description to the records list

    # Create DataFrame with the new 'Record' column
    df = pd.DataFrame({
        "Unnamed: 0": indices,
        "Domain": domains,
        "Sequence": sequences,
        "Record": records  # Add the Record column
    })

    # Save DataFrame to CSV
    df.to_csv(output_csv_path, index=False)

    print(f"Dataset saved to {output_csv_path}")




else:
    raise ValueError("Invalid model. Please choose 'ProtT5' or 'ProstT5'.")


