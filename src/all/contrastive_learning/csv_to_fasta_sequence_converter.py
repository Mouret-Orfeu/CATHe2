import pandas as pd


# Train.csv to fasta
# Paths
csv_path = './data/Dataset/csv/Test.csv'
fasta_path = './data/Dataset/csv/Test.fasta'

# Read CSV
df_train = pd.read_csv(csv_path)

# Open the FASTA file for writing
with open(fasta_path, 'w') as fasta_file:
    # Iterate through each row in the DataFrame
    for idx, row in df_train.iterrows():
        # Write the FASTA header using the value from the 'Unnamed: 0' column
        fasta_file.write(f">{row['Unnamed: 0']}\n")
        # Write the sequence data
        fasta_file.write(f"{row['Sequence']}\n")

print(f"FASTA file created at {fasta_path}")


# Val.csv to fasta
csv_path = './data/Dataset/csv/Val.csv'
fasta_path = './data/Dataset/csv/Val.fasta'

# Read CSV
df_Val = pd.read_csv(csv_path)

# Open the FASTA file for writing
with open(fasta_path, 'w') as fasta_file:
    # Iterate through each row in the DataFrame
    for idx, row in df_Val.iterrows():
        # Write the FASTA header using the value from the 'Unnamed: 0' column
        fasta_file.write(f">{idx+len(df_train)}\n")
        # Write the sequence data
        fasta_file.write(f"{row['Sequence']}\n")

print(f"FASTA file created at {fasta_path}")