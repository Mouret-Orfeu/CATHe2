import pandas as pd 
from Bio import SeqIO

fasta_file_path = "/home/orfeu/Documents/Internship/Work/sandbag_storage/uniprotkb_accession_A0A0C5B5G6_2024_06_20.fasta"

seq = []
desc = []

with open(fasta_file_path) as handle:
    for record in SeqIO.parse(handle, "fasta"):
        seq.append(str(record.seq))
        desc.append(record.description)

df = pd.DataFrame(list(zip(desc, seq)),
               columns =['Record', 'Sequence'])

print(df)
df.to_csv('Dataset.csv')


