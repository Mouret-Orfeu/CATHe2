import pandas as pd 
from Bio import SeqIO

fasta_file_path = "/home/ku76797/Documents/internship/code/CATHe/rcsb_pdb_2VSQ.fasta"

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


