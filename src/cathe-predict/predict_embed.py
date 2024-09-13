# libraries
import numpy as np
from bio_embeddings.embed import ProtTransT5BFDEmbedder
import pandas as pd 
import argparse


parser = argparse.ArgumentParser(description="Run predictions pipeline with FASTA file")
parser.add_argument('--model', type=str,default='ProtT5', required=True, choices=['ProtT5', 'ProstT5'], help="Model to use: ProtT5 (original one) or ProstT5 (new one)")
parser.add_argument('--input_type', type=str,default='AA', required=True, choices=['AA', 'AA+3Di'], help="Input type: AA or AA+3Di (AA+3Di is only supported by ProstT5)")
parser.add_argument('--pdb_path', type=str, default=None, help="Path to the input PDB file with protein structure (required if input_type=AA+3Di)")
args = parser.parse_args()

def embed_sequence():
	
	embedder = ProtTransT5BFDEmbedder()

	ds = pd.read_csv('Dataset.csv')

	sequences_Example = list(ds["Sequence"])
	num_seq = len(sequences_Example)

	i = 0
	length = 1000
	while i < num_seq:
		print("Doing", i, num_seq)
		start = i 
		end = i + length

		sequences = sequences_Example[start:end]

		embeddings = []
		for seq in sequences:
			embeddings.append(np.mean(np.asarray(embedder.embed(seq)), axis=0))

		s_no = start / length
		filename = 'Embeddings/' + 'T5_' + str(s_no) + '.npz'
		embeddings = np.asarray(embeddings)
		# print(embeddings.shape)
		np.savez_compressed(filename, embeddings)
		i += length

# def embed_3Di(pdb_path)
# 	query_db_path = f"{pdb_path}_query"
# 	try:
#         run_command(f"foldseek createdb {pdb_path} {query_db_path}")
#         run_command(f"foldseek lndb {query_db_path}_h {query_db_path}_ss_h")

# 		query_db_ss_fasta = f"{query_db_path}_ss.fasta"
#         run_command(f"foldseek convert2fasta {query_db_path}_ss {query_db_ss_fasta}")
#         print(f"FASTA file created at {query_db_ss_fasta}")

#     except Exception as e:
#         print(f"An error occurred during 3Di computation: {e}")


def main():
		if args.input_type == 'AA':
			embed_sequence()
		elif args.input_type == 'AA+3Di':
			embed_sequence()
			# embed_3Di(args.pdb_path)
		else:
			print("Invalid input type")


if __name__ == '__main__':
    main()