import pandas as pd

# Paths
train_csv_path = './data/Dataset/csv/Train.csv'
test_csv_path = './data/Dataset/csv/Test.csv'
val_csv_path = './data/Dataset/csv/Val.csv'
txt_path = './data/Dataset/annotations/Y_Train_Test_Val_concat_SF.txt'

# Read CSVs
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
val_df = pd.read_csv(val_csv_path)

# Concatenate DataFrames
concat_df = pd.concat([train_df, test_df, val_df], ignore_index=True)

# Helper function to format a row
def format_row(seq_id, sf):
    parts = sf.split('.')
    formatted_row = f"{seq_id:<10} {int(parts[0]):>4} {int(parts[1]):>6} {int(parts[2]):>6} {int(parts[3]):>6} {'1':>6} {'1':>6} {'1':>6} {'1':>6} {'1':>6} {'59':>6} {'1.000':>6}"
    return formatted_row

# Generate formatted rows
rows = [format_row(f"{idx:06}", row['SF']) for idx, row in concat_df.iterrows()]

# Write to TXT file
with open(txt_path, 'w') as txt_file:
    txt_file.write('\n'.join(rows))

print(f"TXT file created at {txt_path}")
