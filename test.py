import pandas as pd

# Load the CSV file
csv_path = './data/Dataset/csv/Lost_SF_and_Train_size.csv'
df = pd.read_csv(csv_path)

# Insert the new column before the last column in the DataFrame
df.insert(len(df.columns), 'Support_threshold', 0)

# Save the updated DataFrame back to the CSV file
df.to_csv(csv_path, index=False)

