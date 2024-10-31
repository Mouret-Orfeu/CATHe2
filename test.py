import pandas as pd

# Load the CSV file
csv_path = './results/perf_dataframe.csv'
df = pd.read_csv(csv_path)

# Insert the new column before the last column in the DataFrame
df.insert(len(df.columns) - 1, 'is_top_50_SF', False)

# Save the updated DataFrame back to the CSV file
df.to_csv(csv_path, index=False)

print("Column 'is_top_50_SF' with default value False added before the last column.")
