import pandas as pd

# Load both CSV files
lost_sf_df = pd.read_csv('./data/Dataset/csv/Lost_SF_and_Train_size.csv')
perf_df = pd.read_csv('./results/perf_dataframe.csv')

# Filter the `perf_df` to match the specified criteria
filtered_perf_df = perf_df[
    (perf_df['Model'] == 'ProstT5_full') &
    (perf_df['Nb_Layer_Block'] == 2) &
    (perf_df['Dropout'] == 0.3) &
    (perf_df['Input_Type'] == 'AA+3Di') &
    (perf_df['Layer_size'] == 2048)
]

# Merge the F1_Score from `filtered_perf_df` into `lost_sf_df` based on `pLDDT_threshold`, `is_top_50_SF`, and `Support_threshold`
merged_df = lost_sf_df.merge(
    filtered_perf_df[['pLDDT_threshold', 'is_top_50_SF', 'Support_threshold', 'F1_Score']],
    how='left',
    left_on=['pLDDT_threshold', 'Top_50_filtering', 'Support_threshold'],
    right_on=['pLDDT_threshold', 'is_top_50_SF', 'Support_threshold']
)

# Drop unnecessary columns from the merge
merged_df = merged_df.drop(columns=['is_top_50_SF'])

# Save the result to a new CSV file
merged_df.to_csv('./perf_and_SF_lost_summary.csv', index=False)
