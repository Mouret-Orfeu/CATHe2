# ANSI escape code for colored text
yellow = '\033[93m'
green = '\033[92m'
reset = '\033[0m'
red = '\033[91m'


import sys
import os
import pandas as pd

# Check if a virtual environment is active
if not hasattr(sys, 'base_prefix') or sys.base_prefix == sys.prefix:
    raise EnvironmentError(f'{red}No virtual environment is activated. Please activate the right venv_2 to run this code. See ReadMe for more details.{reset}')

# Get the name of the activated virtual environment
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path is None:
    raise EnvironmentError(f'{red}Error, venv path is none. Please activate the venv_2. See ReadMe for more details.{reset}')

venv_name = os.path.basename(venv_path)
if venv_name != 'venv_2':
    raise EnvironmentError(f'{red}The activated virtual environment is {venv_name}, not venv_2. However venv_2 must be activated to run this code. See ReadMe for more details.{reset}')

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
