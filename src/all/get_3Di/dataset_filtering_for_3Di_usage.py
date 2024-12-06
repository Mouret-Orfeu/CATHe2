# ANSI escape code for colored text
yellow = "\033[93m"
green = "\033[92m"
reset = "\033[0m"
red = "\033[91m"

print(f"{green}CATHe dataset filtering preparation code running, make sure you set up and activated venv_2{reset}")

import sys
import os


# Check if a virtual environment is active
if not hasattr(sys, 'base_prefix') or sys.base_prefix == sys.prefix:
    raise EnvironmentError(f"{red}No virtual environment is activated. Please activate the right venv_2 to run this code. See ReadMe for more details.{reset}")

# Get the name of the activated virtual environment
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path is None:
    raise EnvironmentError(f"{red}Error, venv path is none. Please activate the venv_2. See ReadMe for more details.{reset}")

venv_name = os.path.basename(venv_path)
if venv_name != "venv_2":
    raise EnvironmentError(f"{red}The activated virtual environment is '{venv_name}', not 'venv_2'. However venv_2 must be activated to run this code. See ReadMe for more details.{reset}")

import pandas as pd 
import csv
from tqdm import tqdm



def save_SF_lost_csv(pLDDT_threshold, total_lost_SF, nb_SF_remaining, Training_set_size, top_50_filtering, support_threshold):
    # Path to the CSV file
    lost_SF_csv_path = "./data/Dataset/csv/Lost_SF_and_Train_size.csv"
    
    # Create a DataFrame with the new data
    update_df = pd.DataFrame({
        'pLDDT_threshold': [pLDDT_threshold],
        'Lost_SF_count': [total_lost_SF],
        'Nb_SF_remaining': [nb_SF_remaining],
        'Training_set_size': [Training_set_size],
        'Top_50_filtering': [top_50_filtering],
        'Support_threshold': [support_threshold]
    })

    # Check if the CSV file already exists
    if os.path.exists(lost_SF_csv_path):
        # Load existing CSV
        df = pd.read_csv(lost_SF_csv_path)
        # Check if there is a matching row for both pLDDT_threshold and top_50_filtering
        condition = (df['pLDDT_threshold'] == pLDDT_threshold) & (df['Top_50_filtering'] == top_50_filtering) & (df['Support_threshold'] == support_threshold)
        
        if condition.any():
            # Update the row where the conditions match
            df.loc[condition, ['Lost_SF_count', 'Nb_SF_remaining', 'Training_set_size']] = [total_lost_SF, nb_SF_remaining, Training_set_size]
        else:
            # Append the new row if no match is found
            df = pd.concat([df, update_df], ignore_index=True)
    else:
        # Create a new DataFrame if the CSV does not exist
        df = update_df

    # Save the updated CSV
    df.to_csv(lost_SF_csv_path, index=False)



def read_fasta(file):
    """Reads a FASTA file and returns a list of tuples (id, header, sequence)."""
    fasta_entries = []
    header = None
    sequence = []
    for line in file:
        line = line.strip()
        if line.startswith(">"):
            if header:
                fasta_entries.append((header.split('_')[0], header, ''.join(sequence)))
            header = line[1:]
            sequence = []
        else:
            sequence.append(line)
    if header:
        fasta_entries.append((header.split('_')[0], header, ''.join(sequence)))
    return fasta_entries

def remove_lost_and_unrepresented_sf(filtered_sf, val_ids_with_3Di, test_ids_with_3Di, df_val, df_test):
    
    # Identify SFs not represented in Train threshold 0
    unique_SF_val = df_val['SF'].unique().tolist()  
    unique_SF_test = df_test['SF'].unique().tolist()
    sf_to_remove_for_val = set(unique_SF_val) - set(filtered_sf)
    sf_to_remove_for_test = set(unique_SF_test) - set(filtered_sf)
    
    
    # Filter Val IDs to remove lost and unrepresented SFs
    df_val_filtered_train_based = df_val[~df_val['SF'].isin(sf_to_remove_for_val)]
    val_filtered_ids = df_val_filtered_train_based[df_val_filtered_train_based['Unnamed: 0'].isin(val_ids_with_3Di)]['Unnamed: 0'].tolist()

    # Filter Test IDs to remove lost and unrepresented SFs
    df_test_filtered_train_based = df_test[~df_test['SF'].isin(sf_to_remove_for_test)]
    test_filtered_ids = df_test_filtered_train_based[df_test_filtered_train_based['Unnamed: 0'].isin(test_ids_with_3Di)]['Unnamed: 0'].tolist()

    return val_filtered_ids, test_filtered_ids  # Include the count of lost SFs

def save_dataset_ids_for_3Di_usage_in_classification(pLDDT_threshold, top_50_filtering, support_threshold):
    # Load the Train_pLDDT.csv file
    df_plddt = pd.read_csv('./data/Dataset/csv/Train_pLDDT.csv')

    # Filter the rows where pLDDT > pLDDT_threshold
    df_plddt_filtered = df_plddt[df_plddt['pLDDT'] > pLDDT_threshold]
    
    # Get the IDs that satisfy the pLDDT threshold
    valid_train_ids = set(df_plddt_filtered['ID'])

    # Read the Train.fasta file and extract domain IDs
    with open('./data/Dataset/3Di/Train.fasta', 'r') as Train_fasta, open('./data/Dataset/3Di/Val.fasta', 'r') as Val_fasta, open('./data/Dataset/3Di/Test.fasta', 'r') as Test_fasta:
        fasta_train_entries = read_fasta(Train_fasta)
        fasta_val_entries = read_fasta(Val_fasta)
        fasta_test_entries = read_fasta(Test_fasta)
    
    df_train = pd.read_csv('./data/Dataset/csv/Train.csv')

    # Load Val and Test data
    csv_val = './data/Dataset/csv/Val.csv'
    csv_test = './data/Dataset/csv/Test.csv'
    df_val = pd.read_csv(csv_val)
    df_test = pd.read_csv(csv_test)

    Val_domain_ids_for_which_I_have_3Di = [int(entry[0]) for entry in fasta_val_entries]
    Val_domain_ids_for_which_I_have_3Di.sort()

    Test_domain_ids_for_which_I_have_3Di = [int(entry[0]) for entry in fasta_test_entries]
    Test_domain_ids_for_which_I_have_3Di.sort()
    test_sf_with_3Di = set(df_test[df_test['Unnamed: 0'].isin(Test_domain_ids_for_which_I_have_3Di)]['SF'].tolist())

    # DEBUG
    all_test_sf = set(df_test['SF'].tolist())
    print(f"lost sf using 3Di for Test: {len(all_test_sf - test_sf_with_3Di)}")



    # Training dataset processing ############################################################

    # Compute the Train Domain_ids of the 50 largest SF (if top_50_filtering is enabled)
    if top_50_filtering:
        # Count the support for each SF and get the top 50 SFs by support
        sf_support_counts = df_train['SF'].value_counts()
        top_sf_support = sf_support_counts.head(50)

        # Get the top SF labels based on support
        top_sf_labels = top_sf_support.index.tolist()

        # Filter Domain_ids in Train.csv where the SF label is in the top SFs
        top_sf_domain_ids = df_train[df_train['SF'].isin(top_sf_labels)]['Unnamed: 0'].tolist()

    # Compute the Train Domain_ids for SF labels that have a support <= support_threshold (if support_threshold is not 0)
    if support_threshold:
        # Count the support for each SF
        sf_support_counts = df_train['SF'].value_counts()

        # Get the SF labels with support <= support_threshold
        low_support_sf_labels = sf_support_counts[sf_support_counts <= support_threshold].index.tolist()

        # Filter Domain_ids in Train.csv where the SF label is in the low support SFs
        low_support_sf_domain_ids = df_train[df_train['SF'].isin(low_support_sf_labels)]['Unnamed: 0'].tolist()

    # pLDDT filtering for Train: Get Train domain IDs that are in valid_train_ids and have 3Di data
    fully_filtered_Train_Domain_ids = [int(entry[0]) for entry in fasta_train_entries if int(entry[0]) in valid_train_ids]

    # If top 50 filtering is enabled, intersect with the top 50 SFs IDs
    if top_50_filtering:
        fully_filtered_Train_Domain_ids = list(set(fully_filtered_Train_Domain_ids) & set(top_sf_domain_ids))
    
    # If support filtering is enabled, remove the low support SFs from the fully filtered Train IDs
    if support_threshold:
        fully_filtered_Train_Domain_ids = list(set(fully_filtered_Train_Domain_ids) - set(low_support_sf_domain_ids))

    # Sort the list in place
    fully_filtered_Train_Domain_ids.sort()  

    # Save the domain IDs for 3Di usage in a CSV file
    df_fully_filtered_Domain_ids = pd.DataFrame({'Domain_id': fully_filtered_Train_Domain_ids})

    if top_50_filtering:
        Train_csv_path = f'./data/Dataset/csv/Train_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_top_50_SF.csv'
    elif support_threshold:
        Train_csv_path = f'./data/Dataset/csv/Train_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_support_threshold_{support_threshold}.csv'
    else:
        Train_csv_path = f'./data/Dataset/csv/Train_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}.csv'

    # Save filtered Train IDs 
    df_fully_filtered_Domain_ids.to_csv(Train_csv_path, index=False)

    # Get the set of SFs remaining in the fully filtered train set
    filtered_Train_df = df_train[df_train['Unnamed: 0'].isin(df_fully_filtered_Domain_ids['Domain_id'])]
    SF_filtered_Train = set(filtered_Train_df['SF'].tolist())

    # Save information about lost SFs and the size of the training set
    total_lost_SF = len(df_train['SF'].unique()) - len(SF_filtered_Train)
    nb_SF_remaining = len(SF_filtered_Train)
    Training_set_size = len(fully_filtered_Train_Domain_ids)

    save_SF_lost_csv(pLDDT_threshold, total_lost_SF, nb_SF_remaining, Training_set_size, top_50_filtering, support_threshold)


    # Val and Test datasets processing ############################################################

    # Filter and save Val and Test data
    val_filtered_ids, test_filtered_ids = remove_lost_and_unrepresented_sf(SF_filtered_Train, Val_domain_ids_for_which_I_have_3Di, Test_domain_ids_for_which_I_have_3Di, df_val, df_test)

    # Prepare DataFrame for Val and Test with idc_3Di_embed and idc_AA_embed
    df_val_filtered = pd.DataFrame({'Domain_id': val_filtered_ids})
    df_test_filtered = pd.DataFrame({'Domain_id': test_filtered_ids})

    if top_50_filtering:
        Test_csv_path = f'./data/Dataset/csv/Test_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_top_50_SF.csv'
        Val_csv_path = f'./data/Dataset/csv/Val_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_top_50_SF.csv'
    elif support_threshold:
        Test_csv_path = f'./data/Dataset/csv/Test_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_support_threshold_{support_threshold}.csv'
        Val_csv_path = f'./data/Dataset/csv/Val_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_support_threshold_{support_threshold}.csv'
    else:
        Test_csv_path = f'./data/Dataset/csv/Test_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}.csv'
        Val_csv_path = f'./data/Dataset/csv/Val_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}.csv'

    # Save Val and Test filtered IDs 
    df_val_filtered.to_csv(Val_csv_path, index=False)
    df_test_filtered.to_csv(Test_csv_path, index=False)



def main():

    top_50_filtering = False
    support_threshold = 10

    # Validate support_threshold
    if not isinstance(support_threshold, int) or support_threshold < 0:
        raise ValueError("support_threshold must be a non-negative integer")
    
        
    # Iterate over thresholds and filter Val and Test datasets
    for pLDDT_threshold in [0, 4, 14, 24, 34, 44, 54, 64, 74, 84]:
        save_dataset_ids_for_3Di_usage_in_classification(pLDDT_threshold, top_50_filtering, support_threshold)

if __name__ == "__main__":
    main()
