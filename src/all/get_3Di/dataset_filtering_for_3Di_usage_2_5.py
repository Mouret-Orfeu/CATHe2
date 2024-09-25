import pandas as pd 
import os
import csv
from tqdm import tqdm

def save_SF_lost_csv(pLDDT_threshold, total_lost_SF, nb_SF_remaining, Training_set_size):
    # Load the CSV file
    lost_SF_csv_path = "./data/Dataset/csv/Lost_SF_and_Train_size.csv"
    
    # Create the DataFrame to update
    update_df = pd.DataFrame({
        'pLDDT_threshold': [pLDDT_threshold],
        'Lost_SF_count': [total_lost_SF],
        'Nb_SF_remaining': [nb_SF_remaining],
        'Training_set_size': [Training_set_size]
    })

    # Check if the CSV file exists
    if os.path.exists(lost_SF_csv_path):
        # Load existing CSV
        df = pd.read_csv(lost_SF_csv_path)
        # Check if the threshold already exists
        if pLDDT_threshold in df['pLDDT_threshold'].values:
            # Update the corresponding row
            df.loc[df['pLDDT_threshold'] == pLDDT_threshold, ['Lost_SF_count', 'Nb_SF_remaining', 'Training_set_size']] = [total_lost_SF, nb_SF_remaining, Training_set_size]
        else:
            # Append new row
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

def save_dataset_ids_for_3Di_usage_in_classification(pLDDT_threshold):
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

    # Get Train domain IDs that are in valid_train_ids and have 3Di data
    Train_domain_ids_for_which_I_have_3Di = [int(entry[0]) for entry in fasta_train_entries if int(entry[0]) in valid_train_ids]

    

    # If pLDDT threshold is 0, add order_id
    if pLDDT_threshold == 0:

        Train_domain_ids_for_which_I_have_3Di.sort()  # Sort the list in place

        # Save the domain IDs for 3Di usage in a CSV file
        df_train_ids_for_which_I_have_3Di = pd.DataFrame({'Domain_id': Train_domain_ids_for_which_I_have_3Di})



        df_train_ids_for_which_I_have_3Di['order_id'] = range(len(df_train_ids_for_which_I_have_3Di))
    else:

        Train_domain_ids_for_which_I_have_3Di.sort()  # Sort the list in place

        # Save the domain IDs for 3Di usage in a CSV file
        df_train_ids_for_which_I_have_3Di = pd.DataFrame({'Domain_id': Train_domain_ids_for_which_I_have_3Di})

        # For other thresholds, add order_id from threshold 0 CSV
        order_ids = []
        with open('./data/Dataset/csv/Train_ids_for_3Di_usage_0.csv', 'r') as unfiltered_id_csv_file:
            reader = pd.read_csv(unfiltered_id_csv_file)
            for domain_id in tqdm(df_train_ids_for_which_I_have_3Di['Domain_id'], desc=f"Processing Train IDs, threshold {pLDDT_threshold}"):
                order_id_row = reader[reader['Domain_id'] == domain_id]
                if not order_id_row.empty:
                    order_id = order_id_row['order_id'].values[0]
                    order_ids.append(order_id)
                else:
                    raise ValueError(f"No order_id found for domain_id {domain_id}")
        df_train_ids_for_which_I_have_3Di['order_id'] = order_ids

    # Save Train IDs with order_id
    df_train_ids_for_which_I_have_3Di.to_csv(f'./data/Dataset/csv/Train_ids_for_3Di_usage_{pLDDT_threshold}.csv', index=False)

    

    # Read the Train.csv file and get SF for the domains that have 3Di data
    filtered_Train_df_for_3Di = df_train[df_train['Unnamed: 0'].isin(Train_domain_ids_for_which_I_have_3Di)]

    # DEBUG
    print(f"len filtered_Train_df_for_3Di {len(filtered_Train_df_for_3Di)}")

    threshold_filtered_Train_df = filtered_Train_df_for_3Di[filtered_Train_df_for_3Di['Unnamed: 0'].isin(valid_train_ids)]
    SF_threshold_filtered_Train = set(threshold_filtered_Train_df['SF'].tolist())

    Training_set_size = len(threshold_filtered_Train_df)

    if Training_set_size != len(df_plddt_filtered):
        raise ValueError(f"Training set size mismatch! Expected {len(df_plddt_filtered)} but got {Training_set_size}")

    # To record the number of SF lost and the training set size
    nb_SF_in_total = len(df_train['SF'].unique())
    nb_SF_for_which_I_have_3Di = len(filtered_Train_df_for_3Di['SF'].unique().tolist())
    nb_SF_after_pLDDT_filtering = len(SF_threshold_filtered_Train)

    nb_SF_lost_when_using_3Di = nb_SF_in_total - nb_SF_for_which_I_have_3Di
    nb_SF_lost_with_pLDDT_filtering = nb_SF_for_which_I_have_3Di - nb_SF_after_pLDDT_filtering
    total_lost_SF = nb_SF_lost_when_using_3Di + nb_SF_lost_with_pLDDT_filtering

    nb_SF_remaining = nb_SF_in_total - nb_SF_lost_when_using_3Di - nb_SF_lost_with_pLDDT_filtering

    # DEBUG
    print(f"nb SF in total (in train): {nb_SF_in_total}")
    print(f"nb_SF_for_which_I_have_3Di: {nb_SF_for_which_I_have_3Di}")
    print(f"nb_SF_after_pLDDT_filtering: {nb_SF_after_pLDDT_filtering}")
    
    print(f"nb SF lost when using 3Di (in train): {nb_SF_lost_when_using_3Di}")
    print(f"nb SF lost with pLDDT filtering (in train): {nb_SF_lost_with_pLDDT_filtering}")

    # Save the number of lost SF and training set size
    save_SF_lost_csv(pLDDT_threshold, total_lost_SF, nb_SF_remaining, Training_set_size)

    # Val and Test

    # Filter and save Val and Test data
    val_filtered_ids, test_filtered_ids = remove_lost_and_unrepresented_sf(SF_threshold_filtered_Train, Val_domain_ids_for_which_I_have_3Di, Test_domain_ids_for_which_I_have_3Di, df_val, df_test)

    # Prepare DataFrame for Val and Test with order_id
    df_val_filtered = pd.DataFrame({'Domain_id': val_filtered_ids})
    df_test_filtered = pd.DataFrame({'Domain_id': test_filtered_ids})

    # For threshold 0, add order_id from 0
    if pLDDT_threshold == 0:
        df_val_filtered['order_id'] = range(len(df_val_filtered))
        df_test_filtered['order_id'] = range(len(df_test_filtered))
    else:
        # For other thresholds, add order_id from 0 threshold CSV
        order_ids_val = []
        with open('./data/Dataset/csv/Val_ids_for_3Di_usage_0.csv', 'r') as unfiltered_id_csv_file:
            reader = pd.read_csv(unfiltered_id_csv_file)
            for domain_id in tqdm(df_val_filtered['Domain_id'], desc=f"Processing Val IDs, threshold {pLDDT_threshold}"):
                order_id_row = reader[reader['Domain_id'] == domain_id]
                if not order_id_row.empty:
                    order_id = order_id_row['order_id'].values[0]
                    order_ids_val.append(order_id)
                else:
                    raise ValueError(f"No order_id found for domain_id {domain_id}")
        df_val_filtered['order_id'] = order_ids_val

        order_ids_test = []
        with open('./data/Dataset/csv/Test_ids_for_3Di_usage_0.csv', 'r') as unfiltered_id_csv_file:
            reader = pd.read_csv(unfiltered_id_csv_file)
            for domain_id in tqdm(df_test_filtered['Domain_id'], desc=f"Processing Test IDs, threshold {pLDDT_threshold}"):
                order_id_row = reader[reader['Domain_id'] == domain_id]
                if not order_id_row.empty:
                    order_id = order_id_row['order_id'].values[0]
                    order_ids_test.append(order_id)
                else:
                    raise ValueError(f"No order_id found for domain_id {domain_id}")
        df_test_filtered['order_id'] = order_ids_test

    # Save Val and Test IDs with order_id
    df_val_filtered.to_csv(f'./data/Dataset/csv/Val_ids_for_3Di_usage_{pLDDT_threshold}.csv', index=False)
    df_test_filtered.to_csv(f'./data/Dataset/csv/Test_ids_for_3Di_usage_{pLDDT_threshold}.csv', index=False)

def main():
        
    # Iterate over thresholds and filter Val and Test datasets
    for pLDDT_threshold in [0, 4, 14, 24, 34, 44, 54, 64, 74, 84]:
        save_dataset_ids_for_3Di_usage_in_classification(pLDDT_threshold)

if __name__ == "__main__":
    main()
