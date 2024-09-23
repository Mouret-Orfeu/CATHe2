import pandas as pd 
import os
import csv
from tqdm import tqdm

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

def remove_lost_and_unrepresented_sf(train_sf_0, threshold_sf, val_ids, test_ids, csv_val, csv_test):
    # Identify lost SFs
    lost_sf_train = set(train_sf_0).difference(set(threshold_sf))
    
    # Load Val and Test data
    df_val = pd.read_csv(csv_val)
    df_test = pd.read_csv(csv_test)
    
    # Identify SFs not represented in Train threshold 0
    unrepresented_sf_in_train_0 = df_val['SF'].unique().tolist() + df_test['SF'].unique().tolist()
    unrepresented_sf_in_train_0 = set(unrepresented_sf_in_train_0).difference(set(train_sf_0))
    
    # Filter Val IDs to remove lost and unrepresented SFs
    df_val_filtered = df_val[~df_val['SF'].isin(lost_sf_train | unrepresented_sf_in_train_0)]
    val_filtered_ids = df_val_filtered[df_val_filtered['Unnamed: 0'].isin(val_ids)]['Unnamed: 0'].tolist()

    # Filter Test IDs to remove lost and unrepresented SFs
    df_test_filtered = df_test[~df_test['SF'].isin(lost_sf_train | unrepresented_sf_in_train_0)]
    test_filtered_ids = df_test_filtered[df_test_filtered['Unnamed: 0'].isin(test_ids)]['Unnamed: 0'].tolist()

    return val_filtered_ids, test_filtered_ids, len(lost_sf_train)  # Include the count of lost SFs

def save_dataset_ids_for_3Di_usage_in_classification(pLDDT_threshold, train_sf_0, csv_writer):
    # Load the Train_pLDDT.csv file
    df_plddt = pd.read_csv('./data/Dataset/csv/Train_pLDDT.csv')

    # Filter the rows where pLDDT > pLDDT_threshold
    df_plddt_filtered = df_plddt[df_plddt['pLDDT'] > pLDDT_threshold]
    
    # Get the IDs that satisfy the pLDDT threshold
    valid_train_ids = set(df_plddt_filtered['ID'])

    # Read the Train.fasta file and extract domain IDs
    with open('./data/Dataset/3Di/Train.fasta', 'r') as Train_fasta:
        fasta_train_entries = read_fasta(Train_fasta)

    # Get Train domain IDs that are in valid_train_ids and have 3Di data
    Train_domain_ids_for_which_I_have_3Di = [int(entry[0]) for entry in fasta_train_entries if int(entry[0]) in valid_train_ids]
    Train_domain_ids_for_which_I_have_3Di.sort()  # Sort the list in place

    # Save the domain IDs for 3Di usage in a CSV file
    df_train_ids_for_which_I_have_3Di = pd.DataFrame({'Domain_id': Train_domain_ids_for_which_I_have_3Di})

    if pLDDT_threshold == 0:
        # Add the "order_id" to df_train_ids_for_which_I_have_3Di with id from 0
        df_train_ids_for_which_I_have_3Di['order_id'] = range(len(df_train_ids_for_which_I_have_3Di))
    else:
        # Add the order ids from the 0 threshold id csv to filter the 3Di embeddings in the future
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

    df_train_ids_for_which_I_have_3Di.to_csv(f'./data/Dataset/csv/Train_ids_for_3Di_usage_{pLDDT_threshold}.csv', index=False)

    # Val and Test
    with open('./data/Dataset/3Di/Val.fasta', 'r') as Val_fasta, open('./data/Dataset/3Di/Test.fasta', 'r') as Test_fasta:
        fasta_val_entries = read_fasta(Val_fasta)
        fasta_test_entries = read_fasta(Test_fasta)

    Val_domain_ids_for_which_I_have_3Di = [int(entry[0]) for entry in fasta_val_entries]
    Val_domain_ids_for_which_I_have_3Di.sort()

    Test_domain_ids_for_which_I_have_3Di = [int(entry[0]) for entry in fasta_test_entries]
    Test_domain_ids_for_which_I_have_3Di.sort()

    # Read the Train.csv file and get SF for the domains that have 3Di data
    df_train = pd.read_csv('./data/Dataset/csv/Train.csv')
    filtered_Train_df = df_train[df_train['Unnamed: 0'].isin(Train_domain_ids_for_which_I_have_3Di)]
    threshold_filtered_Train_df = filtered_Train_df[filtered_Train_df['Unnamed: 0'].isin(valid_train_ids)]
    SF_for_Train_domains_with_3Di = set(threshold_filtered_Train_df['SF'].tolist())

    # Get SFs for Train threshold
    csv_train = f'./data/Dataset/csv/Train.csv'
    threshold_sf = set(df_train[df_train['Unnamed: 0'].isin(Train_domain_ids_for_which_I_have_3Di)]['SF'])

    # Filter and save Val and Test data
    csv_val = './data/Dataset/csv/Val.csv'
    csv_test = './data/Dataset/csv/Test.csv'
    val_filtered_ids, test_filtered_ids, lost_sf_count = remove_lost_and_unrepresented_sf(train_sf_0, threshold_sf, Val_domain_ids_for_which_I_have_3Di, Test_domain_ids_for_which_I_have_3Di, csv_val, csv_test)

    df_val_filtered = pd.DataFrame({'Domain_id': val_filtered_ids})
    df_test_filtered = pd.DataFrame({'Domain_id': test_filtered_ids})

    df_val_filtered.to_csv(f'./data/Dataset/csv/Val_ids_for_3Di_usage_{pLDDT_threshold}.csv', index=False)
    df_test_filtered.to_csv(f'./data/Dataset/csv/Test_ids_for_3Di_usage_{pLDDT_threshold}.csv', index=False)
    
    # Calculate the Training set size and save to the CSV
    training_set_size = len(df_train_ids_for_which_I_have_3Di)
    csv_writer.writerow([pLDDT_threshold, lost_sf_count, training_set_size])

def main():
    # Calculate SFs for Train threshold 0
    csv_path_train = './data/Dataset/csv/Train.csv'
    df_train_ids_0 = pd.read_csv('./data/Dataset/csv/Train_ids_for_3Di_usage_0.csv')
    train_sf_0 = pd.read_csv(csv_path_train)
    train_sf_0 = train_sf_0[train_sf_0['Unnamed: 0'].isin(df_train_ids_0['Domain_id'])]['SF'].unique().tolist()
    
    # Prepare the CSV file for storing lost SF and training set size
    csv_file_path = './data/Dataset/csv/Lost_SF_and_Train_size.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['pLDDT_threshold', 'Lost_SF_count', 'Training_set_size'])  # Header
        
        # Iterate over thresholds and filter Val and Test datasets
        for pLDDT_threshold in [0, 4, 14, 24, 34, 44, 54, 64, 74, 84]:
            save_dataset_ids_for_3Di_usage_in_classification(pLDDT_threshold, train_sf_0, csv_writer)

if __name__ == "__main__":
    main()
