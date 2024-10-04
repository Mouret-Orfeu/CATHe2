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

    # If pLDDT threshold is 0, add idc_3Di_embde
    if pLDDT_threshold == 0:

        Train_domain_ids_for_which_I_have_3Di.sort()  # Sort the list in place

        # Save the domain IDs for 3Di usage in a CSV file
        df_train_ids_for_which_I_have_3Di = pd.DataFrame({'Domain_id': Train_domain_ids_for_which_I_have_3Di})

        df_train_ids_for_which_I_have_3Di['idc_3Di_embed'] = range(len(df_train_ids_for_which_I_have_3Di))
    else:
        Train_domain_ids_for_which_I_have_3Di.sort()  # Sort the list in place

        # Save the domain IDs for 3Di usage in a CSV file
        df_train_ids_for_which_I_have_3Di = pd.DataFrame({'Domain_id': Train_domain_ids_for_which_I_have_3Di})

        # For other thresholds, add idc_3Di_embed from threshold 0 CSV
        idc_3Di_embs = []
        with open('./data/Dataset/csv/Train_ids_for_3Di_usage_0.csv', 'r') as unfiltered_id_csv_file:
            reader = pd.read_csv(unfiltered_id_csv_file)
            for domain_id in tqdm(df_train_ids_for_which_I_have_3Di['Domain_id'], desc=f"Processing Train IDs, threshold {pLDDT_threshold}"):
                order_id_row = reader[reader['Domain_id'] == domain_id]
                if not order_id_row.empty:
                    idc_3Di_embed = order_id_row['idc_3Di_embed'].values[0]
                    idc_3Di_embs.append(idc_3Di_embed)
                else:
                    raise ValueError(f"No idc_3Di_embed found for domain_id {domain_id}")
        df_train_ids_for_which_I_have_3Di['idc_3Di_embed'] = idc_3Di_embs

    # Compute 'idc_AA_embed' column
    def compute_idc_AA_embed(domain_ids, df):
        idc_AA_embed = []
        domain_id_index_map = {domain_id: idx for idx, domain_id in enumerate(df['Unnamed: 0'])}
        for domain_id in domain_ids:
            if domain_id in domain_id_index_map:
                idc_AA_embed.append(domain_id_index_map[domain_id])
            else:
                idc_AA_embed.append(-1)  # In case domain_id is not found in the original dataset
        return idc_AA_embed

    df_train_ids_for_which_I_have_3Di['idc_AA_embed'] = compute_idc_AA_embed(df_train_ids_for_which_I_have_3Di['Domain_id'], df_train)

    # Save Train IDs with idc_3Di_embed and idc_AA_embed
    df_train_ids_for_which_I_have_3Di.to_csv(f'./data/Dataset/csv/Train_ids_for_3Di_usage_{pLDDT_threshold}.csv', index=False)

    # Get the set of SFs remaining in the fully filtered train set
    threshold_filtered_Train_df = df_train[df_train['Unnamed: 0'].isin(df_train_ids_for_which_I_have_3Di['Domain_id'])]
    SF_threshold_filtered_Train = set(threshold_filtered_Train_df['SF'].tolist())

    # Process Val and Test datasets similarly

    # Filter and save Val and Test data
    val_filtered_ids, test_filtered_ids = remove_lost_and_unrepresented_sf(SF_threshold_filtered_Train, Val_domain_ids_for_which_I_have_3Di, Test_domain_ids_for_which_I_have_3Di, df_val, df_test)

    # Prepare DataFrame for Val and Test with idc_3Di_embed and idc_AA_embed
    df_val_filtered = pd.DataFrame({'Domain_id': val_filtered_ids})
    df_test_filtered = pd.DataFrame({'Domain_id': test_filtered_ids})

    # For threshold 0, add idc_3Di_embed from 0
    if pLDDT_threshold == 0:
        df_val_filtered['idc_3Di_embed'] = range(len(df_val_filtered))
        df_test_filtered['idc_3Di_embed'] = range(len(df_test_filtered))
    else:
        # For other thresholds, add idc_3Di_embed from 0 threshold CSV
        idc_3Di_embs_val = []
        with open('./data/Dataset/csv/Val_ids_for_3Di_usage_0.csv', 'r') as unfiltered_id_csv_file:
            reader = pd.read_csv(unfiltered_id_csv_file)
            for domain_id in tqdm(df_val_filtered['Domain_id'], desc=f"Processing Val IDs, threshold {pLDDT_threshold}"):
                order_id_row = reader[reader['Domain_id'] == domain_id]
                if not order_id_row.empty:
                    idc_3Di_embed = order_id_row['idc_3Di_embed'].values[0]
                    idc_3Di_embs_val.append(idc_3Di_embed)
                else:
                    raise ValueError(f"No idc_3Di_embed found for domain_id {domain_id}")
        df_val_filtered['idc_3Di_embed'] = idc_3Di_embs_val

        idc_3Di_embs_test = []
        with open('./data/Dataset/csv/Test_ids_for_3Di_usage_0.csv', 'r') as unfiltered_id_csv_file:
            reader = pd.read_csv(unfiltered_id_csv_file)
            for domain_id in tqdm(df_test_filtered['Domain_id'], desc=f"Processing Test IDs, threshold {pLDDT_threshold}"):
                order_id_row = reader[reader['Domain_id'] == domain_id]
                if not order_id_row.empty:
                    idc_3Di_embed = order_id_row['idc_3Di_embed'].values[0]
                    idc_3Di_embs_test.append(idc_3Di_embed)
                else:
                    raise ValueError(f"No idc_3Di_embed found for domain_id {domain_id}")
        df_test_filtered['idc_3Di_embed'] = idc_3Di_embs_test

    # Compute 'idc_AA_embed' for Val and Test datasets
    df_val_filtered['idc_AA_embed'] = compute_idc_AA_embed(df_val_filtered['Domain_id'], df_val)
    df_test_filtered['idc_AA_embed'] = compute_idc_AA_embed(df_test_filtered['Domain_id'], df_test)

    # Save Val and Test IDs with idc_3Di_embed and idc_AA_embed
    df_val_filtered.to_csv(f'./data/Dataset/csv/Val_ids_for_3Di_usage_{pLDDT_threshold}.csv', index=False)
    df_test_filtered.to_csv(f'./data/Dataset/csv/Test_ids_for_3Di_usage_{pLDDT_threshold}.csv', index=False)



def main():
        
    # Iterate over thresholds and filter Val and Test datasets
    for pLDDT_threshold in [0, 4, 14, 24, 34, 44, 54, 64, 74, 84]:
        save_dataset_ids_for_3Di_usage_in_classification(pLDDT_threshold)

if __name__ == "__main__":
    main()
