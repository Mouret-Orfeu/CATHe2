import pandas as pd 
import os
import csv

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

def save_dataset_ids_for_3Di_usage_in_classification(pLDDT_threshold):
    # Load the Train_pLDDT.csv file
    df_plddt = pd.read_csv('./data/Dataset/csv/Train_pLDDT.csv')

    # Filter the rows where pLDDT > pLDDT_threshold
    df_plddt_filtered = df_plddt[df_plddt['pLDDT'] > pLDDT_threshold]
    
    # Get the IDs that satisfy the pLDDT threshold
    valid_train_ids = set(df_plddt_filtered['ID'])

    # DEBUG
    # print("number of domains in Train for which pLDDT > 0:", len(valid_train_ids))

    # Train

    # Read the Train.fasta file and extract domain IDs
    with open('./data/Dataset/3Di/Train.fasta', 'r') as Train_fasta:
        fasta_train_entries = read_fasta(Train_fasta)

    # Get Train domain IDs that are in valid_train_ids and have 3Di data
    Train_domain_ids_for_which_I_have_3Di = [int(entry[0]) for entry in fasta_train_entries if int(entry[0]) in valid_train_ids]
    Train_domain_ids_for_which_I_have_3Di.sort()  # Sort the list in place

    # print("number of domains in Train for which I have 3Di:", len(Train_domain_ids_for_which_I_have_3Di))

    # Save the domain IDs for 3Di usage in a CSV file
    df_train_ids_for_which_I_have_3Di = pd.DataFrame({'Domain_id': Train_domain_ids_for_which_I_have_3Di})
    df_train_ids_for_which_I_have_3Di.to_csv(f'./data/Dataset/csv/Train_ids_for_3Di_usage_{pLDDT_threshold}.csv', index=False)

    # Val and Test
    
    # I get the list of ids of the Val and Test domains for which I have the 3Di sequence
    with open('./data/Dataset/3Di/Val.fasta', 'r') as Val_fasta, open('./data/Dataset/3Di/Test.fasta', 'r') as Test_fasta:
        fasta_val_entries = read_fasta(Val_fasta)
        fasta_test_entries = read_fasta(Test_fasta)

    # Get Val and Test domain IDs that are in valid_train_ids and have 3Di data
    Val_domain_ids_for_which_I_have_3Di = [int(entry[0]) for entry in fasta_val_entries]
    Val_domain_ids_for_which_I_have_3Di.sort()  # Sort the list in place

    Test_domain_ids_for_which_I_have_3Di = [int(entry[0]) for entry in fasta_test_entries]
    Test_domain_ids_for_which_I_have_3Di.sort()  # Sort the list in place

    # DEBUG
    # print("number of domains in Val for which I have 3Di:", len(Val_domain_ids_for_which_I_have_3Di))
    # print("number of domains in Test for which I have 3Di:", len(Test_domain_ids_for_which_I_have_3Di))

    # I remove from these ids, the ids of the domains for which the SF is not represented by any domains in the Train domains for which I have 3Di

    # Read the Train.csv file and filter to get SF for the domains that have 3Di data
    df_train = pd.read_csv('./data/Dataset/csv/Train.csv')
    filtered_Train_df = df_train[df_train['Unnamed: 0'].isin(Train_domain_ids_for_which_I_have_3Di)]
    threshold_filtered_Train_df = filtered_Train_df[filtered_Train_df['Unnamed: 0'].isin(valid_train_ids)]
    SF_for_Train_domains_with_3Di = set(threshold_filtered_Train_df['SF'].tolist())   

    # DEBUG
    # print("len SF_for_Train_domains_with_3Di:", len(SF_for_Train_domains_with_3Di))

    # Read the Val.csv and Test.csv files
    df_val = pd.read_csv('./data/Dataset/csv/Val.csv')
    df_test = pd.read_csv('./data/Dataset/csv/Test.csv')

    # Get the indices where the SF are in SF_for_Train_domains_with_3Di
    filtered_df_val= df_val[df_val['SF'].isin(SF_for_Train_domains_with_3Di)]
    filtered_df_test= df_test[df_test['SF'].isin(SF_for_Train_domains_with_3Di)]

    ids_filtered_df_val = filtered_df_val['Unnamed: 0'].tolist()
    ids_filtered_df_test = filtered_df_test['Unnamed: 0'].tolist()

    # DEBUG
    # print("number of removed SF in Val:", len(indices_in_val_not_in_train))
    # print("number of removed SF in Test:", len(indices_in_test_not_in_train))

    # Remove the indices from the Val and Test domain ids
    Val_domain_ids_for_which_I_have_3Di_and_which_are_in_Train = [int(id) for id in Val_domain_ids_for_which_I_have_3Di if int(id) in ids_filtered_df_val]
    Test_domain_ids_for_which_I_have_3Di_and_which_are_in_Train = [int(id) for id in Test_domain_ids_for_which_I_have_3Di if int(id) in ids_filtered_df_test]

    #DEBUG
    # print("number of domains in Val for which I have 3Di for this threshold:", len(Val_domain_ids_for_which_I_have_3Di_and_which_are_in_Train))
    # print("number of domains in Test for which I have 3Di for this threshold:", len(Test_domain_ids_for_which_I_have_3Di_and_which_are_in_Train))

    # Save the domain IDs for 3Di usage in CSV files
    df_val_ids_for_which_I_have_3Di = pd.DataFrame({'Domain_id': Val_domain_ids_for_which_I_have_3Di_and_which_are_in_Train})
    df_val_ids_for_which_I_have_3Di.to_csv(f'./data/Dataset/csv/Val_ids_for_3Di_usage_{pLDDT_threshold}.csv', index=False)

    df_test_ids_for_which_I_have_3Di = pd.DataFrame({'Domain_id': Test_domain_ids_for_which_I_have_3Di_and_which_are_in_Train})
    df_test_ids_for_which_I_have_3Di.to_csv(f'./data/Dataset/csv/Test_ids_for_3Di_usage_{pLDDT_threshold}.csv', index=False)

    # Calculate the number of removed superfamilies in test set
    SF_removed_in_test = len(Test_domain_ids_for_which_I_have_3Di) - len(Test_domain_ids_for_which_I_have_3Di_and_which_are_in_Train)

    # Total removed SF
    total_lost_SF = SF_removed_in_test

    # Get the number of rows in df_plddt_filtered (which gives the training set size)
    training_set_size = len(df_plddt_filtered)

    # Save the lost SF count, threshold, and training set size in Lost_SF_and_Train_size.csv
    lost_sf_filename = './data/Dataset/csv/Lost_SF_and_Train_size.csv'
    if not os.path.exists(lost_sf_filename):
        with open(lost_sf_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['pLDDT_threshold', 'Lost_SF_count', 'Training_set_size'])
    
    with open(lost_sf_filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([pLDDT_threshold, total_lost_SF, training_set_size])


    

def main():
    for pLDDT_threshold in [0, 4, 14, 24, 34, 44, 54, 64, 74, 84]:
        save_dataset_ids_for_3Di_usage_in_classification(pLDDT_threshold)

if __name__ == "__main__":
    main()