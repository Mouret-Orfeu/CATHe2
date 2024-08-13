import pandas as pd 

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

def save_dataset_ids_for_3Di_usage_in_classification():
    # Train

    # Read the Train.fasta file and extract domain IDs
    with open('./data/Dataset/3Di/Train.fasta', 'r') as Train_fasta:
        fasta_train_entries = read_fasta(Train_fasta)

    Train_domain_ids_for_which_I_have_3Di = [int(entry[0]) for entry in fasta_train_entries]
    Train_domain_ids_for_which_I_have_3Di.sort()  # Sort the list in place

    print("number of domains in Train for which I have 3Di:", len(Train_domain_ids_for_which_I_have_3Di))

    # Save the domain IDs for 3Di usage in a CSV file
    df_train_ids_for_which_I_have_3Di = pd.DataFrame({'Domain_id': Train_domain_ids_for_which_I_have_3Di})
    df_train_ids_for_which_I_have_3Di.to_csv('./data/Dataset/csv/Train_ids_for_3Di_usage.csv', index=False)

    # Val and Test
    
    # I get the list of ids of the Val and Test domains for which I have the 3Di sequence
    with open('./data/Dataset/3Di/Val.fasta', 'r') as Val_fasta, open('./data/Dataset/3Di/Test.fasta', 'r') as Test_fasta:
        fasta_val_entries = read_fasta(Val_fasta)
        fasta_test_entries = read_fasta(Test_fasta)

    Val_domain_ids_for_which_I_have_3Di = [int(entry[0]) for entry in fasta_val_entries]
    Val_domain_ids_for_which_I_have_3Di.sort()  # Sort the list in place

    Test_domain_ids_for_which_I_have_3Di = [int(entry[0]) for entry in fasta_test_entries]
    Test_domain_ids_for_which_I_have_3Di.sort()  # Sort the list in place

    # I remove from these ids, the ids of the domains for which the SF is not represented by any domains in the Train domains for which I have 3Di

    # Read the Train.csv file and filter to get SF for the domains that have 3Di data
    df_train = pd.read_csv('./data/Dataset/csv/Train.csv')
    filtered_Train_df = df_train[df_train['Unnamed: 0'].isin(Train_domain_ids_for_which_I_have_3Di)]
    SF_for_Train_domains_with_3Di = set(filtered_Train_df['SF'].tolist())   

    print("len SF_for_Train_domains_with_3Di:", len(SF_for_Train_domains_with_3Di))

    # Read the Val.csv and Test.csv files
    df_val = pd.read_csv('./data/Dataset/csv/Val.csv')
    df_test = pd.read_csv('./data/Dataset/csv/Test.csv')

    # Get the indices where the SF is not in SF_for_Train_domains_with_3Di
    indices_in_val_not_in_train = df_val.index[~df_val['SF'].isin(SF_for_Train_domains_with_3Di)].tolist()
    indices_in_test_not_in_train = df_test.index[~df_test['SF'].isin(SF_for_Train_domains_with_3Di)].tolist()

    # Remove the indices from the Val and Test domain ids
    Val_domain_ids_for_which_I_have_3Di_and_which_are_in_Train = [id for id in Val_domain_ids_for_which_I_have_3Di if id not in indices_in_val_not_in_train]
    Test_domain_ids_for_which_I_have_3Di_and_which_are_in_Train = [id for id in Test_domain_ids_for_which_I_have_3Di if id not in indices_in_test_not_in_train]

    # Save the domain IDs for 3Di usage in CSV files
    df_val_ids_for_which_I_have_3Di = pd.DataFrame({'Domain_id': Val_domain_ids_for_which_I_have_3Di_and_which_are_in_Train})
    df_val_ids_for_which_I_have_3Di.to_csv('./data/Dataset/csv/Val_ids_for_3Di_usage.csv', index=False)

    df_test_ids_for_which_I_have_3Di = pd.DataFrame({'Domain_id': Test_domain_ids_for_which_I_have_3Di_and_which_are_in_Train})
    df_test_ids_for_which_I_have_3Di.to_csv('./data/Dataset/csv/Test_ids_for_3Di_usage.csv', index=False)

    



save_dataset_ids_for_3Di_usage_in_classification()