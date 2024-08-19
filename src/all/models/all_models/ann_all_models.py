# the light attention part is inspired from  https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py

import argparse
import pandas as pd 
import os
import numpy as np 
import gc
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras  
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU, Conv1D, Softmax, GlobalAveragePooling1D, Concatenate, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle, resample
import torch
from tqdm import tqdm
import warnings
import seaborn as sns
import math
import csv
import matplotlib
import matplotlib.colors as mcolors
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from memory_profiler import profile




from tensorflow.compat.v1 import ConfigProto

tf.keras.backend.clear_session()
config = ConfigProto()
config.gpu_options.allow_growth = True


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

nb_layer_block_dict = {
        "one": 1,
        "two": 2,
        "three": 3
    }



def domain_id_generator(domain_ids):
    for index, id_value in enumerate(domain_ids):
        yield id_value, index

# @profile
def load_data(model_name, input_type, pLDDT_threshold):
    """Loads data for the specified model."""

    if model_name == 'ProtT5':
        # using the original CATHe datasets for ProtT5
        ds_train = pd.read_csv('./data/Dataset/annotations/Y_Train_SF.csv')
        y_train = list(ds_train["SF"])

        filename = './data/Dataset/embeddings/SF_Train_ProtT5_per_protein.npz'
        X_train = np.load(filename)['arr_0']
        filename = './data/Dataset/embeddings/Other Class/Other_Train_per_protein.npz'
        X_train_other = np.load(filename)['arr_0']

        X_train = np.concatenate((X_train, X_train_other), axis=0)

        for i in range(len(X_train_other)):
            y_train.append('other')

        # val
        ds_val = pd.read_csv('./data/Dataset/annotations/Y_Val_SF.csv')
        y_val = list(ds_val["SF"])

        filename = './data/Dataset/embeddings/SF_Val_ProtT5_per_protein.npz'
        X_val = np.load(filename)['arr_0']

        # filename = './data/Dataset/embeddings/Other Class/Other_Val_US_per_protein.npz'
        filename = './data/Dataset/embeddings/Other Class/Other_Val_per_protein.npz'
        X_val_other = np.load(filename)['arr_0']

        X_val = np.concatenate((X_val, X_val_other), axis=0)

        for i in range(len(X_val_other)):
            y_val.append('other')

        # test
        ds_test = pd.read_csv('./data/Dataset/annotations/Y_Test_SF.csv')
        y_test = list(ds_test["SF"])

        filename = './data/Dataset/embeddings/SF_Test_ProtT5_per_protein.npz'
        X_test = np.load(filename)['arr_0']

        # filename = './data/Dataset/embeddings/Other Class/Other_Test_US_per_protein.npz'
        filename = './data/Dataset/embeddings/Other Class/Other_Test_per_protein.npz'
        X_test_other = np.load(filename)['arr_0']

        X_test = np.concatenate((X_test, X_test_other), axis=0)

        for _ in range(len(X_test_other)):
            y_test.append('other')
    
    else:


        # labels y_train, y_val, y_test

        df_train = pd.read_csv('./data/Dataset/csv/Train.csv')
        df_val = pd.read_csv('./data/Dataset/csv/Val.csv')
        df_test = pd.read_csv('./data/Dataset/csv/Test.csv')
        
        if input_type == '3Di' or input_type == 'AA+3Di':
            # Load the domain IDs for which 3Di data is available

            train_ids_for_3Di_usage = set(pd.read_csv(f'./data/Dataset/csv/Train_ids_for_3Di_usage_{pLDDT_threshold}.csv')['Domain_id'])
            val_ids_for_3Di_usage = set(pd.read_csv(f'./data/Dataset/csv/Val_ids_for_3Di_usage_{pLDDT_threshold}.csv')['Domain_id'])
            test_ids_for_3Di_usage = set(pd.read_csv(f'./data/Dataset/csv/Test_ids_for_3Di_usage_{pLDDT_threshold}.csv')['Domain_id'])

            # Ensure that 'Unnamed: 0' is integer
            df_train['Unnamed: 0'] = df_train['Unnamed: 0'].astype(int)
            df_val['Unnamed: 0'] = df_val['Unnamed: 0'].astype(int)
            df_test['Unnamed: 0'] = df_test['Unnamed: 0'].astype(int)

            # Filter the datasets to keep only the domains for which 3Di data is available
            df_train = df_train[df_train['Unnamed: 0'].isin(train_ids_for_3Di_usage)]
            df_val = df_val[df_val['Unnamed: 0'].isin(val_ids_for_3Di_usage)]
            df_test = df_test[df_test['Unnamed: 0'].isin(test_ids_for_3Di_usage)]

        y_train = df_train['SF'].tolist()
        y_val = df_val['SF'].tolist()
        y_test = df_test['SF'].tolist()

        #     # Filter the DataFrame to include only the rows where the id is present in train_ids_for_3Di_usage
        #     df_train_filtered = df_train[df_train['Unnamed: 0'].isin(train_ids_for_3Di_usage)]
        #     df_val_filtered = df_val[df_val['Unnamed: 0'].isin(val_ids_for_3Di_usage)]
        #     df_test_filtered = df_test[df_test['Unnamed: 0'].isin(test_ids_for_3Di_usage)]

        # y_train = df_train_filtered['SF'].tolist()
        # y_val = df_val_filtered['SF'].tolist()
        # y_test = df_test_filtered['SF'].tolist()

        # #DEBUG
        # print("Length of y_val: ", len(y_val))
        # print("Length of y_test: ", len(y_test))

        
        # features X_train, X_val, X_test
        
        prot_sequence_embeddings_paths = {
            'ProtT5': ('Train_ProtT5_per_protein.npz', 'Val_ProtT5_per_protein.npz', 'Test_ProtT5_per_protein.npz'),
            'ProtT5_new' : ('Train_ProtT5_new_per_protein.npz', 'Val_ProtT5_new_per_protein.npz', 'Test_ProtT5_new_per_protein.npz'),
            'ESM2': ('Train_ESM2_per_protein.npz', 'Val_ESM2_per_protein.npz', 'Test_ESM2_per_protein.npz'),
            'Ankh_large': ('Train_Ankh_large_per_protein.npz', 'Val_Ankh_large_per_protein.npz', 'Test_Ankh_large_per_protein.npz'),
            'Ankh_base': ('Train_Ankh_base_per_protein.npz', 'Val_Ankh_base_per_protein.npz', 'Test_Ankh_base_per_protein.npz'),
            'ProstT5_full': ('Train_ProstT5_full_per_protein.npz', 'Val_ProstT5_full_per_protein.npz', 'Test_ProstT5_full_per_protein.npz'),
            'ProstT5_half': ('Train_ProstT5_half_per_protein.npz', 'Val_ProstT5_half_per_protein.npz','Test_ProstT5_half_per_protein.npz'),
            'TM_Vec': ('Train_TM_Vec_per_protein.npz', 'Val_TM_Vec_per_protein.npz', 'Test_TM_Vec_per_protein.npz')
        }

        prot_3Di_embeddings_paths = {
            'ProstT5_full': ('Train_ProstT5_full_per_protein_3Di.npz', 'Val_ProstT5_full_per_protein_3Di.npz', 'Test_ProstT5_full_per_protein_3Di.npz'),
            'ProstT5_half': ('Train_ProstT5_half_per_protein_3Di.npz', 'Val_ProstT5_half_per_protein_3Di.npz', 'Test_ProstT5_half_per_protein_3Di.npz')
            
        }

        if input_type == 'AA':
            if model_name not in prot_sequence_embeddings_paths:
                raise ValueError("Invalid model name")
        else:
            if model_name not in prot_3Di_embeddings_paths:
                raise ValueError("Invalid model name")
        
        Train_file_name_seq_embed, Val_file_name_seq_embed, Test_file_name_seq_embed = prot_sequence_embeddings_paths[model_name]
        Train_file_name_3Di_embed, Val_file_name_3Di_embed, Test_file_name_3Di_embed = prot_3Di_embeddings_paths[model_name]

        if input_type == 'AA':
            
            X_train = np.load(f'./data/Dataset/embeddings/{Train_file_name_seq_embed}')['arr_0']
            X_val = np.load(f'./data/Dataset/embeddings/{Val_file_name_seq_embed}')['arr_0']
            X_test = np.load(f'./data/Dataset/embeddings/{Test_file_name_seq_embed}')['arr_0']
        
        if input_type == '3Di':

            X_train = np.load(f'./data/Dataset/embeddings/{Train_file_name_3Di_embed}')['arr_0']
            X_val = np.load(f'./data/Dataset/embeddings/{Val_file_name_3Di_embed}')['arr_0']
            X_test = np.load(f'./data/Dataset/embeddings/{Test_file_name_3Di_embed}')['arr_0'] 
        
        if input_type == 'AA+3Di':

            # # Load sequence embeddings
            # X_train_seq = np.load(f'./data/Dataset/embeddings/{Train_file_name_seq_embed}')['arr_0']
            # X_val_seq = np.load(f'./data/Dataset/embeddings/{Val_file_name_seq_embed}')['arr_0']
            # X_test_seq = np.load(f'./data/Dataset/embeddings/{Test_file_name_seq_embed}')['arr_0']

            # # Ensure that 'Unnamed: 0' is integer
            # df_train['Unnamed: 0'] = df_train['Unnamed: 0'].astype(int)
            # df_val['Unnamed: 0'] = df_val['Unnamed: 0'].astype(int)
            # df_test['Unnamed: 0'] = df_test['Unnamed: 0'].astype(int)

            # # Filter the DataFrame to include only the IDs present in train_ids_for_3Di_usage
            # filtered_df_train = df_train[df_train['Unnamed: 0'].isin(train_ids_for_3Di_usage)]
            # filtered_df_val = df_val[df_val['Unnamed: 0'].isin(val_ids_for_3Di_usage)]
            # filtered_df_test = df_test[df_test['Unnamed: 0'].isin(test_ids_for_3Di_usage)]

            # # Get the indices of the rows that should be kept
            # train_indices_to_keep = filtered_df_train.index
            # val_indices_to_keep = filtered_df_val.index
            # test_indices_to_keep = filtered_df_test.index

            # # Use these indices to filter the embeddings arrays
            # X_train_seq_filtered = X_train_seq[train_indices_to_keep]
            # X_val_seq_filtered = X_val_seq[val_indices_to_keep]
            # X_test_seq_filtered = X_test_seq[test_indices_to_keep]

            # # Load 3Di embeddings
            # X_train_3Di = np.load(f'./data/Dataset/embeddings/{Train_file_name_3Di_embed}')['arr_0']
            # X_val_3Di = np.load(f'./data/Dataset/embeddings/{Val_file_name_3Di_embed}')['arr_0']
            # X_test_3Di = np.load(f'./data/Dataset/embeddings/{Test_file_name_3Di_embed}')['arr_0']

            # train_3Di_id_to_keep = list(pd.read_csv(f'./data/Dataset/csv/Train_ids_for_3Di_usage_{pLDDT_threshold}.csv')['order_id'])
            # val_3Di_id_to_keep =   list(pd.read_csv(f'./data/Dataset/csv/Val_ids_for_3Di_usage_{pLDDT_threshold}.csv')['order_id'])
            # test_3Di_id_to_keep =  list(pd.read_csv(f'./data/Dataset/csv/Test_ids_for_3Di_usage_{pLDDT_threshold}.csv')['order_id'])
            
            # #DEBUG
            # # print("Length of X_train_3Di_unfiltered: ", len(X_train_3Di))
            # # print("Length of nb id in train_3Di_id_to_keep: ", len(train_3Di_id_to_keep))
            # # print("Length of X_train_seq_filtered: ", len(X_train_seq_filtered))

            # # print("Length of X_val_3Di_unfiltered: ", len(X_val_3Di))
            # # print("Length of nb id in val_3Di_id_to_keep: ", len(val_3Di_id_to_keep))
            # # print("Length of X_val_seq_filtered: ", len(X_val_seq_filtered))

            # # print("Length of X_test_3Di_unfiltered: ", len(X_test_3Di))
            # # print("Length of nb id in test_3Di_id_to_keep: ", len(test_3Di_id_to_keep))
            # # print("Length of X_test_seq_filtered: ", len(X_test_seq_filtered))


            # # exit()

            # # filter the 3Di embeddings so that only the ones with pLDDT > threshold are kept
            # X_train_3Di_filtered = X_train_3Di[train_3Di_id_to_keep]
            # X_val_3Di_filtered = X_val_3Di[val_3Di_id_to_keep]
            # X_test_3Di_filtered = X_test_3Di[test_3Di_id_to_keep]

            # # Ensure that lengths match before concatenation
            # assert len(X_train_seq_filtered) == len(X_train_3Di_filtered), "Train sequence and 3Di embeddings must have the same length"
            # assert len(X_val_seq_filtered) == len(X_val_3Di_filtered), "Val sequence and 3Di embeddings must have the same length"
            # assert len(X_test_seq_filtered) == len(X_test_3Di_filtered), "Test sequence and 3Di embeddings must have the same length"


            # # Concatenate the sequence and 3Di embeddings along the feature axis
            # X_train = np.concatenate((X_train_seq_filtered, X_train_3Di_filtered), axis=1)
            # X_val = np.concatenate((X_val_seq_filtered, X_val_3Di_filtered), axis=1)
            # X_test = np.concatenate((X_test_seq_filtered, X_test_seq_filtered), axis=1)

            # Load sequence embeddings
            X_train_seq_embeddings = np.load(f'./data/Dataset/embeddings/{Train_file_name_seq_embed}')['arr_0']
            X_val_seq_embeddings = np.load(f'./data/Dataset/embeddings/{Val_file_name_seq_embed}')['arr_0']
            X_test_seq_embeddings = np.load(f'./data/Dataset/embeddings/{Test_file_name_seq_embed}')['arr_0']

            # Get the indices of the rows that should be kept, (not the seq id but the actual row in the csv file, for example the first id of the test set is 0, but the first sequence id is 1035679)
            # Find the row numbers (positions) for train, val, and test IDs in their respective DataFrames
            train_domain_id_to_keep = df_train[df_train['Unnamed: 0'].isin(train_ids_for_3Di_usage)].index.tolist()
            val_domain_id_to_keep = df_val[df_val['Unnamed: 0'].isin(val_ids_for_3Di_usage)].index.tolist()
            test_domain_id_to_keep = df_test[df_test['Unnamed: 0'].isin(test_ids_for_3Di_usage)].index.tolist()

            # Use these indices to filter the embeddings arrays
            X_train_seq_embeddings_filtered = X_train_seq_embeddings[train_domain_id_to_keep]
            X_val_seq_embeddings_filtered = X_val_seq_embeddings[val_domain_id_to_keep]
            X_test_seq_embeddings_filtered = X_test_seq_embeddings[test_domain_id_to_keep]
            
            del X_train_seq_embeddings, X_val_seq_embeddings, X_test_seq_embeddings  # Immediately delete to free memory
            gc.collect()

            # Load 3Di embeddings
            X_train_3Di = np.load(f'./data/Dataset/embeddings/{Train_file_name_3Di_embed}')['arr_0']
            X_val_3Di = np.load(f'./data/Dataset/embeddings/{Val_file_name_3Di_embed}')['arr_0']
            X_test_3Di = np.load(f'./data/Dataset/embeddings/{Test_file_name_3Di_embed}')['arr_0']

            train_3Di_id_to_keep = list(pd.read_csv(f'./data/Dataset/csv/Train_ids_for_3Di_usage_{pLDDT_threshold}.csv')['order_id'])
            val_3Di_id_to_keep =   list(pd.read_csv(f'./data/Dataset/csv/Val_ids_for_3Di_usage_{pLDDT_threshold}.csv')['order_id'])
            test_3Di_id_to_keep =  list(pd.read_csv(f'./data/Dataset/csv/Test_ids_for_3Di_usage_{pLDDT_threshold}.csv')['order_id'])

            #DEBUG
            # print("Length of X_val_3Di: ", len(X_val_3Di))
            # print("Length of val_3Di_id_to_keep: ", len(val_3Di_id_to_keep))

            # print("Length of X_test_3Di: ", len(X_test_3Di))
            # print("Length of test_3Di_id_to_keep: ", len(test_3Di_id_to_keep))

            train_ids_for_3Di_usage_threshold_0 = list(pd.read_csv(f'./data/Dataset/csv/Train_ids_for_3Di_usage_0.csv')['Domain_id'])
            val_ids_for_3Di_usage_threshold_0 = list(pd.read_csv(f'./data/Dataset/csv/Val_ids_for_3Di_usage_0.csv')['Domain_id'])
            test_ids_for_3Di_usage_threshold_0 = list(pd.read_csv(f'./data/Dataset/csv/Test_ids_for_3Di_usage_0.csv')['Domain_id'])
            
            # val_domain_id_to_fasta_row_id_dict = {id_value: index for index, id_value in enumerate(val_ids_for_3Di_usage_threshold_0)}
            # test_domain_id_to_fasta_row_id_dict = {id_value: index for index, id_value in enumerate(test_ids_for_3Di_usage_threshold_0)}
            # val_3Di_id_to_keep = [val_domain_id_to_fasta_row_id_dict[domain_id] for domain_id in val_ids_for_3Di_usage_threshold_0]
            # test_3Di_id_to_keep = [test_domain_id_to_fasta_row_id_dict[domain_id] for domain_id in test_ids_for_3Di_usage_threshold_0]

            # Get the corresponding 3Di ids to keep
            train_3Di_id_to_keep = [index for id_value, index in domain_id_generator(train_ids_for_3Di_usage_threshold_0) if id_value in train_ids_for_3Di_usage]
            val_3Di_id_to_keep =   [index for id_value, index in domain_id_generator(val_ids_for_3Di_usage_threshold_0) if id_value in val_ids_for_3Di_usage]
            test_3Di_id_to_keep =  [index for id_value, index in domain_id_generator(test_ids_for_3Di_usage_threshold_0) if id_value in test_ids_for_3Di_usage]

            # filter the train embeddings so that only the ones with pLDDT > threshold are kept
            X_train_3Di_filtered = X_train_3Di[train_3Di_id_to_keep]

            # filter test and val embeddings so that only the the ones corresponding to SF present in filtered train set are kept
            X_val_3Di_filtered = X_val_3Di[val_3Di_id_to_keep]
            X_test_3Di_filtered = X_test_3Di[test_3Di_id_to_keep]

            #DEBUG  
            # print("Length of X_val_3Di_filtered: ", len(X_val_3Di_filtered))
            # print("Length of X_test_3Di_filtered: ", len(X_test_3Di_filtered))
            # exit()
            # Load 3Di embeddings
            # X_train_3Di_filtered = np.load(f'./data/Dataset/embeddings/{Train_file_name_3Di_embed}')['arr_0']
            # X_val_3Di_filtered = np.load(f'./data/Dataset/embeddings/{Val_file_name_3Di_embed}')['arr_0']
            # X_test_3Di_filtered = np.load(f'./data/Dataset/embeddings/{Test_file_name_3Di_embed}')['arr_0']

            # Ensure that lengths match before concatenation
            assert len(X_train_seq_embeddings_filtered) == len(X_train_3Di_filtered), "Train sequence and 3Di embeddings must have the same length"
            assert len(X_val_seq_embeddings_filtered) == len(X_val_3Di_filtered), "Val sequence and 3Di embeddings must have the same length"
            assert len(X_test_seq_embeddings_filtered) == len(X_test_3Di_filtered), "Test sequence and 3Di embeddings must have the same length"

            # Concatenate the sequence and 3Di embeddings along the feature axis
            X_train = np.concatenate((X_train_seq_embeddings_filtered, X_train_3Di_filtered), axis=1)
            X_val = np.concatenate((X_val_seq_embeddings_filtered, X_val_3Di_filtered), axis=1)
            X_test = np.concatenate((X_test_seq_embeddings_filtered, X_test_3Di_filtered), axis=1)

            del X_train_seq_embeddings_filtered, X_val_seq_embeddings_filtered, X_test_seq_embeddings_filtered, X_train_3Di_filtered, X_val_3Di_filtered, X_test_3Di_filtered  # Immediately delete to free memory
            gc.collect()

        print("\033[92m \nData Loading done\033[0m")

    return X_train, y_train, X_val, y_val, X_test, y_test


def data_preparation(X_train, y_train, y_val, y_test):
    """Prepares the data for training."""
    
    y_tot = y_train + y_val + y_test
    le = preprocessing.LabelEncoder()
    le.fit(y_tot)

    y_train = np.asarray(le.transform(y_train))
    y_val = np.asarray(le.transform(y_val))
    y_test = np.asarray(le.transform(y_test))

    num_classes = len(np.unique(y_tot))
    print("number of classes: ",num_classes)
    
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    print("\033[92mData preparation done\033[0m")

    return X_train, y_train, y_val, y_test, num_classes


def bm_generator(X_t, y_t, batch_size, num_classes):
    val = 0

    while True:
        X_batch = []
        y_batch = []

        for _ in range(batch_size):

            if val == len(X_t):
                val = 0

            X_batch.append(X_t[val])
            y_enc = np.zeros((num_classes))
            y_enc[y_t[val]] = 1
            y_batch.append(y_enc)
            val += 1

        X_batch = np.asarray(X_batch)
        y_batch = np.asarray(y_batch)

        yield X_batch, y_batch


# Keras NN Model
def create_model(model_name, num_classes, nb_layer_block, dropout, input_type, layer_size):
    """Creates and returns a Keras model based on the specified model name and layer blocks."""
    
    
    if input_type == 'AA+3Di':
        input_shapes = {
            'ProtT5_new': (2048,),
            'ProtT5': (2048,),
            'ProstT5_full': (2048,),
            'ProstT5_half': (2048,),
            'ESM2': (2304,),
            'Ankh_large': (2560,),
            'Ankh_base': (1792,),
            'TM_Vec': (1536,)
        }

    elif input_type == 'AA':
        input_shapes = {
            'ProtT5_new': (1024,),
            'ProtT5': (1024,),
            'ProstT5_full': (1024,),
            'ProstT5_half': (1024,),
            'ESM2': (1280,),
            'Ankh_large': (1536,),
            'Ankh_base': (768,),
            'TM_Vec': (512,)
        }
    
    # For 3Di only inputs, only the ProstT5 models are available
    elif input_type == '3Di':
        input_shapes = {
            'ProstT5_full': (1024,),
            'ProstT5_half': (1024,)
        }
    
    else:
        raise ValueError("Invalid input type, must be 'AA', '3Di', or 'AA+3Di'")

    if model_name not in input_shapes:
        raise ValueError("Invalid model name")
    
    input_shape = input_shapes[model_name]
    input_ = Input(shape=input_shape)
    x = input_
    
    for _ in range(nb_layer_block):
        x = Dense(layer_size, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(x)
        x = LeakyReLU(alpha=0.05)(x)
        x = BatchNormalization()(x)
        if dropout:
            x = Dropout(dropout)(x)
    
    out = Dense(num_classes, activation = 'softmax', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(x)
    classifier = Model(input_, out)

    return classifier

# @profile
def train_model(model_name, num_classes, X_train, y_train, X_val, y_val, input_type, nb_layer_block, dropout, layer_size, pLDDT_threshold):
    """Trains the model."""

    print("\033[92mModel training \033[0m")

    base_model_path = f'saved_models/ann_{model_name}'
    base_loss_path = f'results/Loss/ann_{model_name}'

    if dropout:

        save_model_path = f'{base_model_path}_{nb_layer_block}_blocks_dropout_{dropout}_layer_size_{layer_size}_pLDDT_{pLDDT_threshold}.h5'
        save_loss_path = f'{base_loss_path}_{nb_layer_block}_blocks_dropout_{dropout}_layer_size_{layer_size}_pLDDT_{pLDDT_threshold}.png'
    else:
        
        save_model_path = f'{base_model_path}_{nb_layer_block}_blocks_no_dropout_layer_size_{layer_size}_pLDDT_{pLDDT_threshold}.h5'
        save_loss_path = f'{base_loss_path}_{nb_layer_block}_blocks_no_dropout_layer_size_{layer_size}_pLDDT_{pLDDT_threshold}.png'
    
    if input_type == '3Di':

        save_model_path  = save_model_path.replace('.h5', '_3Di.h5')
        save_loss_path = save_loss_path.replace('.png', '_3Di.png')
    
    if input_type == 'AA+3Di':
            
        save_model_path  = save_model_path.replace('.h5', '_AA+3Di.h5')
        save_loss_path = save_loss_path.replace('.png', '_AA+3Di.png')

    num_epochs = 200
    batch_size = 4096


    with tf.device('/gpu:0'):
        # model
        model = create_model(model_name, num_classes, nb_layer_block, dropout, input_type, layer_size)

        # adam optimizer
        opt = keras.optimizers.Adam(learning_rate = 1e-5)
        model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=['accuracy'])

        # callbacks
        mcp_save = keras.callbacks.ModelCheckpoint(save_model_path, save_best_only=True, monitor='val_accuracy', verbose=1)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30)
        callbacks_list = [reduce_lr, mcp_save, early_stop]

        # test and train generators
        train_gen = bm_generator(X_train, y_train, batch_size, num_classes)
        val_gen = bm_generator(X_val, y_val, batch_size, num_classes)
        # test_gen = bm_generator(X_test, y_test, batch_size, num_classes)
        history = model.fit(train_gen, epochs = num_epochs, steps_per_epoch = math.ceil(len(X_train)/(batch_size)), verbose=1, validation_data = val_gen, validation_steps = len(X_val)/batch_size, workers = 0, shuffle = True, callbacks = callbacks_list)

        # Plot the training and validation loss
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.figure()
        plt.plot(epochs, loss, 'b-', label='Training loss', linewidth=1)
        plt.plot(epochs, val_loss, 'r-', label='Validation loss', linewidth=1)
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_loss_path)  # Save the plot
        plt.close()

        print("\033[92mModel training done\033[0m")

        del model, history, loss, val_loss, epochs  # Immediately delete to free memory
        gc.collect()

def save_confusion_matrix(y_test, y_pred, confusion_matrix_path):

    # print("Confusion Matrix")
    matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
    size = matrix.shape[0]
    print(f"Size of confusion matrix: {size}")

    # Find the indices and values of the non-zero elements
    non_zero_indices = np.nonzero(matrix)
    non_zero_values = matrix[non_zero_indices]

    # Combine row indices, column indices, and values into a single array
    non_zero_data = np.column_stack((non_zero_indices[0], non_zero_indices[1], non_zero_values))

    # Save the non-zero entries to a CSV file
    np.savetxt(f'{confusion_matrix_path}.csv', non_zero_data, delimiter=",", fmt='%d')

    # # Read the sparse confusion matrix CSV file
    # df = pd.read_csv(confusion_matrix_path, header=None, names=['row', 'col', 'value'])
    
    
    # # Create an empty confusion matrix
    # confusion_matrix = np.zeros((size, size), dtype=int)
    
    # # Fill the confusion matrix
    # for _, row in df.iterrows():
    #     confusion_matrix[row['row'], row['col']] = row['value']
    
    
    # Create a custom color map that makes zero cells white
    cmap = plt.cm.viridis
    cmap.set_under('white')
    
    # Plot the heatmap with logarithmic scaling
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=False, fmt="d", cmap=cmap, vmin=0.01, cbar_kws={'label': 'Log Scale'})
    
    # Create a purple to yellow color map for the annotations
    norm = mcolors.Normalize(vmin=matrix.min(), vmax=matrix.max())
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
    
    # Emphasize non-zero values by adding colored annotations
    for i in range(size):
        for j in range(size):
            if matrix[i, j] > 0:
                plt.text(j + 0.5, i + 0.5, matrix[i, j],
                         horizontalalignment='center',
                         verticalalignment='center',
                         fontsize=6, color=sm.to_rgba(matrix[i, j]))
    
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix Heatmap (Log Scale)')
    plt.savefig(f'{confusion_matrix_path}.png', bbox_inches='tight')
    plt.close()

# @profile
def evaluate_model(model_name, X_val, y_val, X_test, y_test, nb_layer_block, dropout, input_type, layer_size, pLDDT_threshold):
    """Evaluates the trained model."""

    print("\033[92mModel evaluation \033[0m")
    
    if dropout:
        base_model_path = f'saved_models/ann_{model_name}'
        base_classification_report_path = f'results/classification_report/CR_ANN_{model_name}'
        base_confusion_matrix_path = f'results/confusion_matrices/{model_name}'

        model_path = f'{base_model_path}_{nb_layer_block}_blocks_dropout_{dropout}_layer_size_{layer_size}_pLDDT_{pLDDT_threshold}.h5'


        classification_report_path = f'{base_classification_report_path}_{nb_layer_block}_blocks_dropout_{dropout}_layer_size_{layer_size}_pLDDT_{pLDDT_threshold}.csv'
        confusion_matrix_path = f'{base_confusion_matrix_path}_{nb_layer_block}_blocks_dropout_{dropout}_layer_size_{layer_size}_pLDDT_{pLDDT_threshold}'
        results_file = f'./results/perf_metrics/ann_{model_name}_{nb_layer_block}_blocks_dropout_{dropout}_layer_size_{layer_size}_pLDDT_{pLDDT_threshold}.csv'
    else:
        base_model_path = f'saved_models/ann_{model_name}'
        base_classification_report_path = f'results/classification_report/CR_ANN_{model_name}'
        base_confusion_matrix_path = f'results/confusion_matrices/{model_name}'

        model_path = f'{base_model_path}_{nb_layer_block}_blocks_no_dropout_layer_size_{layer_size}.h5'

        classification_report_path = f'{base_classification_report_path}_{nb_layer_block}_blocks_no_dropout_layer_size_{layer_size}_pLDDT_{pLDDT_threshold}.csv'
        confusion_matrix_path = f'{base_confusion_matrix_path}_{nb_layer_block}_blocks_no_dropout_layer_size_{layer_size}_pLDDT_{pLDDT_threshold}'
        results_file = f'./results/perf_metrics/ann_{model_name}_{nb_layer_block}_blocks_no_dropout_layer_size_{layer_size}_pLDDT_{pLDDT_threshold}.csv'
    
    if input_type == '3Di':
            
        model_path = model_path.replace('.h5', '_3Di.h5')
        classification_report_path = classification_report_path.replace('.csv', '_3Di.csv')
        confusion_matrix_path = confusion_matrix_path.replace('.png', '_3Di.png')
        results_file = results_file.replace('.csv', '_3Di.csv')
    
    if input_type == 'AA+3Di':
                
        model_path = model_path.replace('.h5', '_AA+3Di.h5')
        classification_report_path = classification_report_path.replace('.csv', '_AA+3Di.csv')
        confusion_matrix_path = confusion_matrix_path.replace('.png', '_AA+3Di.png')
        results_file = results_file.replace('.csv', '_AA+3Di.csv')

    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])

        try:
            model = load_model(model_path)
        except:
            raise ValueError(f"Model file '{model_path}' not found, make sure you have trained the model first, to train the model use the --do_training flag")
            
        with tf.device('/gpu:0'):
            writer.writerow(["Validation", ""])
            y_pred_val = model.predict(X_val)
            f1_score_val = f1_score(y_val, y_pred_val.argmax(axis=1), average='weighted')
            acc_score_val = accuracy_score(y_val, y_pred_val.argmax(axis=1))
            writer.writerow(["Validation F1 Score", f1_score_val])
            writer.writerow(["Validation Accuracy Score", acc_score_val])

            writer.writerow(["Regular Testing", ""])
            y_pred_test = model.predict(X_test)
            f1_score_test = f1_score(y_test, y_pred_test.argmax(axis=1), average='macro')
            acc_score_test = accuracy_score(y_test, y_pred_test.argmax(axis=1))
            mcc_score = matthews_corrcoef(y_test, y_pred_test.argmax(axis=1))
            bal_acc = balanced_accuracy_score(y_test, y_pred_test.argmax(axis=1))
            writer.writerow(["Test F1 Score", f1_score_test])
            writer.writerow(["Test Accuracy Score", acc_score_test])
            writer.writerow(["Test MCC", mcc_score])
            writer.writerow(["Test Balanced Accuracy", bal_acc])

            # Save the test F1 score in a DataFrame
            df_results = pd.DataFrame({
                'Model': [model_name],
                'Nb_Layer_Block': [nb_layer_block],
                'Dropout': [dropout],
                'Input_Type': [input_type],
                'Layer_size': [layer_size],
                'pLDDT_threshold': [pLDDT_threshold],
                'F1_Score': [f1_score_test]
            })
            df_results_path = './results/perf_dataframe.csv'

            if os.path.exists(df_results_path):
                df_existing = pd.read_csv(df_results_path)
                # Create a mask to find rows that match the current combination of parameters
                mask = (
                    (df_existing['Model'].astype(type(model_name)) == model_name) &
                    (df_existing['Nb_Layer_Block'].astype(type(nb_layer_block)) == nb_layer_block) &
                    (df_existing['Dropout'].astype(type(dropout)) == dropout) &
                    (df_existing['Input_Type'].astype(type(input_type)) == input_type) &
                    (df_existing['Layer_size'].astype(type(layer_size)) == layer_size) &
                    (df_existing['pLDDT_threshold'].astype(type(pLDDT_threshold)) == pLDDT_threshold) 

                )
                # If a matching row exists, update its F1_Score
                if mask.any():
                    df_existing.loc[mask, 'F1_Score'] = f1_score_test
                else:
                    df_existing = pd.concat([df_existing, df_results], ignore_index=True)
                df_combined = df_existing
            else:
                df_combined = df_results

            df_combined.to_csv(df_results_path, index=False)

            writer.writerow(["Bootstrapping Results", ""])
            num_iter = 1000
            f1_arr = []
            acc_arr = []
            mcc_arr = []
            bal_arr = []

            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")

            print("Evaluation with bootstrapping")
            for it in tqdm(range(num_iter)):
                X_test_re, y_test_re = resample(X_test, y_test, n_samples=len(y_test), random_state=it)
                y_pred_test_re = model.predict(X_test_re, verbose=0)
                f1_arr.append(f1_score(y_test_re, y_pred_test_re.argmax(axis=1), average='macro'))
                acc_arr.append(accuracy_score(y_test_re, y_pred_test_re.argmax(axis=1)))
                mcc_arr.append(matthews_corrcoef(y_test_re, y_pred_test_re.argmax(axis=1)))
                bal_arr.append(balanced_accuracy_score(y_test_re, y_pred_test_re.argmax(axis=1)))

                del X_test_re, y_test_re, y_pred_test_re  # Immediately delete to free memory
                gc.collect()

            writer.writerow(["Accuracy ", np.mean(acc_arr)])
            writer.writerow(["F1-Score", np.mean(f1_arr)])
            writer.writerow(["MCC", np.mean(mcc_arr)])
            writer.writerow(["Balanced Accuracy", np.mean(bal_arr)])
            
            warnings.filterwarnings("default")

            y_pred = model.predict(X_test)
            cr = classification_report(y_test, y_pred.argmax(axis=1), output_dict=True, zero_division=1)
            df = pd.DataFrame(cr).transpose()
            df.to_csv(classification_report_path)

            save_confusion_matrix(y_test, y_pred, confusion_matrix_path)

            print("\033[92mModel evaluation done\033[0m")

            # free all memory
            del model, y_pred_val, y_pred_test, y_pred, cr, df, df_results, df_existing, df_combined, f1_arr, acc_arr, mcc_arr, bal_arr, num_iter
            gc.collect()

            # DEBUG
            print(f"f1_score_test: {f1_score_test}")


def create_arg_parser():
    """Creates and returns the ArgumentParser object."""

    parser = argparse.ArgumentParser(description=
                        'Run training and evaluation for one or all models')
    parser.add_argument('--do_training', type=int, 
                        default=1, 
                        help="Whether to actually train and test the model or just test the saved model, put 0 to skip training, 1 to train")
    
    parser.add_argument('--dropout', type=str, 
                        default='0.3', 
                        help="Whether to use dropout in the model layers or not, and if so, what value, put 0 to not use dropout, a value between 0 and 1 excluded to use dropout with this value, and 'all' to test every values in [0,0.2,0.4]")
    
    parser.add_argument('--layer_size', type=str, 
                        default='2048', 
                        help="To choose the size of the dens layers in the classifier, choose a values in [64,128,256,512, 1024, 2048] or 'all' to test every values")


    parser.add_argument('--nb_layer_block', type=str, 
                        default='two',
                        help="Number of layer block {Dense, LeakyReLU, BatchNormalization, Dropout} in the classifier. Choose between 'one', 'two', 'three', or 'all'")
    
    parser.add_argument('--model', type=str, 
                        default='ProstT5_full', 
                        help="What model to use between ProtT5, ProtT5_new, ESM2, Ankh_large, Ankh_base, ProstT5_full, ProstT5_half, TM_Vec, or all")
    
    parser.add_argument('--classifier_input', type=str, 
                        default='AA+3Di', 
                        help="Whether to use Amino Acids, 3Di or a concatenation of both to train the classifier, put 'AA' for Amino Acids, '3Di' for 3Di, 'AA+3Di' for the concatenation of the 2")
    
    
    parser.add_argument('--pLDDT_threshold', type=int, 
                        default='0', 
                        help="Threshold for pLDDT to filter the trining set of 3Di from hight structure quality, choose from [0, 4, 14, 24, 34, 44, 54, 64, 74, 84] (only useful for 3Di input)")
    
    return parser

def main():

    parser = create_arg_parser()
    args = parser.parse_args()

    model_name = args.model
    do_training = args.do_training
    nb_layer_block = args.nb_layer_block
    input_type = args.classifier_input
    pLDDT_threshold = args.pLDDT_threshold

    if input_type == 'AA':
        pLDDT_threshold = 0.0

    if (input_type == '3Di' or input_type == 'AA+3Di') and model_name == 'ProtT5':
        raise ValueError("Please use 'ProtT5_new' instead of 'ProtT5' when using classifier_input '3Di' or 'AA+3Di'")

    do_training = False if int(args.do_training) == 0 else True
    dropout_tag = float(args.dropout) if args.dropout != 'all' else args.dropout
    layer_size_tag = int(args.layer_size) if args.layer_size != 'all' else args.layer_size

    if dropout_tag == 'all':
        dropout_values = [0, 0.2, 0.4]
    else:
        dropout_values = [dropout_tag]

    if layer_size_tag == 'all':
        layer_size_values = [64, 128, 256, 512]
    else:
        layer_size_values = [layer_size_tag]

    all_model_names = ['ProtT5', 'ProtT5_new', 'ESM2', 'Ankh_large', 'Ankh_base', 'ProstT5_full', 'ProstT5_half', 'TM_Vec']

    if model_name == 'all':
        for model_name in tqdm(all_model_names, desc="Models"):
            if nb_layer_block == 'all':
                for nb_layer_block in tqdm(['one', 'two', 'three'], desc="Layer Blocks", leave=False):
                    for dropout in tqdm(dropout_values, desc="Dropout Values", leave=False):
                        for layer_size in tqdm(layer_size_values, desc="Layer Sizes", leave=False):
                            X_train, y_train, X_val, y_val, X_test, y_test = load_data(model_name, input_type, pLDDT_threshold)
                            X_train, y_train, y_val, y_test, num_classes = data_preparation(X_train, y_train, y_val, y_test)
                            if do_training:
                                train_model(model_name, num_classes, X_train, y_train, X_val, y_val, input_type, nb_layer_block_dict[nb_layer_block], dropout, layer_size, pLDDT_threshold)
                            evaluate_model(model_name, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout, input_type, layer_size, pLDDT_threshold)
                            # Clear memory after evaluation
                            del X_train, y_train, X_val, y_val, X_test, y_test
                            gc.collect()
            else:
                for dropout in tqdm(dropout_values, desc="Dropout Values", leave=False):
                    for layer_size in tqdm(layer_size_values, desc="Layer Sizes", leave=False):
                        X_train, y_train, X_val, y_val, X_test, y_test = load_data(model_name, input_type, pLDDT_threshold)
                        X_train, y_train, y_val, y_test, num_classes = data_preparation(X_train, y_train, y_val, y_test)
                        if do_training:
                            train_model(model_name, num_classes, X_train, y_train, X_val, y_val, input_type, nb_layer_block_dict[nb_layer_block], dropout, layer_size, pLDDT_threshold)
                        evaluate_model(model_name, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout, input_type, layer_size, pLDDT_threshold)
                        # Clear memory after evaluation
                        del X_train, y_train, X_val, y_val, X_test, y_test
                        gc.collect()
    else:
        if nb_layer_block == 'all':
            for nb_layer_block in tqdm(['one', 'two', 'three'], desc="Layer Blocks", leave=False):
                for dropout in tqdm(dropout_values, desc="Dropout Values", leave=False):
                    for layer_size in tqdm(layer_size_values, desc="Layer Sizes", leave=False):
                        X_train, y_train, X_val, y_val, X_test, y_test = load_data(model_name, input_type, pLDDT_threshold)
                        X_train, y_train, y_val, y_test, num_classes = data_preparation(X_train, y_train, y_val, y_test)
                        if do_training:
                            train_model(model_name, num_classes, X_train, y_train, X_val, y_val, input_type, nb_layer_block_dict[nb_layer_block], dropout, layer_size, pLDDT_threshold)
                        evaluate_model(model_name, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout, input_type, layer_size, pLDDT_threshold)
                        # Clear memory after evaluation
                        del X_train, y_train, X_val, y_val, X_test, y_test
                        gc.collect()
        else:
            for dropout in tqdm(dropout_values, desc="Dropout Values", leave=False):
                for layer_size in tqdm(layer_size_values, desc="Layer Sizes", leave=False):
                    X_train, y_train, X_val, y_val, X_test, y_test = load_data(model_name, input_type, pLDDT_threshold)
                    X_train, y_train, y_val, y_test, num_classes = data_preparation(X_train, y_train, y_val, y_test)
                    if do_training:
                        train_model(model_name, num_classes, X_train, y_train, X_val, y_val, input_type, nb_layer_block_dict[nb_layer_block], dropout, layer_size, pLDDT_threshold)
                    evaluate_model(model_name, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout, input_type, layer_size, pLDDT_threshold)
                    # Clear memory after evaluation
                    del X_train, y_train, X_val, y_val, X_test, y_test
                    gc.collect()
                                   

if __name__ == '__main__':
    main()