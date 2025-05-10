# This code is used to train the ANN model for CATH annotation prediction. To do so, many hyperparameters can be tuned like the model which computed the embeddings, the structure of the classifier trained on these embeddings, or different filter on the training dataset. 

# This code is based on Vam-sin version of CATHe ann training: ann_t5.py
# Author: Vamsi Nallapareddy
# Source: https://github.com/vam-sin/CATHe/tree/main (from which this project was forked)
# License: MIT License
# Modifications: Added new models, new embeddings, new filtering criteria, and new hyperparameters to train the model.

# Part of the code from: 
# Author: Rostlab
# Source: https://huggingface.co/Rostlab/ProstT5

# ANSI escape code for colored text
yellow = '\033[93m'
green = '\033[92m'
reset = '\033[0m'
red = '\033[91m'
orange = '\033[33m'

print(f'{green}model training code running (ann_all_new_models.py){reset}')


num_epochs = 200
batch_size = 4096

import sys
import os

# Check if a virtual environment is active
if not hasattr(sys, 'base_prefix') or sys.base_prefix == sys.prefix:
    raise EnvironmentError(f'{red}No virtual environment is activated. Please activate venv_2 to run this code. See ReadMe for more details.{reset}')

# Get the name of the activated virtual environment
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path is None:
    raise EnvironmentError(f'{red}Error, venv path is none. Please activate the venv_2. See ReadMe for more details.{reset}')

# Check if the activated virtual environment is venv_2
venv_name = os.path.basename(venv_path)
if venv_name != 'venv_2':
    raise EnvironmentError(f'{red}The activated virtual environment is {venv_name}, not venv_2. However venv_2 must be activated to run this code. See ReadMe for more details.{reset}')

print(f'{green} ann_all_new_models.py: library imports in progress, may take a long time{reset}')

import argparse
import pandas as pd 
import numpy as np 
import gc
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras  
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU management
from tensorflow.compat.v1 import ConfigProto

tf.keras.backend.clear_session()
config = ConfigProto()
config.gpu_options.allow_growth = True

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print("Using GPU:", gpus[0])
    except RuntimeError as e:
        print(e)

# Test GPU usage by Tensorflow
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print(f"{orange}GPU is being used!{reset}")
else:
    print(f"{orange}GPU is not being used, working on CPU.{reset}")



def load_ids_to_keep(pLDDT_threshold, only_50_largest_SF, support_threshold):
    '''
    Load the domain ids to keep based on the filtering criteria.

    Args:
        pLDDT_threshold (int): The pLDDT threshold used to filter the domains.
        only_50_largest_SF (bool): Whether to keep only the 50 largest superfamilies.
        support_threshold (int): The support threshold used to filter the domains.
    
    Returns:
        train_ids_to_keep (list): List of domain ids to keep for the training set.
        val_ids_to_keep (list): List of domain ids to keep for the validation set.
        test_ids_to_keep (list): List of domain ids to keep for the test set.
    
    '''
    # Load list of domain ids to keep, depending on the filtering criteria
    if only_50_largest_SF:
        train_ids_to_keep = list(pd.read_csv(f'./data/Dataset/csv/Train_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_top_50_SF.csv')['Domain_id'])
        val_ids_to_keep = list(pd.read_csv(f'./data/Dataset/csv/Val_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_top_50_SF.csv')['Domain_id'])
        test_ids_to_keep = list(pd.read_csv(f'./data/Dataset/csv/Test_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_top_50_SF.csv')['Domain_id'])
    
    elif support_threshold!=0:
        train_ids_to_keep = list(pd.read_csv(f'./data/Dataset/csv/Train_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_support_threshold_{support_threshold}.csv')['Domain_id'])
        val_ids_to_keep = list(pd.read_csv(f'./data/Dataset/csv/Val_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_support_threshold_{support_threshold}.csv')['Domain_id'])
        test_ids_to_keep = list(pd.read_csv(f'./data/Dataset/csv/Test_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_support_threshold_{support_threshold}.csv')['Domain_id'])
    else:
        train_ids_to_keep = list(pd.read_csv(f'./data/Dataset/csv/Train_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}.csv')['Domain_id'])
        val_ids_to_keep = list(pd.read_csv(f'./data/Dataset/csv/Val_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}.csv')['Domain_id'])
        test_ids_to_keep = list(pd.read_csv(f'./data/Dataset/csv/Test_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}.csv')['Domain_id'])
    
    return train_ids_to_keep, val_ids_to_keep, test_ids_to_keep

def load_and_filter_3Di_embeddings(Train_file_name_3Di_embed, Val_file_name_3Di_embed, Test_file_name_3Di_embed, train_ids_to_keep, val_ids_to_keep, test_ids_to_keep):
    '''
    Load and filter the 3Di embeddings based on the domain ids to keep.

    Args:
        Train_file_name_3Di_embed (str): The name of the file containing the training 3Di embeddings.
        Val_file_name_3Di_embed (str): The name of the file containing the validation 3Di embeddings.
        Test_file_name_3Di_embed (str): The name of the file containing the test 3Di embeddings.
        train_ids_to_keep (list): List of domain ids to keep for the training set.
        val_ids_to_keep (list): List of domain ids to keep for the validation set.
        test_ids_to_keep (list): List of domain ids to keep for the test set.

    Returns:
        train_embeddings_to_keep_3Di (list): List of 3Di embeddings to keep for the training set.
        val_embeddings_to_keep_3Di (list): List of 3Di embeddings to keep for the validation set.
        test_embeddings_to_keep_3Di (list): List of 3Di embeddings to keep for the test set.
    
    '''
    
    # load 3Di embedding df
    X_train_3Di_df = np.load(f'./data/Dataset/embeddings/{Train_file_name_3Di_embed}')
    X_val_3Di_df = np.load(f'./data/Dataset/embeddings/{Val_file_name_3Di_embed}')
    X_test_3Di_df = np.load(f'./data/Dataset/embeddings/{Test_file_name_3Di_embed}')

    X_train_embedding_dict_3Di = dict(zip(X_train_3Di_df['keys'], X_train_3Di_df['embeddings']))
    X_val_embedding_dict_3Di = dict(zip(X_val_3Di_df['keys'], X_val_3Di_df['embeddings']))
    X_test_embedding_dict_3Di = dict(zip(X_test_3Di_df['keys'], X_test_3Di_df['embeddings']))

    # 3Di embedding filtering
    # For train embeddings
    train_embeddings_to_keep_3Di = []
    for domain_id in train_ids_to_keep:
        if domain_id not in X_train_embedding_dict_3Di:
            raise KeyError(f'Train domain ID {domain_id} not found in the train embeddings dictionary!')
        train_embeddings_to_keep_3Di.append(X_train_embedding_dict_3Di[domain_id])
    
    # For validation embeddings
    val_embeddings_to_keep_3Di = []
    for domain_id in val_ids_to_keep:
        if domain_id not in X_val_embedding_dict_3Di:
            raise KeyError(f'Validation domain ID {domain_id} not found in the validation embeddings dictionary!')
        val_embeddings_to_keep_3Di.append(X_val_embedding_dict_3Di[domain_id])
    
    # For test embeddings
    test_embeddings_to_keep_3Di = []
    for domain_id in test_ids_to_keep:
        if domain_id not in X_test_embedding_dict_3Di:
            raise KeyError(f'Test domain ID {domain_id} not found in the test embeddings dictionary!')
        test_embeddings_to_keep_3Di.append(X_test_embedding_dict_3Di[domain_id])
    
    return train_embeddings_to_keep_3Di, val_embeddings_to_keep_3Di, test_embeddings_to_keep_3Di


def load_data(model_name, input_type, pLDDT_threshold, only_50_largest_SF, support_threshold):
    '''
    Load the filtered data for model training, validation and testing, based on the model name input type and different filtering criteria.

    Args:
        model_name (str): The name of the model used to compute the embeddings.
        input_type (str): The type of input data used for training the model.
        pLDDT_threshold (int): The pLDDT threshold used to filter the domains.
        only_50_largest_SF (bool): Whether to keep only the 50 largest superfamilies.
        support_threshold (int): The support threshold used to filter the domains.

    Returns:
        X_train (np.ndarray): The training data.
        y_train (list): The training labels.
        X_val (np.ndarray): The validation data.
        y_val (list): The validation labels.
        X_test (np.ndarray): The test data.
        y_test (list): The test labels.
    '''

    if model_name == 'ProtT5':
        # For ProtT5, the former code and embeddings datasets are used here, see ./src/model_building/models/ProtT5/ann_ProtT5.py

        # using the original CATHe datasets for ProtT5
        ds_train = pd.read_csv('./data/Dataset/annotations/Y_Train_SF.csv')
        y_train = list(ds_train['SF'])

        filename = './data/Dataset/embeddings/SF_Train_ProtT5.npz'
        X_train = np.load(filename)['arr_0']
        filename = './data/Dataset/embeddings/Other Class/Other_Train.npz'
        X_train_other = np.load(filename)['arr_0']

        X_train = np.concatenate((X_train, X_train_other), axis=0)

        for _ in range(len(X_train_other)):
            y_train.append('other')

        # val
        ds_val = pd.read_csv('./data/Dataset/annotations/Y_Val_SF.csv')
        y_val = list(ds_val['SF'])

        filename = './data/Dataset/embeddings/SF_Val_ProtT5.npz'
        X_val = np.load(filename)['arr_0']

        filename = './data/Dataset/embeddings/Other Class/Other_Val.npz'
        X_val_other = np.load(filename)['arr_0']

        X_val = np.concatenate((X_val, X_val_other), axis=0)

        for _ in range(len(X_val_other)):
            y_val.append('other')

        # test
        ds_test = pd.read_csv('./data/Dataset/annotations/Y_Test_SF.csv')
        y_test = list(ds_test['SF'])

        filename = './data/Dataset/embeddings/SF_Test_ProtT5.npz'
        X_test = np.load(filename)['arr_0']

        filename = './data/Dataset/embeddings/Other Class/Other_Test.npz'
        X_test_other = np.load(filename)['arr_0']

        X_test = np.concatenate((X_test, X_test_other), axis=0)

        for _ in range(len(X_test_other)):
            y_test.append('other')
    
    else:
        # For new models, new embeddings are used.

        # labels y_train, y_val, y_test

        df_train = pd.read_csv('./data/Dataset/csv/Train.csv')
        df_val = pd.read_csv('./data/Dataset/csv/Val.csv')
        df_test = pd.read_csv('./data/Dataset/csv/Test.csv')

        test_perf_AA_only_with_3D_structure_filters = False
        
        if input_type == '3Di' or input_type == 'AA+3Di' or test_perf_AA_only_with_3D_structure_filters:
            # Load the domain IDs corresponding to the filter criteria

            if only_50_largest_SF:
                train_ids_for_3Di_usage = set(pd.read_csv(f'./data/Dataset/csv/Train_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_top_50_SF.csv')['Domain_id'])
                val_ids_for_3Di_usage = set(pd.read_csv(f'./data/Dataset/csv/Val_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_top_50_SF.csv')['Domain_id'])
                test_ids_for_3Di_usage = set(pd.read_csv(f'./data/Dataset/csv/Test_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_top_50_SF.csv')['Domain_id'])

            elif support_threshold!=0:
                train_ids_for_3Di_usage = set(pd.read_csv(f'./data/Dataset/csv/Train_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_support_threshold_{support_threshold}.csv')['Domain_id'])
                val_ids_for_3Di_usage = set(pd.read_csv(f'./data/Dataset/csv/Val_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_support_threshold_{support_threshold}.csv')['Domain_id'])
                test_ids_for_3Di_usage = set(pd.read_csv(f'./data/Dataset/csv/Test_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}_support_threshold_{support_threshold}.csv')['Domain_id'])
            else:
                train_ids_for_3Di_usage = set(pd.read_csv(f'./data/Dataset/csv/Train_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}.csv')['Domain_id'])
                val_ids_for_3Di_usage = set(pd.read_csv(f'./data/Dataset/csv/Val_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}.csv')['Domain_id'])
                test_ids_for_3Di_usage = set(pd.read_csv(f'./data/Dataset/csv/Test_ids_for_3Di_usage_pLDDT_threshold_{pLDDT_threshold}.csv')['Domain_id'])

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

        if input_type == 'AA':
            if model_name not in prot_sequence_embeddings_paths:
                raise ValueError('Invalid model name, choose between ProtT5, ProtT5_new, ESM2, Ankh_large, Ankh_base, ProstT5_full, ProstT5_half, TM_Vec')
        elif input_type == '3Di':
            if model_name != 'ProstT5_full':
                raise ValueError('Invalid model name, if input type is 3Di, only ProstT5_full is available')
        
        Train_file_name_seq_embed, Val_file_name_seq_embed, Test_file_name_seq_embed = prot_sequence_embeddings_paths[model_name]

        if input_type == '3Di' or input_type == 'AA+3Di':
            Train_file_name_3Di_embed = 'Train_ProstT5_full_per_protein_3Di.npz'
            Val_file_name_3Di_embed = 'Val_ProstT5_full_per_protein_3Di.npz'
            Test_file_name_3Di_embed = 'Test_ProstT5_full_per_protein_3Di.npz'

        if input_type == 'AA':

            # Load AA sequence embeddings
            X_train_seq_embeddings_df = np.load(f'./data/Dataset/embeddings/{Train_file_name_seq_embed}')
            X_val_seq_embeddings_df = np.load(f'./data/Dataset/embeddings/{Val_file_name_seq_embed}')
            X_test_seq_embeddings_df = np.load(f'./data/Dataset/embeddings/{Test_file_name_seq_embed}')
            
            X_train = np.array(X_train_seq_embeddings_df['embeddings'])
            X_val = np.array(X_val_seq_embeddings_df['embeddings'])
            X_test = np.array(X_test_seq_embeddings_df['embeddings'])
        
        if input_type == '3Di':
            # Load 3Di embeddings

            # Get doman ids of the sequences to keep
            train_ids_to_keep, val_ids_to_keep, test_ids_to_keep = load_ids_to_keep(pLDDT_threshold, only_50_largest_SF, support_threshold)
            
            # Load and filter 3Di embeddings based on the domain ids to keep
            train_embeddings_to_keep_3Di, val_embeddings_to_keep_3Di, test_embeddings_to_keep_3Di = load_and_filter_3Di_embeddings(Train_file_name_3Di_embed, Val_file_name_3Di_embed, Test_file_name_3Di_embed, train_ids_to_keep, val_ids_to_keep, test_ids_to_keep)

            X_train = np.array(train_embeddings_to_keep_3Di)
            X_val = np.array(val_embeddings_to_keep_3Di)
            X_test = np.array(test_embeddings_to_keep_3Di)
        
        if input_type == 'AA+3Di':


            # Load sequence embeddings
            X_train_seq_embeddings_df = np.load(f'./data/Dataset/embeddings/{Train_file_name_seq_embed}')
            X_val_seq_embeddings_df = np.load(f'./data/Dataset/embeddings/{Val_file_name_seq_embed}')
            X_test_seq_embeddings_df = np.load(f'./data/Dataset/embeddings/{Test_file_name_seq_embed}')

            X_train_embedding_dict_AA = dict(zip(X_train_seq_embeddings_df['keys'], X_train_seq_embeddings_df['embeddings']))
            X_val_embedding_dict_AA = dict(zip(X_val_seq_embeddings_df['keys'], X_val_seq_embeddings_df['embeddings']))
            X_test_embedding_dict_AA = dict(zip(X_test_seq_embeddings_df['keys'], X_test_seq_embeddings_df['embeddings']))

            # Load list of domain ids to keep
            train_ids_to_keep, val_ids_to_keep, test_ids_to_keep = load_ids_to_keep(pLDDT_threshold, only_50_largest_SF, support_threshold)

            # AA seq embedding filtering
            # For train embeddings
            train_embeddings_to_keep_AA = []
            for domain_id in train_ids_to_keep:
                if domain_id not in X_train_embedding_dict_AA:
                    raise KeyError(f'Train domain ID {domain_id} not found in the train embeddings dictionary!')
                train_embeddings_to_keep_AA.append(X_train_embedding_dict_AA[domain_id])

            # For validation embeddings
            val_embeddings_to_keep_AA = []
            for domain_id in val_ids_to_keep:
                if domain_id not in X_val_embedding_dict_AA:
                    raise KeyError(f'Validation domain ID {domain_id} not found in the validation embeddings dictionary!')
                val_embeddings_to_keep_AA.append(X_val_embedding_dict_AA[domain_id])

            # For test embeddings
            test_embeddings_to_keep_AA = []
            for domain_id in test_ids_to_keep:
                if domain_id not in X_test_embedding_dict_AA:
                    raise KeyError(f'Test domain ID {domain_id} not found in the test embeddings dictionary!')
                test_embeddings_to_keep_AA.append(X_test_embedding_dict_AA[domain_id])
            
            # Free memory by deleting the dictionaries
            del X_train_embedding_dict_AA
            del X_val_embedding_dict_AA
            del X_test_embedding_dict_AA

            # Force the garbage collector to run
            gc.collect()

            # Load and filter 3Di embeddings based on the domain ids to keep
            train_embeddings_to_keep_3Di, val_embeddings_to_keep_3Di, test_embeddings_to_keep_3Di = load_and_filter_3Di_embeddings(Train_file_name_3Di_embed, Val_file_name_3Di_embed, Test_file_name_3Di_embed, train_ids_to_keep, val_ids_to_keep, test_ids_to_keep)

            # Ensure that lengths match before concatenation
            assert len(train_embeddings_to_keep_AA) == len(train_embeddings_to_keep_3Di), 'Train sequence and 3Di embeddings must have the same length'
            assert len(val_embeddings_to_keep_AA) == len(val_embeddings_to_keep_3Di), 'Val sequence and 3Di embeddings must have the same length'
            assert len(test_embeddings_to_keep_AA) == len(test_embeddings_to_keep_3Di), 'Test sequence and 3Di embeddings must have the same length'

            # Concatenate the sequence and 3Di embeddings along the feature axis
            X_train = np.concatenate((train_embeddings_to_keep_AA, train_embeddings_to_keep_3Di), axis=1)
            X_val = np.concatenate((val_embeddings_to_keep_AA, val_embeddings_to_keep_3Di), axis=1)
            X_test = np.concatenate((test_embeddings_to_keep_AA, test_embeddings_to_keep_3Di), axis=1)

            del train_embeddings_to_keep_3Di, val_embeddings_to_keep_3Di, test_embeddings_to_keep_3Di  
            # Immediately delete to free memory
            gc.collect()

        print(f'{green} \nData Loading done{reset}')

    return X_train, y_train, X_val, y_val, X_test, y_test


def data_preparation(X_train, y_train, y_val, y_test):
    '''
    Prepares the data for training. (Encodes the labels and shuffles the data.)

    Args:
        X_train (np.ndarray): The training data.
        y_train (list): The training labels.
        y_val (list): The validation labels.
        y_test (list): The test labels.
    
    Returns:
        X_train (np.ndarray): The training data.
        y_train (np.ndarray): The training labels.
        y_val (np.ndarray): The validation labels.
        y_test (np.ndarray): The test labels.
    
    '''
    
    y_tot = y_train + y_val + y_test
    le = preprocessing.LabelEncoder()
    le.fit(y_tot)

    y_train = np.asarray(le.transform(y_train))
    y_val = np.asarray(le.transform(y_val))
    y_test = np.asarray(le.transform(y_test))

    num_classes = len(np.unique(y_tot))
    print('number of classes: ',num_classes)
    
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    print(f'{green}Data preparation done{reset}')

    return X_train, y_train, y_val, y_test, num_classes, le


def bm_generator(X, y, batch_size, num_classes):
    '''
    Generates batches of data for training.

    Args:
        X (np.ndarray): The data.
        y (np.ndarray): The labels.
        batch_size (int): The batch size.
        num_classes (int): The number of classes.
    
    Yields:
        X_batch (np.ndarray): A batch of data.
        y_batch (np.ndarray): A batch of labels.
    
    '''
    val = 0

    while True:
        X_batch = []
        y_batch = []

        for _ in range(batch_size):

            if val == len(X):
                val = 0

            X_batch.append(X[val])
            y_enc = np.zeros((num_classes))
            y_enc[y[val]] = 1
            y_batch.append(y_enc)
            val += 1

        X_batch = np.asarray(X_batch)
        y_batch = np.asarray(y_batch)

        yield X_batch, y_batch


def create_model(model_name, num_classes, nb_layer_block, dropout, input_type, layer_size):
    '''
    Creates and returns a Keras model based on the specified model name and layer blocks.

    Args:
        model_name (str): The name of the model used to compute the embeddings.
        num_classes (int): The number of classes.
        nb_layer_block (int): The number of layer blocks.
        dropout (float): The dropout rate.
        input_type (str): The type of input data used for training the model.
        layer_size (int): The size of the layers.
    
    Returns:
        classifier (Model): The untrained model.
    
    '''
    
    
    if input_type == 'AA+3Di':
        input_shapes = {
            'ProtT5_new': (2048,),
            'ProtT5': (2048,),
            'ProstT5_full': (2048,),
            'ProstT5_half': (2048,),
            
            # for the 650M ESM2 version
            #'ESM2': (2304,),

            # for the 15B ESM2 version 5120
            'ESM2': (6144,),

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

            # for the 650M ESM2 version
            #'ESM2': (1280,),

            # for the 15B ESM2 version 5120
            'ESM2': (5120,),

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
        raise ValueError('Invalid input type, must be AA, 3Di, or AA+3Di')

    if model_name not in input_shapes:
        raise ValueError('Invalid model name')
    
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


def train_model(model_name, num_classes, X_train, y_train, X_val, y_val, input_type, nb_layer_block, dropout, layer_size, pLDDT_threshold, only_50_largest_SF, support_threshold):
    '''
    Train the model and save it.

    Args:
        model_name (str): The name of the model used to compute the embeddings.
        num_classes (int): The number of classes.
        X_train (np.ndarray): The training data.
        y_train (np.ndarray): The training labels.
        X_val (np.ndarray): The validation data.
        y_val (np.ndarray): The validation labels.
        input_type (str): The type of input data used for training the model.
        nb_layer_block (int): The number of layer blocks.
        dropout (float): The dropout rate.
        layer_size (int): The size of the layers.
        pLDDT_threshold (int): The pLDDT threshold used to filter the domains.
        only_50_largest_SF (bool): Whether to keep only the 50 largest superfamilies.
        support_threshold (int): The support threshold used to filter the domains.

    Returns:
        None
    
    
    '''

    print(f'{green}Model training {reset}')

    # The following code defines the name for various files that will be saved during the training process with this specific hyperparaeter combination
    if only_50_largest_SF:
        base_model_path = f'saved_models/ann_{model_name}_top_50_SF'
        base_loss_path = f'results/Loss/ann_{model_name}_top_50_SF'

    else:
        base_model_path = f'saved_models/ann_{model_name}'
        base_loss_path = f'results/Loss/ann_{model_name}'

    

    save_model_path = f'{base_model_path}_{nb_layer_block}_blocks_dropout_{dropout}_layer_size_{layer_size}_pLDDT_{pLDDT_threshold}_support_threshold_{support_threshold}'
    save_loss_path = f'{base_loss_path}_{nb_layer_block}_blocks_dropout_{dropout}_layer_size_{layer_size}_pLDDT_{pLDDT_threshold}_support_threshold_{support_threshold}.png'
    
    
    if input_type == '3Di':

        save_model_path += '_3Di'
        save_loss_path = save_loss_path.replace('.png', '_3Di.png')
    
    if input_type == 'AA+3Di':
            
        save_model_path += '_AA+3Di'
        save_loss_path = save_loss_path.replace('.png', '_AA+3Di.png')


    # Here is the code that effectively trains the model
    with tf.device('/gpu:0'):
        # model
        model = create_model(model_name, num_classes, nb_layer_block, dropout, input_type, layer_size)

        # adam optimizer
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

        # callbacks
        mcp_save = keras.callbacks.ModelCheckpoint(save_model_path, save_best_only=True, monitor='val_accuracy', verbose=1, save_format='tf')
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30)
        callbacks_list = [reduce_lr, mcp_save, early_stop]

        # test and train generators
        train_gen = bm_generator(X_train, y_train, batch_size, num_classes)
        val_gen = bm_generator(X_val, y_val, batch_size, num_classes)
        history = model.fit(train_gen, epochs = num_epochs, steps_per_epoch = math.ceil(len(X_train)/(batch_size)), verbose=1, validation_data = val_gen, validation_steps = math.ceil(len(X_val)/batch_size), shuffle = True, callbacks = callbacks_list)

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
        plt.savefig(save_loss_path)
        plt.show()
        plt.close()

        print(f'{green}Model training done{reset}')

        del model, history, loss, val_loss, epochs  # Immediately delete to free memory
        gc.collect()

def save_confusion_matrix(y_test, y_pred, confusion_matrix_path):
    '''
    Save the confusion matrix as a CSV file and a heatmap image.

    Args:
        y_test (np.ndarray): The true labels.
        y_pred (np.ndarray): The predicted labels.
        confusion_matrix_path (str): The path to save the confusion matrix.
    
    Returns:
        None
    
    '''

    matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
    size = matrix.shape[0]
    print(f'Size of confusion matrix: {size}')

    # Find the indices and values of the non-zero elements
    non_zero_indices = np.nonzero(matrix)
    non_zero_values = matrix[non_zero_indices]

    # Combine row indices, column indices, and values into a single array
    non_zero_data = np.column_stack((non_zero_indices[0], non_zero_indices[1], non_zero_values))

    # Save the non-zero entries to a CSV file
    np.savetxt(f'{confusion_matrix_path}.csv', non_zero_data, delimiter=',', fmt='%d')
    
    # Create a custom color map that makes zero cells white
    cmap = plt.cm.viridis
    cmap.set_under('white')
    
    # Plot the heatmap with logarithmic scaling
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=False, fmt='d', cmap=cmap, vmin=0.01, cbar_kws={'label': 'Log Scale'})
    
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


def evaluate_model(model_name, X_val, y_val, X_test, y_test, nb_layer_block, dropout, input_type, layer_size, pLDDT_threshold, le, only_50_largest_SF, support_threshold):
    '''
    Evaluate the trained model (with and without bootstrapping), saving a classification report, confusion matrix and performance metrics.

    Args:
        model_name (str): The name of the model used to compute the embeddings.
        X_val (np.ndarray): The validation data.
        y_val (np.ndarray): The validation labels.
        X_test (np.ndarray): The test data.
        y_test (np.ndarray): The test labels.
        nb_layer_block (int): The number of layer blocks.
        dropout (float): The dropout rate.
        input_type (str): The type of input data used for training the model.
        layer_size (int): The size of the layers.
        pLDDT_threshold (int): The pLDDT threshold used to filter the domains.
        le (LabelEncoder): The label encoder.
        only_50_largest_SF (bool): Whether to keep only the 50 largest superfamilies.
        support_threshold (int): The support threshold used to filter the domains.
    
    Returns:
        None
    
    '''

    print(f'{green}Model evaluation {reset}')

    # The following code defines the name for various files that will be saved during the training process with this specific hyperparaeter combination
    if only_50_largest_SF:
        model_name = f'{model_name}_top_50_SF'
    
    
    base_model_path = f'./saved_models/ann_{model_name}'
    base_classification_report_path = f'results/classification_report/CR_ANN_{model_name}'
    base_confusion_matrix_path = f'results/confusion_matrices/{model_name}'

    model_path = f'{base_model_path}_{nb_layer_block}_blocks_dropout_{dropout}_layer_size_{layer_size}_pLDDT_{pLDDT_threshold}_support_threshold_{support_threshold}'


    classification_report_path = f'{base_classification_report_path}_{nb_layer_block}_blocks_dropout_{dropout}_layer_size_{layer_size}_pLDDT_{pLDDT_threshold}_support_threshold_{support_threshold}.csv'
    confusion_matrix_path = f'{base_confusion_matrix_path}_{nb_layer_block}_blocks_dropout_{dropout}_layer_size_{layer_size}_pLDDT_{pLDDT_threshold}_support_threshold_{support_threshold}'
    results_file = f'./results/perf_metrics/ann_{model_name}_{nb_layer_block}_blocks_dropout_{dropout}_layer_size_{layer_size}_pLDDT_{pLDDT_threshold}_support_threshold_{support_threshold}.csv'
    
    if input_type == '3Di':

        model_path += '_3Di'
            
        classification_report_path = classification_report_path.replace('.csv', '_3Di.csv')
        confusion_matrix_path = f"{confusion_matrix_path}_3Di"
        results_file = results_file.replace('.csv', '_3Di.csv')
    
    if input_type == 'AA+3Di':
                
        model_path += '_AA+3Di'
        classification_report_path = classification_report_path.replace('.csv', '_AA+3Di.csv')
        confusion_matrix_path = f"{confusion_matrix_path}_AA+3Di"
        results_file = results_file.replace('.csv', '_AA+3Di.csv')

    # Load the model and evaluate it
    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])

        try:
            model = load_model(model_path)
        except:
            raise ValueError(f'Model file {model_path} not found, make sure you have trained the model first, to train the model use the --do_training flag')
            
        with tf.device('/gpu:0'):
            writer.writerow(['Validation', ''])


            try:
                y_pred_val = model.predict(X_val)
            except Exception as e:
                print(f'Error during model prediction on validation data: {str(e)}')
                raise e

            f1_score_val = f1_score(y_val, y_pred_val.argmax(axis=1), average='weighted')
            acc_score_val = accuracy_score(y_val, y_pred_val.argmax(axis=1))
            writer.writerow(['Validation F1 Score', f1_score_val])
            writer.writerow(['Validation Accuracy Score', acc_score_val])

            writer.writerow(['Regular Testing', ''])
            y_pred_test = model.predict(X_test)
            f1_score_test = f1_score(y_test, y_pred_test.argmax(axis=1), average='macro')
            acc_score_test = accuracy_score(y_test, y_pred_test.argmax(axis=1))
            mcc_score = matthews_corrcoef(y_test, y_pred_test.argmax(axis=1))
            bal_acc = balanced_accuracy_score(y_test, y_pred_test.argmax(axis=1))
            writer.writerow(['Test F1 Score', f1_score_test])
            writer.writerow(['Test Accuracy Score', acc_score_test])
            writer.writerow(['Test MCC', mcc_score])
            writer.writerow(['Test Balanced Accuracy', bal_acc])

            # Remove '_top_50_SF' suffix from model_name if it's present, there is already a column for this information
            model_name = model_name.replace('_top_50_SF', '')


            # Save the test F1 score in a DataFrame
            df_results = pd.DataFrame({
                'Model': [model_name],
                'Nb_Layer_Block': [nb_layer_block],
                'Dropout': [dropout],
                'Input_Type': [input_type],
                'Layer_size': [layer_size],
                'pLDDT_threshold': [pLDDT_threshold],
                'is_top_50_SF': [bool(only_50_largest_SF)],
                'Support_threshold': [support_threshold],
                'F1_Score': [f1_score_test]
            })

            # This is the dataframe showing the results of all the models with all tried hyperparameters combinations (beeing the F1 scoreon the test dataset)
            df_results_path = './results/perf_dataframe.csv'

            # This code updates perf_dataframe.csv
            if os.path.exists(df_results_path):
                df_existing = pd.read_csv(df_results_path)
                # print('Columns in df_existing:', df_existing.columns.tolist())
                # Create a mask to find rows that match the current combination of parameters
                mask = (
                    (df_existing['Model'].astype(type(model_name)) == model_name) &
                    (df_existing['Nb_Layer_Block'].astype(type(nb_layer_block)) == nb_layer_block) &
                    (df_existing['Dropout'].astype(type(dropout)) == dropout) &
                    (df_existing['Input_Type'].astype(type(input_type)) == input_type) &
                    (df_existing['Layer_size'].astype(type(layer_size)) == layer_size) &
                    (df_existing['pLDDT_threshold'].astype(type(pLDDT_threshold)) == pLDDT_threshold) &
                    (df_existing['is_top_50_SF'].astype(bool) == bool(only_50_largest_SF)) &
                    (df_existing['Support_threshold'].astype(type(support_threshold)) == support_threshold)

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


            # Get all evaluation metrics with bootstrapping
            writer.writerow(['Bootstrapping Results', ''])
            num_iter = 1000
            f1_arr = []
            acc_arr = []
            mcc_arr = []
            bal_arr = []

            warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')

            print('Evaluation with bootstrapping')
            for it in tqdm(range(num_iter)):
                X_test_re, y_test_re = resample(X_test, y_test, n_samples=len(y_test), random_state=it)
                y_pred_test_re = model.predict(X_test_re, verbose=0)
                f1_arr.append(f1_score(y_test_re, y_pred_test_re.argmax(axis=1), average='macro'))
                acc_arr.append(accuracy_score(y_test_re, y_pred_test_re.argmax(axis=1)))
                mcc_arr.append(matthews_corrcoef(y_test_re, y_pred_test_re.argmax(axis=1)))
                bal_arr.append(balanced_accuracy_score(y_test_re, y_pred_test_re.argmax(axis=1)))

                del X_test_re, y_test_re, y_pred_test_re  # Immediately delete to free memory
                gc.collect()

            std_f1 = np.std(f1_arr)
            std_acc = np.std(acc_arr)

            writer.writerow(['Accuracy ', np.mean(acc_arr)])
            writer.writerow(['Accuracy std ', std_acc])
            writer.writerow(['F1-Score', np.mean(f1_arr)])
            writer.writerow(['F1-Score Std', std_f1])
            writer.writerow(['MCC', np.mean(mcc_arr)])
            writer.writerow(['Balanced Accuracy', np.mean(bal_arr)])

            
            
            warnings.filterwarnings('default')


            # save Classification report (CR) with the actual labels (the CR shows precision, recall, F1 score and support for every Super Falimies (SF))
            y_pred = model.predict(X_test)
            y_pred_labels = le.inverse_transform(y_pred.argmax(axis=1))
            y_test_labels = le.inverse_transform(y_test)
            cr = classification_report(y_test_labels, y_pred_labels, output_dict=True, zero_division=1)
            df = pd.DataFrame(cr).transpose()

            # Rename the index to 'SF' so that it becomes the first column in the CSV
            df.index.name = 'SF'

            df.to_csv(classification_report_path)

            save_confusion_matrix(y_test, y_pred, confusion_matrix_path)

            print(f'{green}Model evaluation done{reset}')

            # free all memory
            del model, y_pred_val, y_pred_test, y_pred, cr, df, df_results, df_existing, df_combined, f1_arr, acc_arr, mcc_arr, bal_arr, num_iter
            gc.collect()

            print(f'f1_score (on test dataset): {f1_score_test}')


def create_arg_parser():
    '''
    Creates and returns the ArgumentParser object.
    
    Args:
        None

    Returns:
        parser (ArgumentParser): The ArgumentParser object
    
    '''

    parser = argparse.ArgumentParser(description=
                        'Run training and evaluation for one or all models')
    parser.add_argument('--do_training', type=int, 
                        default=1, 
                        help='Whether to actually train and test the model or just test the saved model, put 0 to skip training, 1 to train')
    
    parser.add_argument('--dropout', type=float, 
                        default=0.3, 
                        help='Whether to use dropout in the model layers or not. Put 0 to not use dropout, a value between 0 and 1 excluded to use dropout with this value')
    
    parser.add_argument('--layer_size', type=int, 
                        default=2048, 
                        help='To choose the size of the dens layers in the classifier. The value should be a positive integer, 1024 or 2048 is recommended')


    parser.add_argument('--nb_layer_block', type=int, 
                        default=2,
                        help='Number of layer block in the classifier ({Dense, LeakyReLU, BatchNormalization, Dropout}= 1 layer block). The value should be a positive integer, 2 or 3 is recommended')
    
    parser.add_argument('--model', type=str, 
                        default='ProtT5', 
                        help='What model to use between ProtT5, ProtT5_new, ESM2, Ankh_large, Ankh_base, ProstT5_full, ProstT5_half, TM_Vec. Make sure to download the corresponding embeddings or compute them with ./src/model_building/models/ProstT5_Ankh_TMVec_ESM2_ProtT5new/embed_all_new_models.py')
    
    parser.add_argument('--classifier_input', type=str, 
                        default='AA', 
                        help='Whether to use Amino Acids, 3Di or a concatenation of both to train the classifier, put AA for Amino Acids, 3Di for 3Di, AA+3Di for the concatenation of the 2')
    
    
    parser.add_argument('--pLDDT_threshold', type=int, 
                        default=0, 
                        help='Threshold for pLDDT to filter the trining set of 3Di from hight structure quality, choose from [0, 4, 14, 24, 34, 44, 54, 64, 74, 84] (if classifier_input is AA, pLDDT_threshold will be set to 0, as it is only relevant for 3D structures from which 3Di are derived)')
    
    parser.add_argument('--only_50_largest_SF', type=int, 
                        default=0,
                        help='Whether to train only with the 50 most represented superfamilies or not, put 0 to use all the superfamilies, 1 to use only the 50 largest')
    
    parser.add_argument('--support_threshold', type=int, 
                        default=0,
                        help='Whether to filter the training set to only keep the SF with a support > support_threshold, put 0 to not filter, and any number >0 to filter for this value (you have to run dataset_filtering_for_3Di_usage.py with the same support_threshold before, or download the already computed filter csv to use this option)')

    
    return parser

def main():

    parser = create_arg_parser()
    args = parser.parse_args()

    model_name = args.model
    do_training = args.do_training
    nb_layer_block = args.nb_layer_block
    input_type = args.classifier_input
    pLDDT_threshold = args.pLDDT_threshold
    only_50_largest_SF = args.only_50_largest_SF
    support_threshold = args.support_threshold
    dropout = args.dropout
    layer_size=  args.layer_size

    # Argument checks:
    # Validate support_threshold
    if not isinstance(support_threshold, int) or support_threshold < 0:
        raise ValueError('support_threshold must be a non-negative integer')

    if input_type == 'AA':
        pLDDT_threshold = 0

    if only_50_largest_SF and support_threshold != 0:
        print(f'{yellow}Warning: When only_50_largest_SF is on (1), support_threshold will be ignored as the 50 most represented SF have a good support already{reset}')

    if (input_type == '3Di' or input_type == 'AA+3Di') and model_name == 'ProtT5':
        raise ValueError('Please use ProtT5_new instead of ProtT5 when using classifier_input 3Di or AA+3Di, see ReadMe for more details')

    do_training = False if int(args.do_training) == 0 else True

    print(f'{yellow}Hyperparameters{reset}')
    print(f'{yellow}Model Name: {model_name}{reset}')
    print(f'{yellow}Input Type: {input_type}{reset}')
    print(f'{yellow}Number of Layer Blocks: {nb_layer_block}{reset}')
    print(f'{yellow}Dropout: {dropout}{reset}')
    print(f'{yellow}Layer Size: {layer_size}{reset}')
    print(f'{yellow}pLDDT Threshold: {pLDDT_threshold}{reset}')
    print(f'{yellow}Only 50 Largest SF: {only_50_largest_SF}{reset}')
    print(f'{yellow}Support Threshold: {support_threshold}{reset}')
    print(f'{yellow}Do Training: {do_training}{reset}')
    print('\n')

    
    
    # Date loading
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(model_name, input_type, pLDDT_threshold, only_50_largest_SF, support_threshold)

    # Data preparation
    X_train, y_train, y_val, y_test, num_classes, le = data_preparation(X_train, y_train, y_val, y_test)

    # Training
    if do_training:
        train_model(model_name, num_classes, X_train, y_train, X_val, y_val, input_type, nb_layer_block, dropout, layer_size, pLDDT_threshold, only_50_largest_SF, support_threshold)
    
    # Evaluation
    evaluate_model(model_name, X_val, y_val, X_test, y_test, nb_layer_block, dropout, input_type, layer_size, pLDDT_threshold, le, only_50_largest_SF, support_threshold)
    # Clear memory after evaluation
    del X_train, y_train, X_val, y_val, X_test, y_test
    gc.collect()
                                

if __name__ == '__main__':
    main()