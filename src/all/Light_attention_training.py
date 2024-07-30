# -*- coding: utf-8 -*-
# the light attention part is inspired from  https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py


import argparse
import time
import torch
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer, T5Tokenizer, T5EncoderModel
from tqdm import tqdm
import sys
import ankh
from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
import re
import gc
import os
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras  
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU, Conv1D, Softmax, GlobalAveragePooling1D, Concatenate, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle, resample
import warnings
import seaborn as sns
import math
import csv
import matplotlib
import matplotlib.colors as mcolors
matplotlib.use('Agg')
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using device: {}".format(device))


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

def load_data(model_name, light_attention):
    """Loads data for the specified model."""

    df_train = pd.read_csv('./data/Dataset/csv/Train.csv')
    # Extract Super Families (SF column) 
    y_train = df_train['SF'].tolist()

    df_val = pd.read_csv('./data/Dataset/csv/Val.csv')
    # Extract Super Families (SF column)
    y_val = df_val['SF'].tolist()

    df_test = pd.read_csv('./data/Dataset/csv/Test.csv')
    # Extract Super Families (SF column)
    y_test = df_test['SF'].tolist()

    if light_attention:
        file_paths = {
            'ProtT5_new' : ('Train_ProtT5_new_per_AA.npz', 'Val_ProtT5_new_per_AA.npz', 'Test_ProtT5_new_per_AA.npz'),
            'ESM2': ('Train_ESM2_per_AA.npz', 'Val_ESM2_per_AA.npz', 'Test_ESM2_per_AA.npz'),
            'Ankh_large': ('Train_Ankh_large_per_AA.npz', 'Val_Ankh_large_per_AA.npz', 'Test_Ankh_large_per_AA.npz'),
            'Ankh_base': ('Train_Ankh_base_per_AA.npz', 'Val_Ankh_base_per_AA.npz', 'Test_Ankh_base_per_AA.npz'),
            'ProstT5_full': ('Train_ProstT5_full_per_AA.npz', 'Val_ProstT5_full_per_AA.npz', 'Test_ProstT5_full_per_AA.npz'),
            'ProstT5_half': ('Train_ProstT5_half_per_AA.npz', 'Val_ProstT5_half_per_AA.npz','Test_ProstT5_half_per_AA.npz')
        }
    else:
        file_paths = {
            'ProtT5': ('Train_ProtT5_per_protein.npz', 'Val_ProtT5_per_protein.npz', 'Test_ProtT5_per_protein.npz'),
            'ProtT5_new' : ('Train_ProtT5_new_per_protein.npz', 'Val_ProtT5_new_per_protein.npz', 'Test_ProtT5_new_per_protein.npz'),
            'ESM2': ('Train_ESM2_per_protein.npz', 'Val_ESM2_per_protein.npz', 'Test_ESM2_per_protein.npz'),
            'Ankh_large': ('Train_Ankh_large_per_protein.npz', 'Val_Ankh_large_per_protein.npz', 'Test_Ankh_large_per_protein.npz'),
            'Ankh_base': ('Train_Ankh_base_per_protein.npz', 'Val_Ankh_base_per_protein.npz', 'Test_Ankh_base_per_protein.npz'),
            'ProstT5_full': ('Train_ProstT5_full_per_protein.npz', 'Val_ProstT5_full_per_protein.npz', 'Test_ProstT5_full_per_protein.npz'),
            'ProstT5_half': ('Train_ProstT5_half_per_protein.npz', 'Val_ProstT5_half_per_protein.npz','Test_ProstT5_half_per_protein.npz'),
            'TM_Vec': ('Train_TM_Vec_per_protein.npz', 'Val_TM_Vec_per_protein.npz', 'Test_TM_Vec_per_protein.npz')
        }

    if model_name not in file_paths:
        raise ValueError("Invalid model name")
    
    Train_file_name, Val_file_name, Test_file_name = file_paths[model_name]
    X_train = np.load(f'./data/Dataset/embeddings/{Train_file_name}')['arr_0']
    X_val = np.load(f'./data/Dataset/embeddings/{Val_file_name}')['arr_0']
    X_test = np.load(f'./data/Dataset/embeddings/{Test_file_name}')['arr_0']

    print("\033[92mData Loading done\033[0m")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def data_preparation(X_train, y_train, y_val, y_test):
    """Prepares the data for training."""
    
    y_tot = y_train + y_val + y_test
    le = preprocessing.LabelEncoder()
    le.fit(y_tot)

    y_train = np.asarray(le.transform(y_train))
    y_val = np.asarray(le.transform(y_val))
    

    num_classes = len(np.unique(y_tot))
    print("number of classes: ",num_classes)
    
    X_train, y_train = shuffle(X_train, y_train, random_state=42)

    print("\033[92mData preparation done\033[0m")

    return X_train, y_train, y_val, num_classes


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


def light_attention_module(inputs, embeddings_dim, kernel_size, conv_dropout = 0.25):
    # Feature convolution
    feature_conv = Conv1D(embeddings_dim, kernel_size, padding='same')(inputs)
    feature_conv = Dropout(conv_dropout)(feature_conv)
    
    # Attention convolution
    attention_conv = Conv1D(embeddings_dim, kernel_size, padding='same')(inputs)
    
    # Masking (assuming the mask is passed as an additional input)
    mask = Lambda(lambda x: tf.expand_dims(tf.reduce_any(tf.not_equal(x, 0), axis=-1), axis=-1))(inputs)
    attention_conv = tf.where(mask, attention_conv, tf.fill(tf.shape(attention_conv), -1e9))
    
    # Softmax to get attention scores
    attention_scores = Softmax(axis=-1)(attention_conv)
    
    # Apply attention scores to the features
    weighted_sum = tf.reduce_sum(feature_conv * attention_scores, axis=-1)
    max_pool = tf.reduce_max(feature_conv, axis=-1)
    
    # Concatenate the weighted sum and max pooled features
    concatenated = Concatenate(axis=-1)([weighted_sum, max_pool])
    
    return concatenated

# Keras NN Model
def create_model(model_name, num_classes, nb_layer_block, dropout, light_attention):
    """Creates and returns a Keras model based on the specified model name and layer blocks."""
    
    if light_attention:
        input_shapes = {
        'ProtT5_new': (None, 1024),  # (sequence_length, embedding_dim)
        'ProtT5': (None, 1024),
        'ProstT5_full': (None, 1024),
        'ProstT5_half': (None, 1024),
        'ESM2': (None, 1280),
        'Ankh_large': (None, 1536),
        'Ankh_base': (None, 768),
        'TM_Vec': (None, 512)
    }
    else:
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

    if model_name not in input_shapes:
        raise ValueError("Invalid model name")

    

    if light_attention:
        input_shape = input_shapes[model_name] # make sure that the input shape addaps to the sequence lenght
        input_ = Input(shape=input_shape)
        x = light_attention_module(input_ , embeddings_dim=input_shape[1], kernel_size=9)
    else:
        input_shape = input_shapes[model_name]
        input_ = Input(shape=input_shape)
        x = input_
    
    
    for _ in range(nb_layer_block):
        x = Dense(128, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(x)
        x = LeakyReLU(alpha=0.05)(x)
        x = BatchNormalization()(x)
        if dropout:
            x = Dropout(0.5)(x)
    
    out = Dense(num_classes, activation = 'softmax', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(x)
    classifier = Model(input_, out)

    return classifier

def train_model(model_name, num_classes, X_train_embeddings, y_train, X_val_embeddings, y_val, nb_layer_block, dropout, light_attention, first_batch_bool):
    """Trains the model."""

    if dropout:
        base_model_path = f'saved_models/ann_{model_name}'
        base_loss_path = f'results/Loss/ann_{model_name}'

        if light_attention:
            save_model_path = f'{base_model_path}_{nb_layer_block}_blocks_dropout_{dropout}_light_attention.h5'
            save_loss_path = f'{base_loss_path}_{nb_layer_block}_blocks_dropout_{dropout}_light_attention.png'
        else:
            save_model_path = f'{base_model_path}_{nb_layer_block}_blocks_dropout_{dropout}.h5'
            save_loss_path = f'{base_loss_path}_{nb_layer_block}_blocks_dropout_{dropout}.png'
    else:
        base_model_path = f'saved_models/ann_{model_name}'
        base_loss_path = f'results/Loss/ann_{model_name}'

        if light_attention:
            save_model_path = f'{base_model_path}_{nb_layer_block}_blocks_no_dropout_light_attention.h5'
            save_loss_path = f'{base_loss_path}_{nb_layer_block}_blocks_no_dropout_light_attention.png'
        else:
            save_model_path = f'{base_model_path}_{nb_layer_block}_blocks_no_dropout.h5'
            save_loss_path = f'{base_loss_path}_{nb_layer_block}_blocks_no_dropout.png'

    num_epochs = 200
    batch_size = 4096


    with tf.device('/gpu:0'):
        # model

        if first_batch_bool:
            model = create_model(model_name, num_classes, nb_layer_block, dropout, light_attention)
        else:
            model = load_model(save_model_path)

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

def evaluate_model(model_name, X_val, y_val, X_test, y_test, nb_layer_block, dropout):
    """Evaluates the trained model."""

    y_test = np.asarray(le.transform(y_test))
    
    if dropout:
        base_model_path = f'saved_models/ann_{model_name}'
        base_classification_report_path = f'results/classification_report/CR_ANN_{model_name}'
        base_confusion_matrix_path = f'results/confusion_matrices/{model_name}'

        model_path = f'{base_model_path}_{nb_layer_block}_blocks_dropout_{dropout}.h5'


        classification_report_path = f'{base_classification_report_path}_{nb_layer_block}_blocks_dropout_{dropout}.csv'
        confusion_matrix_path = f'{base_confusion_matrix_path}_{nb_layer_block}_blocks_dropout_{dropout}'
        results_file = f'./results/perf_metrics/ann_{model_name}_{nb_layer_block}_blocks_dropout_{dropout}.csv'
    else:
        base_model_path = f'saved_models/ann_{model_name}'
        base_classification_report_path = f'results/classification_report/CR_ANN_{model_name}'
        base_confusion_matrix_path = f'results/confusion_matrices/{model_name}'

        model_path = f'{base_model_path}_{nb_layer_block}_blocks_no_dropout.h5'

        classification_report_path = f'{base_classification_report_path}_{nb_layer_block}_blocks_no_dropout.csv'
        confusion_matrix_path = f'{base_confusion_matrix_path}_{nb_layer_block}_blocks_no_dropout'
        results_file = f'./results/perf_metrics/ann_{model_name}_{nb_layer_block}_blocks_no_dropout.csv'

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
                'F1_Score': [f1_score_test]
            })
            df_results_path = './results/perf_dataframe.csv'

            if os.path.exists(df_results_path):
                df_existing = pd.read_csv(df_results_path)
                # Create a mask to find rows that match the current combination of parameters
                mask = (
                    (df_existing['Model'] == model_name) &
                    (df_existing['Nb_Layer_Block'] == nb_layer_block) &
                    (df_existing['Dropout'] == dropout)
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

# all_models functions ##########################################################################

def get_model(model_name):

    print(f"Loading {model_name}")

    if model_name == 'ProtT5_new':
        tokenizer = T5Tokenizer.from_pretrained("./data/Dataset/weights/ProtT5/prot_t5_xl_uniref50", do_lower_case=False )
        model = T5EncoderModel.from_pretrained("./data/Dataset/weights/ProtT5/prot_t5_xl_uniref50")
        gc.collect()

    elif model_name == 'ESM2':
        model_path = "facebook/esm2_t33_650M_UR50D"
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_deep = None

    elif model_name == 'Ankh_large':
        model, tokenizer = ankh.load_large_model()
        model_deep = None
        
    elif model_name == 'Ankh_base':
        model, tokenizer = ankh.load_base_model()
        model_deep = None


    elif model_name in ['ProstT5_full', 'ProstT5_half']:
        model_path = "Rostlab/ProstT5"
        print("Loading ProstT5 from: {}".format(model_path))
        model = T5EncoderModel.from_pretrained(model_path)
        tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
        model_deep = None
                
    else:
        
        sys.exit(f"Stopping execution due to model '{model_name}' not found. Choose from: ESM2, Ankh_large, Ankh_base, ProstT5_full, ProstT5_half, TM_Vec.")
    
    model.to(device)
    model.eval()
        
    return model, tokenizer


def get_sequences(seq_path):

    print("Reading sequences")

    sequences = {}
    df = pd.read_csv(seq_path)
    for _, row in df.iterrows():
        sequences[int(row['Unnamed: 0'])] = row['Sequence']  
    
    return sequences



def embedding_set_up(seq_path_Train, seq_path_Val, model_name, is_3Di, max_seq_len=3263):
    emb_dict_Train = dict()
    emb_dict_Val = dict()
    seq_dict_Train = get_sequences(seq_path_Train)
    seq_dict_Val = get_sequences(seq_path_Val)
    model, tokenizer = get_model(model_name)

    if model_name == 'ProstT5_half':
        model = model.half()
    if model_name in ['ProstT5_full', 'ProstT5_half']:
        prefix = "<fold2AA>" if is_3Di else "<AA2fold>"
        print(f"Input is 3Di: {is_3Di}")
    else:
        prefix = None

    print('########################################')
    print('Total number of sequences: {}'.format(len(seq_dict_Train)))

    
    print("Average sequence length: {}".format(avg_length))
    print("Number of sequences >{}: {}".format(max_seq_len, n_long))

    return emb_dict_Train, emb_dict_Val, seq_dict_Train, seq_dict_Val, model, tokenizer, prefix


def batch_training(model_name, do_training, nb_layer_block, light_attention, dropout, X_train_embeddings, X_val_embeddings, ids):

    if light_attention and model_name in ['ProtT5', 'TM_Vec']:
        raise ValueError("Light attention cannot be used with ProtT5 or TM_Vec models, if you wanted to use ProtT5, use ProtT5_new instead")
    

    find y_train with ids
    find y_val with ids

    X_train, y_train, y_val, num_classes = data_preparation(X_train, y_train, y_val)
    if do_training:
        #DEBUG
        print(f"Training model with {nb_layer_block} blocks and dropout {dropout}")

        train_model(model_name, num_classes, X_train, y_train, X_val, y_val, nb_layer_block_dict[nb_layer_block], dropout, light_attention)

def preprocess_sequence(seq, model_name, prefix, batch, pdb_id):
    if model_name is 'ProtT5_new':
        # add a spaces between AA
        seq = " ".join(seq)

    # replace non-standard AAs
    seq = seq.replace('U', 'X').replace('Z', 'X').replace('O', 'X').replace('B', 'X')
    seq_len = len(seq)
    if model_name in ['ProstT5_full', 'ProstT5_half']:
        seq = prefix + ' ' + ' '.join(list(seq))
    batch.append((pdb_id, seq, seq_len))

    return batch

def embed_batch(model_name, model, tokenizer, batch, emb_dict, prefix):

    pdb_ids, seqs, seq_lens = zip(*batch)
    batch = list()


    if model_name in ['Ankh_large', 'Ankh_base']:
        # Split sequences into individual tokens
        seqs = [list(seq) for seq in seqs]
        

    token_encoding = tokenizer.batch_encode_plus(seqs, 
                                            add_special_tokens=True, 
                                            padding="longest", 
                                            is_split_into_words =(model_name in ['ESM2','Ankh_base','Ankh_large']),
                                            return_tensors='pt'
                                            ).to(device)

    try:
        with torch.no_grad():
            embedding_repr = model(token_encoding.input_ids, 
                                    attention_mask=token_encoding.attention_mask)
    except RuntimeError:
        raise RuntimeError("RuntimeError during embedding for the sequences {} of length (L={})".format(pdb_ids, seq_lens))
    
    # batch-size x seq_len x embedding_dim
    # extra token is added at the end of the seq
    for batch_idx, identifier in enumerate(pdb_ids):
        s_len = seq_lens[batch_idx]
        # account for prefix in offset
        emb = embedding_repr.last_hidden_state[batch_idx, 1:s_len+1]
        emb_dict[identifier] = emb.detach().cpu().numpy().squeeze()
        nb_processed_sequences += 1

    return emb_dict, nb_processed_sequences, pdb_ids

def train_classifyer_from_AA_sequences(seq_path_Train, seq_path_Val, model_name, do_training, nb_layer_block, light_attention, dropout, is_3Di,
                   max_residues=4096, max_seq_len=3263, nb_seq_max_in_batch=1000):
    

    
    emb_dict_Train, emb_dict_Val, seq_dict_Train, seq_dict_Val, model, tokenizer, prefix = embedding_set_up(seq_path_Train, seq_path_Val, model_name, is_3Di, max_seq_len)

    if do_training:

        Train_batch = list()
        Val_batch = list()
        nb_processed_sequences = 0
        for seq_idx, (pdb_id, seq) in enumerate(tqdm(seq_dict_Train, desc="Embedding sequences"), 1):
            Train_batch = preprocess_sequence(seq, model_name, prefix, Train_batch, pdb_id)
            seq_len = len(seq)


            # count residues in current Train_batch and add the last sequence length to
            # avoid that batches with (n_res_batch > max_residues) get processed 
            n_res_batch = sum([s_len for _, _, s_len in Train_batch])
            if len(Train_batch) >= nb_seq_max_in_batch or n_res_batch >= max_residues or seq_idx == len(seq_dict_Train) or seq_len > max_seq_len:
                emb_dict_Train, nb_processed_sequences, pdb_ids = embed_batch(model_name, model, tokenizer, Train_batch, emb_dict_Train, prefix, nb_processed_sequences)

                X_train_embeddings = np.array([emb_dict_Train[pdb_id] for pdb_id in pdb_ids])
                X_Val_embeddings = np.array([emb_dict_Val[pdb_id] for pdb_id in pdb_ids])

                batch_training(model_name, do_training, nb_layer_block, light_attention, dropout, X_train_embeddings, X_val_embeddings, ids)
                
                train the classifier here with the batch embeddings X_train_embeddings and X_Val_embeddings
                    

    
    X_val = sequences
    y_val = annotations
    X_test = sequences 
    y_test = annotations

    evaluate_model(model_name, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout)

    return True

def create_arg_parser():
    """Creates and returns the ArgumentParser object."""

    parser = argparse.ArgumentParser(description=
                        'Run training and evaluation for one or all models')
    parser.add_argument('--do_training', type=int, 
                        default=1, 
                        help="Whether to actually train and test the model or just test the saved model, put 0 to skip training, 1 to train")
    
    parser.add_argument('--dropout', type=str, 
                        default='0', 
                        help="Whether to use dropout in the model layers or not, and if so, what value, put 0 to not use dropout, a value between 0 and 1 excluded to use dropout with this value")
    
    parser.add_argument('--model', type=str, 
                        default='all', 
                        help="What model to use between ProtT5_new, ESM2, Ankh_large, Ankh_base, ProstT5_full or ProstT5_half")
    
    parser.add_argument('--nb_layer_block', type=str, 
                        default='one',
                        help="Number of layer block {Dense, LeakyReLU, BatchNormalization, Dropout} in the classifier. Choose between 'one', 'two' or 'three'")
    
    parser.add_argument('--light_attention', type=int, 
                        default=0, 
                        help="Whether to use light attention or not, put 0 to not use light attention, 1 to use light attention")

    parser.add_argument('--is_3Di', type=int,
                        default=0,
                        help="1 if you want to embed 3Di, 0 if you want to embed AA sequences. Default: 0")
    return parser


def process_datasets(model_name, do_training, nb_layer_block, light_attention, dropout, is_3Di):
    print(f"Embedding with {model_name}")

    
    seq_path_Train = f"./data/Dataset/csv/Train.csv"
    seq_path_Val = f"./data/Dataset/csv/Val.csv"

    train_classifyer_from_AA_sequences(
        seq_path_Train,
        seq_path_Val,
        model_name, 
        do_training, 
        nb_layer_block, 
        light_attention, 
        dropout, 
        is_3Di
    )

def main():

    parser = create_arg_parser()
    args = parser.parse_args()

    model_name = args.model
    do_training = args.do_training
    nb_layer_block = args.nb_layer_block
    light_attention = args.light_attention
    dropout = args.dropout

    do_training = False if int(args.do_training) == 0 else True
    is_3Di = False if int(args.is_3Di) == 0 else True

    print(f"Embedding with {model_name}")
    process_datasets(model_name, do_training, nb_layer_block, light_attention, dropout, is_3Di)

    
    
    
    


if __name__ == '__main__':
    main()
