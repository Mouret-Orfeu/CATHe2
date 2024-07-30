# the light attention part is inspired from  https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py

import argparse
import pandas as pd 
import os
import numpy as np 
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

def load_data(model_name):
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

        for i in range(len(X_test_other)):
            y_test.append('other')
    
    else:
        df_train = pd.read_csv('./data/Dataset/csv/Train.csv')
        # Extract Super Families (SF column) 
        y_train = df_train['SF'].tolist()

        df_val = pd.read_csv('./data/Dataset/csv/Val.csv')
        # Extract Super Families (SF column)
        y_val = df_val['SF'].tolist()

        df_test = pd.read_csv('./data/Dataset/csv/Test.csv')
        # Extract Super Families (SF column)
        y_test = df_test['SF'].tolist()

        
        
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
def create_model(model_name, num_classes, nb_layer_block, dropout):
    """Creates and returns a Keras model based on the specified model name and layer blocks."""
    
    
    
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

def train_model(model_name, num_classes, X_train, y_train, X_val, y_val, nb_layer_block, dropout, light_attention):
    """Trains the model."""

    if dropout:
        base_model_path = f'saved_models/ann_{model_name}'
        base_loss_path = f'results/Loss/ann_{model_name}'

        
        
        save_model_path = f'{base_model_path}_{nb_layer_block}_blocks_dropout_{dropout}.h5'
        save_loss_path = f'{base_loss_path}_{nb_layer_block}_blocks_dropout_{dropout}.png'
    else:
        base_model_path = f'saved_models/ann_{model_name}'
        base_loss_path = f'results/Loss/ann_{model_name}'

       
        
        save_model_path = f'{base_model_path}_{nb_layer_block}_blocks_no_dropout.h5'
        save_loss_path = f'{base_loss_path}_{nb_layer_block}_blocks_no_dropout.png'

    num_epochs = 200
    batch_size = 4096


    with tf.device('/gpu:0'):
        # model
        model = create_model(model_name, num_classes, nb_layer_block, dropout)

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


def create_arg_parser():
    """Creates and returns the ArgumentParser object."""

    parser = argparse.ArgumentParser(description=
                        'Run training and evaluation for one or all models')
    parser.add_argument('--do_training', type=int, 
                        default=1, 
                        help="Whether to actually train and test the model or just test the saved model, put 0 to skip training, 1 to train")
    
    parser.add_argument('--dropout', type=str, 
                        default='0', 
                        help="Whether to use dropout in the model layers or not, and if so, what value, put 0 to not use dropout, a value between 0 and 1 excluded to use dropout with this value, and 'all' to test every values in [0.1,0.2,0.3,0.4]")
    
    parser.add_argument('--model', type=str, 
                        default='all', 
                        help="What model to use between ProtT5, ProtT5_new, ESM2, Ankh_large, Ankh_base, ProstT5_full, ProstT5_half, TM_Vec, or all")
    
    parser.add_argument('--nb_layer_block', type=str, 
                        default='one',
                        help="Number of layer block {Dense, LeakyReLU, BatchNormalization, Dropout} in the classifier. Choose between 'one', 'two', 'three', or 'all'")
    return parser

def main():

    parser = create_arg_parser()
    args = parser.parse_args()

    model_name = args.model
    do_training = args.do_training
    nb_layer_block = args.nb_layer_block

    do_training = False if int(args.do_training) == 0 else True
    dropout_tag = float(args.dropout) if args.dropout != 'all' else args.dropout

    if dropout_tag == 'all':
        pass
    else:
        dropout = dropout_tag

    


    if model_name == 'all': 
        all_model_names = ['ProtT5', 'ProtT5_new', 'ESM2', 'Ankh_large', 'Ankh_base', 'ProstT5_full', 'ProstT5_half', 'TM_Vec']
        for model_name in all_model_names:
            if nb_layer_block == 'all':
                for nb_layer_block in ['one', 'two', 'three']:
                    if dropout_tag == 'all':
                        for dropout in [0.1, 0.2, 0.3, 0.4]:
                            X_train, y_train, X_val, y_val, X_test, y_test = load_data(model_name)
                            X_train, y_train, y_val, y_test, num_classes = data_preparation(X_train, y_train, y_val, y_test)
                            if do_training:
                                train_model(model_name, num_classes, X_train, y_train, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout)
                            evaluate_model(model_name, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout)
                    else:
                        X_train, y_train, X_val, y_val, X_test, y_test = load_data(model_name)
                        X_train, y_train, y_val, y_test, num_classes = data_preparation(X_train, y_train, y_val, y_test)
                        if do_training:
                            train_model(model_name, num_classes, X_train, y_train, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout)
                        evaluate_model(model_name, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout)
            else:
                if dropout_tag == 'all':
                    for dropout in [0.1, 0.2, 0.3, 0.4]:
                        X_train, y_train, X_val, y_val, X_test, y_test = load_data(model_name)
                        X_train, y_train, y_val, y_test, num_classes = data_preparation(X_train, y_train, y_val, y_test)
                        if do_training:
                            train_model(model_name, num_classes, X_train, y_train, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout)
                        evaluate_model(model_name, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout)
                else:
                    X_train, y_train, X_val, y_val, X_test, y_test = load_data(model_name)
                    X_train, y_train, y_val, y_test, num_classes = data_preparation(X_train, y_train, y_val, y_test)
                    if do_training:
                        train_model(model_name, num_classes, X_train, y_train, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout)
                    evaluate_model(model_name, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout)
    
    else:
        #DEBUG
        print(f"Model: {model_name}")

        if nb_layer_block == 'all':
            #DEBUG
            print(f"Layer Block: {nb_layer_block}")

            for nb_layer_block in ['one', 'two', 'three']:
                if dropout_tag == 'all':
                    #DEBUG
                    print(f"Dropout: {dropout_tag}")

                    for dropout in [0.1, 0.2, 0.3, 0.4]:
                        X_train, y_train, X_val, y_val, X_test, y_test = load_data(model_name)
                        X_train, y_train, y_val, y_test, num_classes = data_preparation(X_train, y_train, y_val, y_test)
                        if do_training:
                            train_model(model_name, num_classes, X_train, y_train, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout)
                        evaluate_model(model_name, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout)
                else:
                    #DEBUG
                    print(f"Dropout: {dropout_tag}")

                    X_train, y_train, X_val, y_val, X_test, y_test = load_data(model_name)
                    X_train, y_train, y_val, y_test, num_classes = data_preparation(X_train, y_train, y_val, y_test)
                    if do_training:
                        #DEBUG
                        print(f"Training model with {nb_layer_block} blocks and dropout {dropout}")

                        train_model(model_name, num_classes, X_train, y_train, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout)
                    evaluate_model(model_name, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout)
        else:
            #DEBUG
            print(f"Layer Block: {nb_layer_block}")

            if dropout_tag == 'all':
                #DEBUG
                print(f"Dropout: {dropout_tag}")

                for dropout in [0.1, 0.2, 0.3, 0.4]:
                    X_train, y_train, X_val, y_val, X_test, y_test = load_data(model_name)
                    X_train, y_train, y_val, y_test, num_classes = data_preparation(X_train, y_train, y_val, y_test)
                    if do_training:
                        #DEBUG
                        print(f"Training model with {nb_layer_block} blocks and dropout {dropout}")

                        train_model(model_name, num_classes, X_train, y_train, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout)
                    evaluate_model(model_name, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout)
            else:
                #DEBUG
                print(f"Dropout: {dropout_tag}")

                X_train, y_train, X_val, y_val, X_test, y_test = load_data(model_name)
                X_train, y_train, y_val, y_test, num_classes = data_preparation(X_train, y_train, y_val, y_test)
                if do_training:
                    #DEBUG
                    print(f"Training model with {nb_layer_block} blocks and dropout {dropout}")

                    train_model(model_name, num_classes, X_train, y_train, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout)
                evaluate_model(model_name, X_val, y_val, X_test, y_test, nb_layer_block_dict[nb_layer_block], dropout)

if __name__ == '__main__':
    main()