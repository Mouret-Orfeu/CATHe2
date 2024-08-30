# Part of the code from https://huggingface.co/Rostlab/ProstT5

import pandas as pd 
import numpy as np 
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras  
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle, resample
import torch
import warnings
import math
import csv
import seaborn as sns
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
matplotlib.use('Agg')

from sklearn.metrics import classification_report, confusion_matrix


# GPU config for Vamsi's Laptop
from tensorflow.compat.v1 import ConfigProto


tf.keras.backend.clear_session()
config = ConfigProto()
config.gpu_options.allow_growth = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dropout = True
# dropout = False
dropout_value = 0.2
nb_layer_block = 1

# model_type = "half"
# model_type = "full"

model_type = "half_CL"
# model_type = "full_CL"

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

def evaluate_model(model_type, X_val, y_val, X_test, y_test, nb_layer_block, dropout, dropout_value):
    """Evaluates the trained model."""
    
    model_name = f'ProstT5_{model_type}'

    if dropout :
        base_model_path = f'saved_models/ann_{model_name}'
        base_classification_report_path = f'results/classification_report/CR_ANN_{model_name}_dropout_{dropout_value}'
        base_confusion_matrix_path = f'results/confusion_matrices/{model_name}_dropout_{dropout_value}'

        model_path = f'{base_model_path}_{nb_layer_block}_blocks_dropout_{dropout_value}.h5'


        classification_report_path = f'{base_classification_report_path}_{nb_layer_block}_blocks_dropout_{dropout_value}.csv'
        confusion_matrix_path = f'{base_confusion_matrix_path}_{nb_layer_block}_blocks_dropout_{dropout_value}'
        results_file = f'./results/perf_metrics/ann_{model_name}_{nb_layer_block}_blocks_dropout_{dropout_value}.csv'
    else:
        base_model_path = f'saved_models/ann_{model_name}'
        base_classification_report_path = f'results/classification_report/CR_ANN_{model_name}_no_dropout'
        base_confusion_matrix_path = f'results/confusion_matrices/{model_name}_no_dropout'

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

            print("F1 score: ", f1_score_test)

            print("\033[92mModel evaluation done\033[0m")

# dataset import #################################################################################

# train 

df_train = pd.read_csv('./data/Dataset/csv/Train.csv')
# Extract Super Families (SF column) 
y_train = df_train['SF'].tolist()

filename = f'./data/Dataset/embeddings/Train_ProstT5_{model_type}.npz'
X_train = np.load(filename)['arr_0']

# val

df_val = pd.read_csv('./data/Dataset/csv/Val.csv')
# Extract Super Families (SF column)
y_val = df_val['SF'].tolist()

filename = f'./data/Dataset/embeddings/Val_ProstT5_{model_type}.npz'
X_val = np.load(filename)['arr_0']

# test

df_test = pd.read_csv('./data/Dataset/csv/Test.csv')
# Extract Super Families (SF column)
y_test = df_test['SF'].tolist()

filename = f'./data/Dataset/embeddings/Test_ProstT5_{model_type}.npz'
X_test = np.load(filename)['arr_0']


# Training preparation ############################################################################

# y process
y_tot = y_train + y_val + y_test
le = preprocessing.LabelEncoder()
le.fit(y_tot)

# label_to_idx_dict = {label: idx for idx, label in enumerate(le.classes_)}

y_train = np.asarray(le.transform(y_train))
y_val = np.asarray(le.transform(y_val))
y_test = np.asarray(le.transform(y_test))

# y_train = np.array([label_to_idx_dict[label] for label in y_train])
# y_val = np.array([label_to_idx_dict[label] for label in y_val])
# y_test = np.array([label_to_idx_dict[label] for label in y_test])

num_classes = len(np.unique(y_tot))
print(num_classes)
print("Loaded X and y")


X_train, y_train = shuffle(X_train, y_train, random_state=42)
print("Shuffled")

# generator
def bm_generator(X_t, y_t, batch_size):
    val = 0

    while True:
        X_batch = []
        y_batch = []

        for j in range(batch_size):

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

# Training and evaluation ################################################################################

# batch size
bs = 4096

# Keras NN Model
def create_model():

    if model_type == "full_CL" or model_type == "half_CL":
        input_ = Input(shape = (1024,))
    else:
        input_ = Input(shape = (1024,))
    
    x = input_

    for _ in range(nb_layer_block):
        x = Dense(128, kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(input_)
        x = LeakyReLU(alpha = 0.05)(x)
        x = BatchNormalization()(x)
        if dropout :
            x = Dropout(dropout_value)(x)
    
    out = Dense(num_classes, activation = 'softmax', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(x)
    classifier = Model(input_, out)

    return classifier

# training
num_epochs = 200

with tf.device('/gpu:0'):
    # model
    model = create_model()

    # adam optimizer
    opt = keras.optimizers.Adam(learning_rate = 1e-5)
    model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics=['accuracy'])

    # callbacks
    if dropout :
        model_path = f'saved_models/ann_ProstT5_{model_type}_{nb_layer_block}_blocks_dropout_{dropout_value}.h5'
    else:
        model_path = f'saved_models/ann_ProstT5_{model_type}_{nb_layer_block}_blocks_no_dropout.h5'

    mcp_save = keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30)
    callbacks_list = [reduce_lr, mcp_save, early_stop]

    # test and train generators
    train_gen = bm_generator(X_train, y_train, bs)
    val_gen = bm_generator(X_val, y_val, bs)
    test_gen = bm_generator(X_test, y_test, bs)
    # history = model.fit(train_gen, epochs = num_epochs, steps_per_epoch = math.ceil(len(X_train)/(bs)), verbose=1, validation_data = val_gen, validation_steps = len(X_val)/bs, workers = 0, shuffle = True, callbacks = callbacks_list)
    model = load_model(model_path)

    if dropout :
        plot_path = f'results/Loss/ProstT5_{model_type}_{nb_layer_block}_dropout_{dropout_value}.png'
    else:
        plot_path = f'results/Loss/ProstT5_{model_type}_{nb_layer_block}_no_dropout.png'

    # Plot the training and validation loss
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(loss) + 1)
    # plt.figure()
    # plt.plot(epochs, loss, 'b-', label='Training loss', linewidth=1)
    # plt.plot(epochs, val_loss, 'r-', label='Validation loss', linewidth=1)
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.savefig(plot_path)  # Save the plot
    # plt.close()

    evaluate_model(model_type, X_val, y_val, X_test, y_test, nb_layer_block, dropout, dropout_value)

'''
half
First run 

Validation

F1 Score:  0.8719911573504435
Acc Score 0.893458500968559

Regular Testing

F1 Score:  0.7327831765545498
Acc Score:  0.8814064362336115
MCC:  0.8809294222903166
Bal Acc:  0.7538498659351749

Bootstrapping Results

Accuracy:  0.8811777413587604 0.003946529607170833
F1-Score:  0.7395372295597611 0.006702172687649625
MCC:  0.8806987518101628 0.003958227328046929
Bal Acc:  0.773875020634396 0.005738367438081772

Classification Report Validation

Confusion Matrix
[[150   0   0 ...   0   0   0]
 [  0   2   0 ...   0   0   0]
 [  0   0   0 ...   0   0   0]
 ...
 [  0   0   0 ...   0   0   0]
 [  0   0   0 ...   0   1   0]
 [  0   0   0 ...   0   0   1]]

F1 Score
0.7327831765545498







2nd run

loss: 0.8994 - accuracy: 0.8618 - val_loss: 0.7553 - val_accuracy: 0.8966 - lr: 1.0000e-09

Validation

F1 Score:  0.8748210676936903
Acc Score 0.8959916554909849

Regular Testing

F1 Score:  0.7380297220339255
Acc Score:  0.8849821215733016
MCC:  0.8845170908939579
Bal Acc:  0.7593515749390974


Bootstrapping Results

Accuracy:  0.8848127234803337 0.0038629465624301436
F1-Score:  0.7447440598337535 0.006770882677865061
MCC:  0.8843460764902418 0.0038749005383619597
Bal Acc:  0.779480736580121 0.005810461838427936

Classification Report Validation

error

Confusion Matrix
[[151   0   0 ...   0   0   0]
 [  0   2   0 ...   0   0   0]
 [  0   0   0 ...   0   0   0]
 ...
 [  0   0   0 ...   0   0   0]
 [  0   0   0 ...   0   1   0]
 [  0   0   0 ...   0   0   1]]






 3rd run

Validation
F1 Score:  0.8789980654501429
Acc Score 0.8989718372820742

Regular Testing
F1 Score:  0.7417655359749582
Acc Score:  0.8851311084624554
MCC:  0.8846651906911425
Bal Acc:  0.7620069050689151

Bootstrapping Results
Accuracy:  0.8849095649582837 0.003848691629648413
F1-Score:  0.7483663586541225 0.006861601769890976
MCC:  0.8844417795621147 0.0038605606045651213
Bal Acc:  0.7815885856590119 0.005857068422754682

Classification Report Validation

Confusion Matrix
[[150   0   0 ...   0   0   0]
 [  0   2   0 ...   0   0   0]
 [  0   0   0 ...   0   0   0]
 ...
 [  0   0   0 ...   0   0   0]
 [  0   0   0 ...   0   1   0]
 [  0   0   0 ...   0   0   1]]

 


 4th run

 loss: 0.8825 - accuracy: 0.8623 - val_loss: 0.7420 - val_accuracy: 0.8993 - lr: 1.0000e-07
Validation
F1 Score:  0.877830672028923
Acc Score 0.8983758009238564

Regular Testing
F1 Score:  0.7382650500946161
Acc Score:  0.8851311084624554
MCC:  0.8846648727347662
Bal Acc:  0.75947438790924

Bootstrapping Results

Accuracy:  0.8848551847437426 0.003893133453404982
F1-Score:  0.7455093887946028 0.006745001034596315
MCC:  0.8843867725851623 0.003904714664583348
Bal Acc:  0.7794136948500072 0.005800158076804598

Classification Report Validation

F1 Score
0.7382650500946161

'''

'''
full

1st run

loss: 0.8775 - accuracy: 0.8625 - val_loss: 0.7250 - val_accuracy: 0.9003 - lr: 1.0000e-06
Validation
F1 Score:  0.878289920229444
Acc Score 0.8985248100134108

Regular Testing
F1 Score:  0.7408674674029917
Acc Score:  0.884684147794994
MCC:  0.8842130300547021
Bal Acc:  0.7615936810529366

Bootstrapping Results

Accuracy:  0.8844672228843862 0.0038354447672870676
F1-Score:  0.7471304839614815 0.006737653472158867
MCC:  0.8839941324649042 0.003847138286390804
Bal Acc:  0.781125624536717 0.005781797354727902

Classification Report Validation

F1 Score
0.7408674674029917


'''