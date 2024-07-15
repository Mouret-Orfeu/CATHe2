# Part of the code from https://huggingface.co/Rostlab/ProstT5

import pandas as pd 
import numpy as np 
from sklearn import preprocessing
import math
import tensorflow as tf
from tensorflow import keras  
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle, resample
import torch
from transformers import T5Tokenizer, T5EncoderModel
import re
from tqdm import tqdm
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


# GPU config for Vamsi's Laptop
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

tf.keras.backend.clear_session()
config = ConfigProto()
config.gpu_options.allow_growth = True

# I comment out all gpu memory restrictions
# gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)

# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# LIMIT = 3 * 1024
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=LIMIT)])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Virtual devices must be set before GPUs have been initialized
#         print(e)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# functions #####################################################################################

def sort_and_save_embeddings(input_npz_path, output_npz_path, name):
    """
    Sort embeddings from an NPZ file by index and save the sorted embeddings to a new NPZ file.

    Parameters:
    input_npz_path (str): Path to the input NPZ file with unsorted embeddings.
    output_npz_path (str): Path to the output NPZ file to save the sorted embeddings.
    name : name of the embedding dataset processed
    """
    print("Sorting ",name," embeddings")

    # Load the embeddings from the NPZ file
    embeddings_not_ordered = np.load(input_npz_path)

    # Create a dictionary where keys are indices (converted to int) and values are embeddings
    embeddings_dict = {int(key): value for key, value in tqdm(embeddings_not_ordered.items(), desc="Creating dictionary")}

    # Sort the keys in ascending order
    sorted_keys = sorted(embeddings_dict.keys())

    # Create a list of embeddings in the sorted order
    sorted_embeddings = [embeddings_dict[key] for key in tqdm(sorted_keys, desc="Sorting embeddings")]

    # Save the sorted embeddings
    np.savez(output_npz_path, sorted_embeddings)

# dataset import #################################################################################

# train 

df_train = pd.read_csv('./data/Dataset/csv/Train.csv')
# Extract Super Families (SF column) 
y_train = df_train['SF'].tolist()
# # Extract AA Sequences
# AA_sequences_train = df_train['Sequence'].tolist()

# sort_and_save_embeddings("./data/Dataset/embeddings/Train_ProstT5_not_ordered.npz", "./data/Dataset/embeddings/Train_ProstT5.npz", "Train")

filename = './data/Dataset/embeddings/Train_ProstT5.npz'
X_train = np.load(filename)['arr_0']

# val

df_val = pd.read_csv('./data/Dataset/csv/Val.csv')
# Extract Super Families (SF column)
y_val = df_val['SF'].tolist()
# # Extract AA Sequences
# AA_sequences_val = df_val['Sequence'].tolist()

# sort_and_save_embeddings("./data/Dataset/embeddings/Val_ProstT5_not_ordered.npz", "./data/Dataset/embeddings/Val_ProstT5.npz", "Val")

filename = './data/Dataset/embeddings/Val_ProstT5.npz'
X_val = np.load(filename)['arr_0']

# test

df_test = pd.read_csv('./data/Dataset/csv/Test.csv')
# Extract Super Families (SF column)
y_test = df_test['SF'].tolist()
# # Extract AA Sequences
# AA_sequences_test = df_test['Sequence'].tolist()

# AA_sequence_lists = [AA_sequences_train, AA_sequences_val, AA_sequences_test]

# sort_and_save_embeddings("./data/Dataset/embeddings/Test_ProstT5_not_ordered.npz", "./data/Dataset/embeddings/Test_ProstT5.npz", "Test")

filename = './data/Dataset/embeddings/Test_ProstT5.npz'
X_test = np.load(filename)['arr_0']


# Training preparation ############################################################################

# y process
y_tot = []

for i in range(len(y_train)):
    y_tot.append(y_train[i])

for i in range(len(y_val)):
    y_tot.append(y_val[i])

for i in range(len(y_test)):
    y_tot.append(y_test[i])

le = preprocessing.LabelEncoder()
le.fit(y_tot)

y_train = np.asarray(le.transform(y_train))
y_val = np.asarray(le.transform(y_val))
y_test = np.asarray(le.transform(y_test))

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
    input_ = Input(shape = (1024,))
    
    x = Dense(128, kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(input_)
    x = LeakyReLU(alpha = 0.05)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # x = Dense(128, kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(x)
    # x = LeakyReLU(alpha = 0.05)(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x) 
    
    # x = Dense(128, kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(x)
    # x = LeakyReLU(alpha = 0.05)(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.5)(x) 
    
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
    mcp_save = keras.callbacks.ModelCheckpoint('saved_models/ann_ProstT5.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30)
    callbacks_list = [reduce_lr, mcp_save, early_stop]

    # test and train generators
    train_gen = bm_generator(X_train, y_train, bs)
    val_gen = bm_generator(X_val, y_val, bs)
    test_gen = bm_generator(X_test, y_test, bs)
    history = model.fit(train_gen, epochs = num_epochs, steps_per_epoch = math.ceil(len(X_train)/(bs)), verbose=1, validation_data = val_gen, validation_steps = len(X_val)/bs, workers = 0, shuffle = True, callbacks = callbacks_list)
    # model = load_model('saved_models/ann_ProstT5.h5')

    # Plot the training and validation loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    print("Validation")
    y_pred_val = model.predict(X_val)
    f1_score_val = f1_score(y_val, y_pred_val.argmax(axis=1), average = 'weighted')
    acc_score_val = accuracy_score(y_val, y_pred_val.argmax(axis=1))
    print("F1 Score: ", f1_score_val)
    print("Acc Score", acc_score_val)

    print("Regular Testing")
    y_pred_test = model.predict(X_test)
    f1_score_test = f1_score(y_test, y_pred_test.argmax(axis=1), average = 'macro')
    acc_score_test = accuracy_score(y_test, y_pred_test.argmax(axis=1))
    mcc_score = matthews_corrcoef(y_test, y_pred_test.argmax(axis=1))
    bal_acc = balanced_accuracy_score(y_test, y_pred_test.argmax(axis=1))
    print("F1 Score: ", f1_score_test)
    print("Acc Score: ", acc_score_test)
    print("MCC: ", mcc_score)
    print("Bal Acc: ", bal_acc)

    print("Bootstrapping Results")
    num_iter = 1000
    f1_arr = []
    acc_arr = []
    mcc_arr = []
    bal_arr = []
    for it in range(num_iter):
        # print("Iteration: ", it)
        X_test_re, y_test_re = resample(X_test, y_test, n_samples = len(y_test), random_state=it)
        y_pred_test_re = model.predict(X_test_re)
        print(y_test_re)
        f1_arr.append(f1_score(y_test_re, y_pred_test_re.argmax(axis=1), average = 'macro'))
        acc_arr.append(accuracy_score(y_test_re, y_pred_test_re.argmax(axis=1)))
        mcc_arr.append(matthews_corrcoef(y_test_re, y_pred_test_re.argmax(axis=1)))
        bal_arr.append(balanced_accuracy_score(y_test_re, y_pred_test_re.argmax(axis=1)))


    print("Accuracy: ", np.mean(acc_arr), np.std(acc_arr))
    print("F1-Score: ", np.mean(f1_arr), np.std(f1_arr))
    print("MCC: ", np.mean(mcc_arr), np.std(mcc_arr))
    print("Bal Acc: ", np.mean(bal_arr), np.std(bal_arr))

with tf.device('/gpu:0'):
    y_pred = model.predict(X_test)
    print("Classification Report Validation")
    cr = classification_report(y_test, y_pred.argmax(axis=1), output_dict=True)
    df = pd.DataFrame(cr).transpose()
    df.to_csv('results/CR_ANN_ProstT5.csv')
    
    print("Confusion Matrix")
    matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
    print(matrix)
    
    # Plot the confusion matrix 
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print("F1 Score")
    print(f1_score(y_test, y_pred.argmax(axis=1), average='macro'))

'''
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

'''

