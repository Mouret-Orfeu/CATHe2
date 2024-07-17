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

# dataset import #################################################################################

# model_type = "large"
model_type = "base"

# train 

df_train = pd.read_csv('./data/Dataset/csv/Train.csv')
# Extract Super Families (SF column) 
y_train = df_train['SF'].tolist()
# # Extract AA Sequences
# AA_sequences_train = df_train['Sequence'].tolist()

# sort_and_save_embeddings("./data/Dataset/embeddings/Train_ProstT5_not_ordered.npz", "./data/Dataset/embeddings/Train_ProstT5.npz", "Train")

filename = f'./data/Dataset/embeddings/Train_Ankh_{model_type}.npz'
X_train = np.load(filename)['arr_0']

# val

df_val = pd.read_csv('./data/Dataset/csv/Val.csv')
# Extract Super Families (SF column)
y_val = df_val['SF'].tolist()
# # Extract AA Sequences
# AA_sequences_val = df_val['Sequence'].tolist()

# sort_and_save_embeddings("./data/Dataset/embeddings/Val_ProstT5_not_ordered.npz", "./data/Dataset/embeddings/Val_ProstT5.npz", "Val")

filename = f'./data/Dataset/embeddings/Val_Ankh_{model_type}.npz'
X_val = np.load(filename)['arr_0']

# test

df_test = pd.read_csv('./data/Dataset/csv/Test.csv')
# Extract Super Families (SF column)
y_test = df_test['SF'].tolist()
# # Extract AA Sequences
# AA_sequences_test = df_test['Sequence'].tolist()

# AA_sequence_lists = [AA_sequences_train, AA_sequences_val, AA_sequences_test]

# sort_and_save_embeddings("./data/Dataset/embeddings/Test_ProstT5_not_ordered.npz", "./data/Dataset/embeddings/Test_ProstT5.npz", "Test")

filename = f'./data/Dataset/embeddings/Test_Ankh_{model_type}.npz'
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
    if model_type == "base":
        input_ = Input(shape = (768,))
    else:
        input_ = Input(shape = (1536,))
    
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
    mcp_save = keras.callbacks.ModelCheckpoint(f'saved_models/ann_Ankh_{model_type}.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30)
    callbacks_list = [reduce_lr, mcp_save, early_stop]

    # test and train generators
    train_gen = bm_generator(X_train, y_train, bs)
    val_gen = bm_generator(X_val, y_val, bs)
    test_gen = bm_generator(X_test, y_test, bs)
    history = model.fit(train_gen, epochs = num_epochs, steps_per_epoch = math.ceil(len(X_train)/(bs)), verbose=1, validation_data = val_gen, validation_steps = len(X_val)/bs, workers = 0, shuffle = True, callbacks = callbacks_list)
    # model = load_model(f'saved_models/ann_Ankh_{model_type}.h5')

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
    plt.savefig(f'results/Loss/Ankh_{model_type}_loss.png')  # Save the plot
    plt.close()

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
    df.to_csv(f'results/CR_ANN_Ankh_{model_type}.csv')

    
    # Print and save confusion matrix as CSV
    matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))

    # Find the indices and values of the non-zero elements
    non_zero_indices = np.nonzero(matrix)
    non_zero_values = matrix[non_zero_indices]

    # Combine row indices, column indices, and values into a single array
    non_zero_data = np.column_stack((non_zero_indices[0], non_zero_indices[1], non_zero_values))

    # Save the non-zero entries to a CSV file
    np.savetxt(f'results/confusion_matrices/Ankh_{model_type}_confusion_matrix_non_zero.csv', non_zero_data, delimiter=",", fmt='%d')


    print("F1 Score")
    print(f1_score(y_test, y_pred.argmax(axis=1), average='macro', zero_division=0))

'''
Ankh base, 1st run

loss: 1.2834 - accuracy: 0.7912 - val_loss: 1.0716 - val_accuracy: 0.8386 - lr: 1.0000e-07

Validation
F1 Score:  0.8098861522133841
Acc Score 0.8372820742065266

Regular Testing
F1 Score:  0.6270442365165978
Acc Score:  0.8222586412395709
MCC:  0.8215920908909697
Bal Acc:  0.648935557394889

Bootstrapping Results

Accuracy:  0.8220589988081048 0.0046951685793675795
F1-Score:  0.6380571053284291 0.007031199643319887
MCC:  0.8213899253071208 0.00470339084342334
Bal Acc:  0.6731344328836549 0.006442662860302075


Classification Report Validation

Confusion Matrix
[[149   0   0 ...   0   0   0]
 [  0   2   0 ...   0   0   0]
 [  0   0   0 ...   0   0   0]
 ...
 [  0   0   0 ...   1   0   0]
 [  0   0   0 ...   0   0   0]
 [  0   0   0 ...   0   0   1]]



Ankh base, 2nd run 
loss: 1.3076 - accuracy: 0.7895 - val_loss: 1.1167 - val_accuracy: 0.8361 - lr: 1.0000e-07

Validation
F1 Score:  0.8049144908934985
Acc Score 0.8343018924154374

Regular Testing
F1 Score:  0.6181843416765387
Acc Score:  0.8189809296781884
MCC:  0.8183144670504615
Bal Acc:  0.6396406715601936

Bootstrapping Results

Accuracy:  0.8187631108462455 0.004770230184031273
F1-Score:  0.6308499878562631 0.0070703741887524285
MCC:  0.8180940180626146 0.004777670557058419
Bal Acc:  0.6649107545937295 0.006557151570136621

F1 Score
0.6181843416765387

'''


'''

Ankh large, 1st run 

Epoch 137: val_accuracy did not improve from 0.85803
loss: 1.2081 - accuracy: 0.8205 - val_loss: 1.0857 - val_accuracy: 0.8562 - lr: 1.0000e-08

Validation
F1 Score:  0.8301588637844687
Acc Score 0.8553121740426166

Regular Testing
F1 Score:  0.6460539974384623
Acc Score:  0.8413289630512515
MCC:  0.840729233523283
Bal Acc:  0.6678289419989185

Bootstrapping Results

Accuracy:  0.8412458283671037 0.00453055543049212
F1-Score:  0.659710774816739 0.007260282942655905
MCC:  0.8406445803500666 0.004540436539813436
Bal Acc:  0.694433948777581 0.0065618395131070775

Classification Report Validation

'''