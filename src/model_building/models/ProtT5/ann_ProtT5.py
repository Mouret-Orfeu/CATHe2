'''
ANN model trained on the T5 embeddings
'''

# ANSI escape code for colored text
yellow = '\033[93m'
green = '\033[92m'
reset = '\033[0m'
red = '\033[91m'


import sys
import os

# Check if a virtual environment is active
if not hasattr(sys, 'base_prefix') or sys.base_prefix == sys.prefix:
    raise EnvironmentError(f'{red}No virtual environment is activated. Please activate the right venv_2 to run this code. See ReadMe for more details.{reset}')

# Get the name of the activated virtual environment
venv_path = os.environ.get('VIRTUAL_ENV')
if venv_path is None:
    raise EnvironmentError(f'{red}Error, venv path is none. Please activate the venv_2. See ReadMe for more details.{reset}')

venv_name = os.path.basename(venv_path)
if venv_name != 'venv_2':
    raise EnvironmentError(f'{red}The activated virtual environment is {venv_name}, not venv_2. However venv_2 must be activated to run this code. See ReadMe for more details.{reset}')

# libraries
import pandas as pd 
import numpy as np 
import math
from sklearn import preprocessing
import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, matthews_corrcoef, balanced_accuracy_score
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow import keras
from sklearn.utils import resample
import torch
import seaborn as sns
import matplotlib
import warnings
from tqdm import tqdm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# GPU config for Vamsi' Laptop
from tensorflow.compat.v1 import ConfigProto

tf.keras.backend.clear_session()
config = ConfigProto()
config.gpu_options.allow_growth = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


new_embeddings = False
# new_embeddings = True

if new_embeddings:
    # train 

    df_train = pd.read_csv('./data/Dataset/csv/Train.csv')
    # Extract Super Families (SF column) 
    y_train = df_train['SF'].tolist()

    filename = f'./data/Dataset/embeddings/Train_ProtT5_new.npz'
    X_train = np.load(filename)['arr_0']

    # val

    df_val = pd.read_csv('./data/Dataset/csv/Val.csv')
    # Extract Super Families (SF column)
    y_val = df_val['SF'].tolist()

    filename = f'./data/Dataset/embeddings/Val_ProtT5_new.npz'
    X_val = np.load(filename)['arr_0']

    # test

    df_test = pd.read_csv('./data/Dataset/csv/Test.csv')
    # Extract Super Families (SF column)
    y_test = df_test['SF'].tolist()

    filename = f'./data/Dataset/embeddings/Test_ProtT5_new.npz'
    X_test = np.load(filename)['arr_0']

else:

    # dataset import
    # train 
    ds_train = pd.read_csv('./data/Dataset/annotations/Y_Train_SF.csv')
    y_train = list(ds_train['SF'])

    filename = './data/Dataset/embeddings/SF_Train_ProtT5.npz'
    X_train = np.load(filename)['arr_0']
    filename = './data/Dataset/embeddings/Other Class/Other_Train.npz'
    X_train_other = np.load(filename)['arr_0']

    X_train = np.concatenate((X_train, X_train_other), axis=0)

    for i in range(len(X_train_other)):
        y_train.append('other')

    # val
    ds_val = pd.read_csv('./data/Dataset/annotations/Y_Val_SF.csv')
    y_val = list(ds_val['SF'])

    filename = './data/Dataset/embeddings/SF_Val_ProtT5.npz'
    X_val = np.load(filename)['arr_0']

    filename = './data/Dataset/embeddings/Other Class/Other_Val.npz'
    X_val_other = np.load(filename)['arr_0']

    X_val = np.concatenate((X_val, X_val_other), axis=0)

    for i in range(len(X_val_other)):
        y_val.append('other')

    # test
    ds_test = pd.read_csv('./data/Dataset/annotations/Y_Test_SF.csv')
    y_test = list(ds_test['SF'])

    filename = './data/Dataset/embeddings/SF_Test_ProtT5.npz'
    X_test = np.load(filename)['arr_0']

    filename = './data/Dataset/embeddings/Other Class/Other_Test.npz'
    X_test_other = np.load(filename)['arr_0']

    X_test = np.concatenate((X_test, X_test_other), axis=0)

    for i in range(len(X_test_other)):
        y_test.append('other')

    # y process
    y_tot = y_train + y_val + y_test
    le = preprocessing.LabelEncoder()
    le.fit(y_tot)

    y_train = np.asarray(le.transform(y_train))
    y_val = np.asarray(le.transform(y_val))
    y_test = np.asarray(le.transform(y_test))

    num_classes = len(np.unique(y_tot))
    print(num_classes)
    print('Loaded X and y')

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    print('Shuffled')

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
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

    # callbacks
    mcp_save = keras.callbacks.ModelCheckpoint('saved_models/ann_t5_m1.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30)
    callbacks_list = [reduce_lr, mcp_save, early_stop]

    # test and train generators
    train_gen = bm_generator(X_train, y_train, bs)
    val_gen = bm_generator(X_val, y_val, bs)
    test_gen = bm_generator(X_test, y_test, bs)
    history = model.fit(train_gen, epochs = num_epochs, steps_per_epoch = math.ceil(len(X_train)/(bs)), verbose=1, validation_data = val_gen, validation_steps = len(X_val)/bs, workers = 0, shuffle = True, callbacks = callbacks_list)
    model = load_model('saved_models/ann_t5_m1.h5')

    # Save the training and validation loss
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
    plt.savefig(f'results/Loss/ProtT5_loss.png')  # Save the plot
    plt.close()

    print('Validation')
    y_pred_val = model.predict(X_val)
    f1_score_val = f1_score(y_val, y_pred_val.argmax(axis=1), average = 'weighted')
    acc_score_val = accuracy_score(y_val, y_pred_val.argmax(axis=1))
    print('F1 Score: ', f1_score_val)
    print('Acc Score', acc_score_val)

    print('Regular Testing')
    y_pred_test = model.predict(X_test)
    f1_score_test = f1_score(y_test, y_pred_test.argmax(axis=1), average = 'macro')
    acc_score_test = accuracy_score(y_test, y_pred_test.argmax(axis=1))
    mcc_score = matthews_corrcoef(y_test, y_pred_test.argmax(axis=1))
    bal_acc = balanced_accuracy_score(y_test, y_pred_test.argmax(axis=1))
    print('F1 Score: ', f1_score_test)
    print('Acc Score: ', acc_score_test)
    print('MCC: ', mcc_score)
    print('Bal Acc: ', bal_acc)

    print('Bootstrapping Results')
    num_iter = 1000
    f1_arr = []
    acc_arr = []
    mcc_arr = []
    bal_arr = []

    #Suppress warnings that will rise when bootstrapping samples contain classes not in the original sample
    warnings.filterwarnings('ignore', message='y_pred contains classes not in y_true')

    for it in tqdm(range(num_iter)):
        X_test_re, y_test_re = resample(X_test, y_test, n_samples = len(y_test), random_state=it)
        y_pred_test_re = model.predict(X_test_re, verbose=0)
        f1_arr.append(f1_score(y_test_re, y_pred_test_re.argmax(axis=1), average = 'macro'))
        acc_arr.append(accuracy_score(y_test_re, y_pred_test_re.argmax(axis=1)))
        mcc_arr.append(matthews_corrcoef(y_test_re, y_pred_test_re.argmax(axis=1)))
        bal_arr.append(balanced_accuracy_score(y_test_re, y_pred_test_re.argmax(axis=1)))


    print('Accuracy: ', np.mean(acc_arr), np.std(acc_arr))
    print('F1-Score: ', np.mean(f1_arr), np.std(f1_arr))
    print('MCC: ', np.mean(mcc_arr), np.std(mcc_arr))
    print('Bal Acc: ', np.mean(bal_arr), np.std(bal_arr))

    # Allow the warning back 
    warnings.filterwarnings('default')

with tf.device('/gpu:0'):
    y_pred = model.predict(X_test)
    print('Classification Report Validation')
    cr = classification_report(y_test, y_pred.argmax(axis=1), output_dict = True, zero_division=1)
    df = pd.DataFrame(cr).transpose()
    df.to_csv('results/CR_ANN_T5_m1.csv')
    print('Confusion Matrix')
    matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))

    # Plot the confusion matrix 
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(f'results/confusion_matrices/ProtT5.png')  # Save the plot
    plt.close()

    print('F1 Score')
    print(f1_score(y_test, y_pred.argmax(axis=1), average = 'macro', zero_division=0))

'''
Tweak Other data:
a) Remove class 6
b) 25 from each class in test and val
'''

'''
Way too many other samples in val and test
a) 1024x2 (0.5) - 256

loss: 0.7481 - accuracy: 0.8624

F1 Score:  0.7072250344894865
Acc Score:  0.7396251673360107
MCC:  0.7439367570350309
Bal Acc:  0.7923599556589267

1024x2 (0.7) - 128 - Way too much dropout could be the issue

loss: 0.9094 - accuracy: 0.8437

F1 Score:  0.6667057506407006
Acc Score:  0.7293618920124945
MCC:  0.7303856719461872
Bal Acc:  0.7329597090747823

1024x2 (0.2) - 64 - Lesser dropout and batch size

b) 1024x1 (0.5) - 256

loss: 0.6177 - accuracy: 0.8829

F1 Score:  0.733353474457612
Acc Score:  0.7824631860776439
MCC:  0.7784467329168209
Bal Acc:  0.7812439121501705

1024x1 (0.7) - 128

loss: 0.7542 - accuracy: 0.8670

F1 Score:  0.6986591587279944
Acc Score:  0.7735385988398037
MCC:  0.7675747629977221
Bal Acc:  0.7428095072413309

c) 512x1 (0.5) - 256

loss: 0.6355 - accuracy: 0.8816

F1 Score:  0.7254044771293813
Acc Score:  0.7812360553324409
MCC:  0.7767163559433566
Bal Acc:  0.7753046550297541

512x1 (0.7) - 128

loss: 0.7978 - accuracy: 0.8586

F1 Score:  0.693131626493494
Acc Score:  0.7697456492637216
MCC:  0.7637481679099832
Bal Acc:  0.737607007720706

d) 256x1 (0.5) - 256

loss: 0.6915 - accuracy: 0.8689

F1 Score:  0.7243804437541794
Acc Score:  0.7781124497991968
MCC:  0.7735994225222987
Bal Acc:  0.7675602930508759

256x1 (0.7) - 128

loss: 0.9098 - accuracy: 0.8407

F1 Score:  0.66246664683889
Acc Score:  0.7563587684069611
MCC:  0.7495764493185506
Bal Acc:  0.7074274947707329

e) 128x1 (0.5) - 256

loss: 0.8186 - accuracy: 0.8486

F1 Score:  0.6951230931500092
Acc Score:  0.751450245426149
MCC:  0.748408553084235
Bal Acc:  0.7447784080111233

128x1 (0.7) - 128

loss: 1.0598 - accuracy: 0.8157

F1 Score:  0.6311465193058615
Acc Score:  0.7290272199910754
MCC:  0.7239960283763703
Bal Acc:  0.6808103473897157

f) 64x1 (0.5)

loss: 1.0196 - accuracy: 0.8185

F1 Score:  0.6433417577302714
Acc Score:  0.7223337795626952
MCC:  0.7204875021664188
Bal Acc:  0.7028595078082839

g) 32x1 (0.5)

loss: 1.4273 - accuracy: 0.7471

F1 Score:  0.5630673433712948
Acc Score:  0.6733601070950469
MCC:  0.6724525341384242
Bal Acc:  0.6292679814893505

h) 10x1 (0.5)

loss: 3.1310 - accuracy: 0.3750

F1 Score:  0.1079921301796054
Acc Score:  0.4312806782686301
MCC:  0.411300882685397
Bal Acc:  0.1321880984240405

i) 5x1 (0.5)

loss: 3.1235 - accuracy: 0.3895

F1 Score:  0.03960764520810487
Acc Score:  0.3036590807675145
MCC:  0.284572097914033
Bal Acc:  0.05724529191135837

j) 2x1 (0.5)

loss: 4.4821 - accuracy: 0.1912

F1 Score:  0.001816504556647293
Acc Score:  0.1504908522980812
MCC:  0.09186020167865079
Bal Acc:  0.0053293377212412455

k) 1x1 (0.5)

loss: 5.0721 - accuracy: 0.1054

F1 Score:  0.00054492457427147
Acc Score:  0.08311021865238732
MCC:  0.022393191155585926
Bal Acc:  0.0014463794762714455

'''
