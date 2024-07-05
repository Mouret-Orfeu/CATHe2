# Part of the code from https://huggingface.co/Rostlab/ProstT5

import pandas as pd 
import numpy as np 
from sklearn import preprocessing
import math
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle, resample
import torch
from transformers import T5Tokenizer, T5EncoderModel
import re
from tqdm import tqdm

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

def sort_embeddings(input_npz_path, output_npz_path, name):
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

    # Convert the list to a dictionary with string keys to save as NPZ
    sorted_embeddings_dict = {str(key): value for key, value in tqdm(zip(sorted_keys, sorted_embeddings), desc="Creating sorted dictionary")}

    # Save the sorted embeddings
    np.savez(output_npz_path, **sorted_embeddings_dict)

# dataset import #################################################################################

# train 

df_train = pd.read_csv('./data/CATHe Dataset/csv/Train.csv')
# Extract Super Families (SF column) 
y_train = df_train['SF'].tolist()
# # Extract AA Sequences
# AA_sequences_train = df_train['Sequence'].tolist()

sort_embeddings("./data/CATHe Dataset/embeddings/Train_ProstT5_not_ordered.npz", "./data/CATHe Dataset/embeddings/Train_ProstT5.npz", "Train")

# val

df_val = pd.read_csv('./data/CATHe Dataset/csv/Val.csv')
# Extract Super Families (SF column)
y_val = df_val['SF'].tolist()
# # Extract AA Sequences
# AA_sequences_val = df_val['Sequence'].tolist()

sort_embeddings("./data/CATHe Dataset/embeddings/Val_ProstT5_not_ordered.npz", "./data/CATHe Dataset/embeddings/Val_ProstT5.npz", "Val")

# test

df_test = pd.read_csv('./data/CATHe Dataset/csv/Test.csv')
# Extract Super Families (SF column)
y_test = df_test['SF'].tolist()
# # Extract AA Sequences
# AA_sequences_test = df_test['Sequence'].tolist()

# AA_sequence_lists = [AA_sequences_train, AA_sequences_val, AA_sequences_test]

sort_embeddings("./data/CATHe Dataset/embeddings/Test_ProstT5_not_ordered.npz", "./data/CATHe Dataset/embeddings/Test_ProstT5.npz", "Test")


# AA Sequence embedding ############################################################################

# # Load the tokenizer
# tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)

# # Load the model
# model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)

# # only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
# model.full() if device=='cpu' else model.half()



# # replace all rare/ambiguous amino acids by X (3Di sequences does not have those) and introduce white-space between all sequences (AAs and 3Di)
# cleaned_AA_sequence_lists = []
# for AA_sequence_list in AA_sequence_lists:
#     cleaned_AA_sequence_list = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in AA_sequence_list]
#     cleaned_AA_sequence_lists.append(cleaned_AA_sequence_list)


# # add pre-fixes accordingly (this already expects 3Di-sequences to be lower-case)
# # if you go from AAs to 3Di (or if you want to embed AAs), you need to prepend "<AA2fold>"
# # if you go from 3Di to AAs (or if you want to embed 3Di), you need to prepend "<fold2AA>"
# embed_prepared_cleaned_AA_sequence_lists = [ [ "<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s
#                       for s in sequence_list
#                     ] for sequence_list in cleaned_AA_sequence_lists
#                 ]


# # Find the maximum length of sequences across all AA Sequence lists
# max_length = max([len(sequence) for sublist in embed_prepared_cleaned_AA_sequence_lists for sequence in sublist])

# # Calculate per-protein embeddings for the training dataset
# def get_per_protein_embeddings(embedding_repr, input_ids):
#     embeddings = []
#     for i in range(embedding_repr.last_hidden_state.shape[0]):
#         # Extract embeddings excluding special tokens
#         mask = (input_ids[i] != tokenizer.pad_token_id) & (input_ids[i] != tokenizer.cls_token_id) & (input_ids[i] != tokenizer.sep_token_id)
#         emb = embedding_repr.last_hidden_state[i, mask]  # Use mask directly without slicing
#         per_protein_embedding = emb.mean(dim=0)
#         embeddings.append(per_protein_embedding.cpu().numpy())
#     return embeddings

# from tqdm import tqdm

# # Helper function to process batches
# def process_batches(sequence_list, batch_size, max_length, tokenizer, model):
#     all_embeddings = []
#     for i in tqdm(range(0, len(sequence_list), batch_size), desc="Processing batches"):
#         batch_sequences = sequence_list[i:i+batch_size]
#         ids_batch = tokenizer.batch_encode_plus(batch_sequences, add_special_tokens=True, padding="max_length", max_length=max_length, return_tensors='pt').to(device)
        
#         with torch.no_grad():
#             embedding_repr_batch = model(
#                 ids_batch.input_ids, 
#                 attention_mask=ids_batch.attention_mask
#             )
        
#         batch_embeddings = get_per_protein_embeddings(embedding_repr_batch, ids_batch.input_ids)
#         all_embeddings.extend(batch_embeddings)
#     return all_embeddings


# # Process each data type in batches
# batch_size = 4
# data_types = ['train', 'val', 'test']
# embeddings = {}

# for i, data_type in enumerate(data_types):
#     embeddings[data_type] = process_batches(embed_prepared_cleaned_AA_sequence_lists[i], batch_size, max_length, tokenizer, model)

# # Access embeddings
# X_train = embeddings['train']
# X_val = embeddings['val']
# X_test = embeddings['test']


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
bs = 512

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
    mcp_save = keras.callbacks.ModelCheckpoint('saved_models/ann_t5_m1.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30)
    callbacks_list = [reduce_lr, mcp_save, early_stop]

    # test and train generators
    train_gen = bm_generator(X_train, y_train, bs)
    val_gen = bm_generator(X_val, y_val, bs)
    test_gen = bm_generator(X_test, y_test, bs)
    history = model.fit_generator(train_gen, epochs = num_epochs, steps_per_epoch = math.ceil(len(X_train)/(bs)), verbose=1, validation_data = val_gen, validation_steps = len(X_val)/bs, workers = 0, shuffle = True, callbacks = callbacks_list)
    model = load_model('saved_models/ann_t5_m1.h5')

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
    cr = classification_report(y_test, y_pred.argmax(axis=1), output_dict = True)
    df = pd.DataFrame(cr).transpose()
    df.to_csv('results/CR_ANN_T5_m1.csv')
    print("Confusion Matrix")
    matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
    print(matrix)
    print("F1 Score")
    print(f1_score(y_test, y_pred.argmax(axis=1), average = 'macro'))
