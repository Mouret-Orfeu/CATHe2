# Part of the code from https://github.com/tymor22/tm-vec/blob/master/ipynb/repo_EMBED.ipynb
# Part of the code from https://github.com/tymor22/tm-vec/blob/master/tm_vec/tm_vec_utils.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tm_vec.embed_structure_model import trans_basic_block, trans_basic_block_Config
from tm_vec.tm_vec_utils import featurize_prottrans, embed_tm_vec, encode
from transformers import T5EncoderModel, T5Tokenizer
import gc
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Fonctions ####################################################################################

#Function to extract ProtTrans embedding for a sequence
def featurize_prottrans(sequences, model, tokenizer, device):

    sequences = [(" ".join(sequences[i])) for i in range(len(sequences))]
    sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)

    embedding = embedding.last_hidden_state.cpu().numpy()

    features = []
    for seq_num in range(len(sequences)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len - 1]
        features.append(seq_emd)

    prottrans_embedding = torch.tensor(features[0])
    prottrans_embedding = torch.unsqueeze(prottrans_embedding, 0).to(device)

    return (prottrans_embedding)



#Embed a protein using tm_vec (takes as input a prottrans embedding)
def embed_tm_vec(prottrans_embedding, model_deep, device):
    padding = torch.zeros(prottrans_embedding.shape[0:2]).type(torch.BoolTensor).to(device)
    tm_vec_embedding = model_deep(prottrans_embedding, src_mask=None, src_key_padding_mask=padding)

    return(tm_vec_embedding.cpu().detach().numpy())

def encode(sequences, model_deep, model, tokenizer, device):
    i = 0
    embed_all_sequences=[]
    while i < len(sequences):
        protrans_sequence = featurize_prottrans(sequences[i:i+1], model, tokenizer, device)
        embedded_sequence = embed_tm_vec(protrans_sequence, model_deep, device)
        embed_all_sequences.append(embedded_sequence)
        i = i + 1
    return np.concatenate(embed_all_sequences, axis=0)

# Dataset import ###################################################################################


# Train
df_train = pd.read_csv('./data/Dataset/csv/Train.csv')
y_train = df_train['SF'].tolist()
# AA_sequences_train = df_train['Sequence'].tolist()

filename = './data/Dataset/embeddings/Train_TM_Vec.npz'
X_train = np.load(filename)['arr_0']

# Val
df_val = pd.read_csv('./data/Dataset/csv/Val.csv')
y_val = df_val['SF'].tolist()
# AA_sequences_val = df_val['Sequence'].tolist()

filename = './data/Dataset/embeddings/Val_TM_vec.npz'
X_val = np.load(filename)['arr_0']

# Test
df_test = pd.read_csv('./data/Dataset/csv/Test.csv')
y_test = df_test['SF'].tolist()

filename = './data/Dataset/embeddings/Test_TM_Vec.npz'
X_test = np.load(filename)['arr_0']
# AA_sequences_test = df_test['Sequence'].tolist()

# # AA Sequence dictionary
# AA_sequences_dict = {
#     "train": AA_sequences_train,
#     "val": AA_sequences_val,
#     "test": AA_sequences_test
# }

# data_types = ['train', 'val', 'test']

# # Load the ProtTrans model and tokenizer
# tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False)
# model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
# gc.collect()

# model = model.to(device)
# model = model.eval()

# # TM-Vec model paths
# tm_vec_model_cpnt = "./src/all/models/TM_Vec/TM_Vec_config/last.ckpt"
# tm_vec_model_config = "./src/all/models/TM_Vec/TM_Vec_config/params.json"

# # Load the TM-Vec model
# tm_vec_model_config = trans_basic_block_Config.from_json(tm_vec_model_config)
# model_deep = trans_basic_block.load_from_checkpoint(tm_vec_model_cpnt, config=tm_vec_model_config)
# model_deep = model_deep.to(device)
# model_deep = model_deep.eval()

# # Process each data type in batches
# batch_size = 1000
# embeddings = {}

# # Loop through the datasets
# for data_type in data_types:
#     # Loop through the sequences and embed them in batches
#     i = 0
#     batch_sequences_emdeddings = []
#     while i < len(AA_sequences_dict[data_type]):
#         batch_sequences = AA_sequences_dict[data_type][i:i + batch_size]
#         embedded_sequence = encode(batch_sequences, model_deep, model, tokenizer, device)
        
#         # Append the embedded sequences to the list
#         batch_sequences_emdeddings.extend(embedded_sequence)
#         i += batch_size

#     embeddings[data_type] = batch_sequences_emdeddings

# # Optionally, you can also clear the cache at the end of the process
# torch.cuda.empty_cache()

# # Access embeddings
# train_embeddings = embeddings['train']
# val_embeddings = embeddings['val']
# test_embeddings = embeddings['test']


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

# renaming train_embeddings
X_train = train_embeddings
del train_embeddings

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
bs = 128

# Keras NN Model
def create_model():
    input_ = Input(shape = (1024,))
    
    x = Dense(128, kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(input_)
    x = LeakyReLU(alpha = 0.05)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(x)
    x = LeakyReLU(alpha = 0.05)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x) 
    
    x = Dense(128, kernel_initializer = 'glorot_uniform', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4), bias_regularizer=regularizers.l2(1e-4), activity_regularizer=regularizers.l2(1e-5))(x)
    x = LeakyReLU(alpha = 0.05)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x) 
    
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