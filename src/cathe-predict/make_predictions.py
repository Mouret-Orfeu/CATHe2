# This code uses the computed embeddings and loaded models to the CATH annotation prediction

# libraries
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from tensorflow.keras.models import load_model
import tensorflow_addons as tfa
import time
import argparse

yellow = '\033[93m'
reset = '\033[0m'

# Parse command-line arguments for the model and the input type
parser = argparse.ArgumentParser(description='Run predictions pipeline with FASTA file')
parser.add_argument('--model', type=str,default='ProtT5', choices=['ProtT5', 'ProstT5'], help='Model to use: ProtT5 (original one) or ProstT5 (new one)')
parser.add_argument('--input_type', type=str,default='AA', choices=['AA', 'AA+3Di'], help='Input type: AA or AA+3Di (AA+3Di is only supported by ProstT5). If you select AA+3Di, ensure to provide pdb files in ./src/cathe-predict/PDB_folder, from which 3Di sequences will be extracted.')
args = parser.parse_args()

st = time.time()
# load data	
if args.model == 'ProtT5':
    filename = './src/cathe-predict/Embeddings/Embeddings_ProtT5.npz'
    embeds = np.load(filename)['arr_0']
elif args.model == 'ProstT5':
    filename = './src/cathe-predict/Embeddings/Embeddings_ProstT5_AA.npz'
    data = np.load(filename)  # Load the .npz file
    keys = data['keys']  
    embeddings = data['embeddings']
    embeds_dict_AA = dict(zip(keys, embeddings))
    embeds = np.array([embeds_dict_AA[i] for i in embeds_dict_AA.keys()])
else:
    raise ValueError('Invalid model. Please choose ProtT5 or ProstT5.')

# Load 3Di embeddings and concatenate with AA embeddings if input type is AA+3Di
if args.input_type == 'AA+3Di':
    if args.model == 'ProtT5':
        raise ValueError('AA+3Di input type is only available with ProstT5 model.')
    
    # Load 3Di embeddings
    filename = './src/cathe-predict/Embeddings/3Di_embeddings.npz'
    data = np.load(filename)

    keys_3Di = data['keys']  

    embeddings_3Di = data['embeddings']
    embeds_dict_3Di = dict(zip(keys_3Di, embeddings_3Di))
    
    # Verify that keys match exactly
    if set(embeds_dict_AA.keys()) != set(embeds_dict_3Di.keys()):
        raise ValueError('The keys in the AA and 3Di embeddings do not match.')
    
    # Concatenate embeddings where keys match
    embeds = np.array([np.concatenate((embeds_dict_AA[key], embeds_dict_3Di[key])) for key in embeds_dict_AA.keys()])

# annotations
ds_train = pd.read_csv('./src/cathe-predict/Y_Train_SF.csv')

y_train = list(ds_train['SF'])

ds_val = pd.read_csv('./src/cathe-predict/Y_Val_SF.csv')

y_val = list(ds_val['SF'])

# test
ds_test = pd.read_csv('./src/cathe-predict/Y_Test_SF.csv')

y_test = list(ds_test['SF'])

# Model label management 
y_tot = []

for i in range(len(y_train)):
    y_tot.append(y_train[i])

for i in range(len(y_val)):
    y_tot.append(y_val[i])

for i in range(len(y_test)):
    y_tot.append(y_test[i])
    y_tot.append('other')

le = preprocessing.LabelEncoder()
le.fit(y_tot)

classes = le.classes_

# Model loading
if args.model == 'ProtT5' and args.input_type == 'AA':
	model = load_model('./src/cathe-predict/CATHe.h5', custom_objects={'loss': tfa.losses.SigmoidFocalCrossEntropy()})
elif args.model == 'ProstT5' and args.input_type == 'AA':
    model_path = './src/cathe-predict/ann_ProstT5_full_2_blocks_dropout_0.1_layer_size_1024_pLDDT_0_support_threshold_0'
    model = load_model(model_path, custom_objects={'loss': tfa.losses.SigmoidFocalCrossEntropy()})

elif args.model == 'ProstT5' and args.input_type == 'AA+3Di':
	model = load_model('./src/cathe-predict/ann_ProstT5_full_2_blocks_dropout_0.3_layer_size_2048_pLDDT_24_support_threshold_0_AA+3Di', custom_objects={'loss': tfa.losses.SigmoidFocalCrossEntropy()})

# Prediction
y_pred = model.predict(embeds)

count = 0
sfam_thresh = []
sequence_thresh = []
record_thresh = []
pred_prob = []
embeds_thresh = []
un_embeds_thresh = []

ds = pd.read_csv('./src/cathe-predict/Dataset.csv')

sequences = list(ds['Sequence'])
record = list(ds['Record'])

for i in range(len(y_pred)):
	sfam_thresh.append(classes[np.argmax(y_pred[i])])
	sequence_thresh.append(sequences[i])
	record_thresh.append(record[i])
	max_val = max(y_pred[i])
	pred_prob.append(max_val)
	embeds_thresh.append(embeds[i])
	un_embeds_thresh.append(embeds[i])


df = pd.DataFrame(list(zip(record_thresh, sequence_thresh, sfam_thresh, pred_prob)), columns =['Record', 'Sequence', 'CATHe_Predicted_SFAM', 'CATHe_Prediction_Probability'])
# print(df)
df.to_csv('./src/cathe-predict/Results.csv')
# print(len(embeds_thresh), len(un_embeds_thresh))
en = time.time()

# failure check for the Results.csv file
if df.empty or df.shape[0] == 0:
    raise RuntimeError('The Results.csv file is empty or was not created properly.')

