# libraries
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv1D, Flatten, Input, LeakyReLU, Add
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, normalize
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, matthews_corrcoef, balanced_accuracy_score
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
from tensorflow import keras
import time
import argparse


# Parse command-line arguments for the model and the input type
parser = argparse.ArgumentParser(description="Run predictions pipeline with FASTA file")
parser.add_argument('--model', type=str,default='ProtT5', choices=['ProtT5', 'ProstT5'], help="Model to use: ProtT5 (original one) or ProstT5 (new one)")
parser.add_argument('--input_type', type=str,default='AA', choices=['AA', 'AA+3Di'], help="Input type: AA or AA+3Di (AA+3Di is only supported by ProstT5). If you select AA+3Di, ensure to provide pdb files in ./src/cathe-predict/pdb_folder, from which 3Di sequences will be extracted.")
args = parser.parse_args()

st = time.time()
# load data
filename = 'Embeddings_T5.npz'
embeds = np.load(filename)['arr_0']
# print(len(embeds))

# # annotations
ds_train = pd.read_csv('./src/cathe-predict/Y_Train_SF.csv')

y_train = list(ds_train["SF"])

ds_val = pd.read_csv('./src/cathe-predict/Y_Val_SF.csv')

y_val = list(ds_val["SF"])

# test
ds_test = pd.read_csv('./src/cathe-predict/Y_Test_SF.csv')

y_test = list(ds_test["SF"])

# y process
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

model = load_model('./src/cathe-predict/CATHe.h5', custom_objects={'loss': tfa.losses.SigmoidFocalCrossEntropy()})

y_pred = model.predict(embeds)

# print(y_pred.shape)

count = 0
sfam_thresh = []
sequence_thresh = []
record_thresh = []
pred_prob = []
embeds_thresh = []
un_embeds_thresh = []

ds = pd.read_csv('./src/cathe-predict/Dataset.csv')

sequences = list(ds["Sequence"])
record = list(ds["Record"])

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
# print(en-st)
