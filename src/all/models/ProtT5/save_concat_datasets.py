import pandas as pd
import numpy as np

# using the original CATHe datasets for ProtT5
ds_train = pd.read_csv('./data/Dataset/annotations/Y_Train_SF.csv')
y_train = list(ds_train["SF"])

filename = './data/Dataset/embeddings/SF_Train_ProtT5.npz'
X_train = np.load(filename)['arr_0']
filename = './data/Dataset/embeddings/Other Class/Other_Train.npz'
X_train_other = np.load(filename)['arr_0']

X_train = np.concatenate((X_train, X_train_other), axis=0)

for i in range(len(X_train_other)):
    y_train.append('other')

# Save X_train and y_train
np.savez('./data/Dataset/embeddings/Train_ProtT5.npz', arr_0=X_train)
pd.DataFrame({'SF': y_train}).to_csv('./data/Dataset/annotations/Combined_Y_Train_SF.csv', index=False)

# val
ds_val = pd.read_csv('./data/Dataset/annotations/Y_Val_SF.csv')
y_val = list(ds_val["SF"])

filename = './data/Dataset/embeddings/SF_Val_ProtT5.npz'
X_val = np.load(filename)['arr_0']

# filename = './data/Dataset/embeddings/Other Class/Other_Val_US.npz'
filename = './data/Dataset/embeddings/Other Class/Other_Val.npz'
X_val_other = np.load(filename)['arr_0']

X_val = np.concatenate((X_val, X_val_other), axis=0)

for i in range(len(X_val_other)):
    y_val.append('other')

# Save X_val and y_val
np.savez('./data/Dataset/embeddings/Val_ProtT5.npz', arr_0=X_val)
pd.DataFrame({'SF': y_val}).to_csv('./data/Dataset/annotations/Combined_Y_Val_SF.csv', index=False)

# test
ds_test = pd.read_csv('./data/Dataset/annotations/Y_Test_SF.csv')
y_test = list(ds_test["SF"])

filename = './data/Dataset/embeddings/SF_Test_ProtT5.npz'
X_test = np.load(filename)['arr_0']

# filename = './data/Dataset/embeddings/Other Class/Other_Test_US.npz'
filename = './data/Dataset/embeddings/Other Class/Other_Test.npz'
X_test_other = np.load(filename)['arr_0']

X_test = np.concatenate((X_test, X_test_other), axis=0)

for i in range(len(X_test_other)):
    y_test.append('other')

# Save X_test and y_test
np.savez('./data/Dataset/embeddings/Test_ProtT5.npz', arr_0=X_test)
pd.DataFrame({'SF': y_test}).to_csv('./data/Dataset/annotations/Combined_Y_Test_SF.csv', index=False)

print("Files saved successfully.")
