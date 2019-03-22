import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================
# Preprocessing
# =========================================

df = pd.read_csv('data.csv')
dfFifa = df[['Name','Age','Overall','Potential']]
# print(dfFifa)
dftarget = dfFifa[dfFifa['Age'] <= 25][dfFifa['Overall'] >= 80][dfFifa['Potential'] >= 80]
labeltarget = np.array([1]*len(dftarget))
dftarget['Label'] = labeltarget
# print(dftarget)
# print(labeltarget)
dfnontarget = dfFifa.drop(dftarget.index)
labelnontarget = np.array([0]*len(dfnontarget))
dfnontarget['Label'] = labelnontarget
# print(dfnontarget) 
dflabeled = dftarget.append(dfnontarget, ignore_index = True)
# print(dflabeled)

# =========================================
# Split Train (90%) and Test (10%)
# =========================================

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    dflabeled[['Age', 'Overall', 'Potential']],
    dflabeled['Label'],
    test_size = .1
)
# print(len(x_train))     #16386
# print(len(x_test))      #1821

#================================================
#Import ML Model (LogReg, DecTree, KNN)
#================================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

#================================================
#Stratified K-Fold Cross Validation
#================================================

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
k = StratifiedKFold(n_splits = 100)
Data = dflabeled[['Age', 'Overall', 'Potential']].values
# print(Data)

#K-Fold
for train_index, test_index in k.split(Data, dflabeled['Label']):
    x_train = Data[train_index]
    y_train = dflabeled['Label'][train_index]

#Testing Cross Validation Score
print('Akurasi Model Regresi Logistik:',
    round(cross_val_score(
    LogisticRegression(solver = 'lbfgs', multi_class = 'auto'),
    x_train,
    y_train
).mean() * 100), 'persen')

print('Akurasi Model Decision Tree:',
    round(cross_val_score(
    DecisionTreeClassifier(),
    x_train,
    y_train
).mean() * 100), 'persen')

def k_value():
    k = round((len(x_train)+len(x_test)) ** .5)
    if (k % 2 == 0):
        return k + 1
    else:
        return k
# print(k_value())
print('Akurasi Model KNN:',
    round(cross_val_score(
    KNeighborsClassifier(n_neighbors = k_value()),
    x_train,
    y_train
).mean() * 100), 'persen')

'''
Output:
Akurasi Model Regresi Logistik: 93.0 persen
Akurasi Model Decision Tree: 89.0 persen
Akurasi Model KNN: 94.0 persen

Simpulan:
algoritma KNN memiliki akurasi terbaik
'''
