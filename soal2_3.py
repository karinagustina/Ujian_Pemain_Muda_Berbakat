import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================
# Preprocessing
# =========================================

df = pd.read_csv('data.csv')
dfFifa = df[['Name','Age','Overall','Potential']]

dftarget = dfFifa[dfFifa['Age'] <= 25][dfFifa['Overall'] >= 80][dfFifa['Potential'] >= 80]
labeltarget = np.array([1]*len(dftarget))
dftarget['Label'] = labeltarget

dfnontarget = dfFifa.drop(dftarget.index)
labelnontarget = np.array([0]*len(dfnontarget))
dfnontarget['Label'] = labelnontarget

dflabeled = dftarget.append(dfnontarget, ignore_index = True)

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
#KNN Algorithm
#================================================

#Define k_value
def k_value():
    k = round((len(x_train)+len(x_test)) ** .5)
    if (k % 2 == 0):
        return k + 1
    else:
        return k
# print(k_value())

#Import KNN Model
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(               
    n_neighbors = k_value()
)

#Fitting Model
model.fit(x_train, y_train)

#Accuracy
# print(model.score(x_test, y_test))

#================================================
#Prediction
#================================================

# print(model.predict([['Age', 'Overall', 'Potential']]))

#Andik Vermansyah	
print(model.predict([[27, 87, 90]]))    #output 0 => tidak direkrut

#Awan Setho Raharjo	
print(model.predict([[22, 75, 83]]))    #output 0 => tidak direkrut

# Bambang Pamungkas
print(model.predict([[38, 85, 75]]))    #output 0 => tidak direkrut

# Cristian Gonzales	
print(model.predict([[43, 90, 85]]))    #output 0 => tidak direkrut

# Egy Maulana Vikri	
print(model.predict([[18, 88, 90]]))    #output 1 => direkrut

# Evan Dimas	
print(model.predict([[24, 85, 87]]))    #output 1 => direkrut

# Febri Hariyadi	
print(model.predict([[23, 77, 80]]))    #output 0 => tidak direkrut

# Hansamu Yama Pranata	
print(model.predict([[24, 82, 85]]))    #output 1 => direkrut

# Septian David Maulana	
print(model.predict([[22, 83, 80]]))    #output 0 => tidak direkrut

# Stefano Lilipaly	
print(model.predict([[29, 88, 86]]))    #output 0 => tidak direkrut
