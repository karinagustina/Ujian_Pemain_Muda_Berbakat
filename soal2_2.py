import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
# print(dir(df))

dfFifa = df[['Name','Age','Overall','Potential']]
# print(dfFifa)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    dfFifa['data'],
    dfFifa['target'],
    test_size = .1
)