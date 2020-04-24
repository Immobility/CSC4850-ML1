import numpy as np
import sklearn
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from numpy.random import randint

pd.set_option('display.float_format', lambda x: '%.8f' % x)
dfx = pd.read_csv('~/CSC4850-ML1/csv_files/TrainData2.csv', header= None)
dfy = pd.read_csv('~/CSC4850-ML1/csv_files/TrainLabel2.csv', header= None)
dfx.values[dfx > 10] = 0

X_test = pd.read_csv('~/CSC4850-ML1/csv_files/TestData2.csv', header= None)

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(dfx, dfy.values.ravel())
print(knn.predict(X_test))
