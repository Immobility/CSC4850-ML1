import numpy as np
import sklearn
import pandas as pd
import os
from numpy.random import randint

pd.set_option('display.float_format', lambda x: '%.8f' % x)
df = pd.read_csv('~/CSC4850-ML1/csv_files/TrainData1.csv')
df.values[df > 10] = 0
print(df)
