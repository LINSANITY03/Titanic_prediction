from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

import tensorflow as tf

# Load dataset.
dftrain = pd.read_csv(
    'https://storage.googleapis.com/tf-datasets/titanic/train.csv')  # training data
dfeval = pd.read_csv(
    'https://storage.googleapis.com/tf-datasets/titanic/eval.csv')  # testing data
y_train = dftrain.pop('survived')  # get survived column data from dataframe
y_eval = dfeval.pop('survived')  # get survived column data from dataframe

'''print(tf.__version__)  # tf version 2.13.0
print(dftrain.head())  # return 5 items in our dataframe
print(dftrain.describe())  # getting statistical data
print(dftrain.shape)  # (627, 9) getting the rows and column
print(y_train.head())  # bool representation of survived in true or false'''
'''plt.hist(dftrain.age, bins=50)  # shows histogram with each plot holding 50 data points'''
'''dftrain.age.hist(bins=20)
dftrain.sex.value_counts().plot(kind='barh') # counts total male and female people'''
# dftrain['class'].value_counts().plot(kind='barh')
pd.concat([dftrain, y_train], axis=1).groupby(
    'sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
# combine/join dftrain and y-train to produce complete dataset then grouby 'sex' column to plot a survived average
plt.show()
