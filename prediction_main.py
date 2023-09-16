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
'''# dftrain['class'].value_counts().plot(kind='barh')
pd.concat([dftrain, y_train], axis=1).groupby(
    'sex').survived.mean().plot(kind='barh').set_xlabel('% survive')
# combine/join dftrain and y-train to produce complete dataset then grouby 'sex' column to plot a survived average
plt.show()'''

# we need to extract the feature from the given dataset
# we are splitting the feature using numeric and non-numeric set
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
# we use the inbuilt function in tensorflow to get all the unique value represented in the data of certain features
for feature_name in CATEGORICAL_COLUMNS:
    # gets a list of all unique values from given feature column
    vocabulary = dftrain[feature_name].unique()
    print(vocabulary)
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(
        feature_name, vocabulary))

# similar for the numeric ones we get the features in float format
for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(
        feature_name, dtype=tf.float32))

print(feature_columns)
'''[VocabularyListCategoricalColumn(key='sex', vocabulary_list=('male', 'female'), dtype=tf.string, default_value=-1, num_oov_buckets=0), VocabularyListCategoricalColumn(key='n_siblings_spouses', vocabulary_list=(1, 0, 3, 4, 2, 5, 8), dtype=tf.int64, default_value=-1, num_oov_buckets=0), VocabularyListCategoricalColumn(key='parch', vocabulary_list=(0, 1, 2, 5, 3, 4), dtype=tf.int64, default_value=-1, num_oov_buckets=0), VocabularyListCategoricalColumn(key='class', vocabulary_list=('Third', 'First', 'Second'), dtype=tf.string, default_value=-1, num_oov_buckets=0), VocabularyListCategoricalColumn(key='deck', vocabulary_list=('unknown', 'C', 'G', 'A', 'B', 'D', 'F', 'E'), dtype=tf.string, default_value=-1, num_oov_buckets=0), VocabularyListCategoricalColumn(key='embark_town', vocabulary_list=('Southampton', 'Cherbourg', 'Queenstown', 'unknown'), dtype=tf.string, default_value=-1, num_oov_buckets=0), VocabularyListCategoricalColumn(key='alone', vocabulary_list=('n', 'y'), dtype=tf.string, default_value=-1, num_oov_buckets=0), NumericColumn(key='age', shape=(1,), 
default_value=None, dtype=tf.float32, normalizer_fn=None), NumericColumn(key='fare', shape=(1,), default_value=None, dtype=tf.float32, normalizer_fn=None)]'''
