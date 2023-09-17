# Prediction model of surviving possibilities

In this project we use pandas to collect data from a source and in-built tensorflow model to train the data.

To run this project,

- Create a virutal environment and activate the environment.
  `virtualenv venv
\venv\Scripts\activate`

- Run the **prediction_main.py** file.
  `python prediction_main.py`

**1. Data Collection**
We get titanic training and evaluation data from google drive links.

import pandas as pd
...

dftrain = pd.read_csv(
'https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data

dfeval = pd.read_csv(
'https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data

**2. Feature Extraction**
Using the in-built feature column function of tensorflow, we get all the unique value from each column of the pandas file.

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
'embark_town', 'alone']

NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []

# we use the inbuilt function in tensorflow to get all the unique value represented in the data of certain features

for feature_name in CATEGORICAL_COLUMNS: # gets a list of all unique values from given feature column
vocabulary = dftrain[feature_name].unique()
feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(
feature_name, vocabulary))

# similar for the numeric ones we get the features in float format

for feature_name in NUMERIC_COLUMNS:
feature_columns.append(tf.feature_column.numeric_column(
feature_name, dtype=tf.float32))

**3. Data Preparation**
We need to make sure the data are in appropritate format for the tensorflow model. So, we convert the datas into data.Dataset object using tf.data.Dataset function

```
# create tf.data.Dataset object with data and its label
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))

```

**4. Choosing a Model**
Our goal is to predict the chance of survivility. So, a simple linear model would do the trick.

    ```
    linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
    ```

**5. Training the model**
We use the data we convert to data.Dataset object to the model.

    ```
    linear_est.train(train_input_fn)  # train
    ```

**6. Evaluate the model**
Test the unseen dataset to measure the performance of the trained model.

    ```
    result = linear_est.evaluate(eval_input_fn)
    ```

**7. Make prediction**
Using the evaluated model predict the survivor possibilty and plot the stats into graph using matplot for better readability.

    ```
    pred_dicts = list(linear_est.predict(eval_input_fn))

    probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
    probs.plot(kind='hist', bins=20, title='predicted probabilities')
    plt.show()

    ```
