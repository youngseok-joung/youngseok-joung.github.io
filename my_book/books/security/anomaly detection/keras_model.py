# %%

__version__ = "0.1"
__author__ = 'Youngseok Joung'

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as splitter
from keras import regularizers
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import cProfile
import pstats
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
from joblib import dump, load


# %%

def labelEncoding(model_name, data):
    for column in data.columns:
        # If the data type of the cell is 'object'(Categorical), it will be transformed as a numerical
        if data[column].dtype == type(object):
            le_file_path = 'result/' + model_name + '/' + model_name + '_' + column + '_encoder.pkl'
            print(os.path.exists(le_file_path))
            if os.path.exists(le_file_path):
                pkl_file = open(le_file_path, 'rb')
                le = pickle.load(pkl_file)
                pkl_file.close()
                data[column] = le.transform(data[column])
            else:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])
                # exporting the departure encoder
                output = open(le_file_path, 'wb')
                pickle.dump(le, output)
                output.close()
            if column == 'result':
                le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                print(le_name_mapping)

    return data, le


# %%

def Preprocessing(model_name, data):
    y = data.result
    x = data.drop('result', axis=1)

    # Preprocessing: Split 7:3 Train: Test
    x_train, x_test, y_train, y_test = splitter(x, y, test_size=0.3)

    return x_train, x_test, y_train, y_test


# %%

def train_and_test(model_name, x_train, x_test, y_train, y_test):
    # Profile: Start
    profile = cProfile.Profile()
    profile.enable()

    # train and test
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    val_indices = 200
    x_val = x_train[-val_indices:]
    y_val = y_train[-val_indices:]
    # train and test
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_dim=x_train.shape[1], kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=15, batch_size=512, validation_data=(x_val, y_val))
    y_pred = model.predict(x_test)

    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    # Profile: End
    profile.disable()
    profile.dump_stats('output.prof')
    stream = open('result/' + model_name + '/' + model_name + '_profiling.txt', 'w')
    stats = pstats.Stats('output.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()
    os.remove('output.prof')

    # Estimation: Confusion Matrix & classification-report
    _confusion_matrix = confusion_matrix(y_test, y_pred)
    _classification_report = classification_report(y_test, y_pred)

    with open('result/' + model_name + '/' + model_name + '_output.txt', 'w') as f:
        f.write("\n---Confusion Matrix---\n")
        f.write(np.array2string(_confusion_matrix, separator=', '))
        f.write("\n---Classification Report---\n")
        f.write(_classification_report)

    # Freezing model for production
    dump(model, 'result/' + model_name + '/' + model_name + '_model.joblib')

    return _confusion_matrix, _classification_report


# %%

model_name = 'keras_kdd'
# model_name = 'keras_nsl_kdd'
dataset_name = 'kdd_prediction'
# dataset_name = 'kdd_prediction_NSL'

data = pd.read_csv('./dataset/' + dataset_name + '.csv', delimiter=',',
                   dtype={'protocol_type': str, 'service': str, 'flag': str, 'result': str})
print(data.head)

# %%

# labeling
data, _ = labelEncoding(model_name, data)

# %%

# Preprocessing
x_train, x_test, y_train, y_test = Preprocessing(model_name, data)

# %%

# Train and Test
cm, cr = train_and_test(model_name, x_train, x_test, y_train, y_test)
print('\n-----Confusion Matrix-----\n')
print(cm)
print('\n-----Classification Report-----\n')
print(cr)


# %%

def production(model_name, data):
    real_data, le = labelEncoding(model_name, data)
    real_y = real_data.result
    real_x = real_data.drop('result', axis=1)
    print(real_y)
    print(real_x)

    clf = load('result/' + model_name + '/' + model_name + '_model.joblib')
    yy_pred = clf.predict(real_x)
    print(yy_pred)
    real_label = le.inverse_transform(yy_pred)

    return real_label


# %%

# Production
real_data = pd.read_csv('./dataset/kdd_prediction.csv', delimiter=',',
                        dtype={'protocol_type': str, 'service': str, 'flag': str, 'result': str})
real_data = real_data.head(1)
print(real_data)

expected_label = production(model_name, real_data)
print(expected_label)

# %%


