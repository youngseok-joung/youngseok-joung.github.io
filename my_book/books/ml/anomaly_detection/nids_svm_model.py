# %%

__version__ = "0.1"
__author__ = 'Youngseok Joung'

import pandas as pd
from sklearn.model_selection import train_test_split as splitter
from sklearn.svm import SVC
import cProfile
import pstats
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load


# %%

def train_and_test(model_name, data):
    print(data)
    for column in data.columns:
        if data[column].dtype == type(object):
            le = LabelEncoder()
            data[column] = le.fit_transform(data[column])
    encodedData = data
    print(encodedData)

    y = data.result
    x = data.drop('result', axis=1)

    # Preprocessing: Split 7:3 Train: Test
    x_train, x_test, y_train, y_test = splitter(x, y, test_size=0.3)

    # Profile: Start
    profile = cProfile.Profile()
    profile.enable()

    # train and test
    model = SVC()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # Profile: End
    profile.disable()
    profile.dump_stats('output.prof')
    stream = open('result/' + model_name + '_profiling.txt', 'w')
    stats = pstats.Stats('output.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()
    os.remove('output.prof')

    # Estimation: Confusion Matrix & classification-report
    conf_matrix = confusion_matrix(y_test, y_pred)
    classif_report = classification_report(y_test, y_pred)

    f = open('result/' + model_name + '_output.txt', 'w')
    f.write("---Confusion Matrix---")
    f.write(conf_matrix)
    f.write("---Classification Report---")
    f.write(classif_report)
    f.close()

    dump(model, 'result/' + model_name + '_model.joblib')
    return x_test, y_test


# %%

model_name = 'svm_kdd'
data = pd.read_csv('./dataset/kdd_prediction.csv', delimiter=',',
                   dtype={'protocol_type': str, 'service': str, 'flag': str, 'result': str})
print(data.head)

# %%

x_test, y_test = train_and_test(model_name, data)
# data = pd.read_csv('./dataset/kdd_prediction_NSL.csv', delimiter=',', dtype={'protocol_type': str, 'service': str, 'flag': str, 'result': str})
# train_and_test('svm_nsl_kdd', data)


# %%

clf = load('result/' + model_name + '_model.joblib')
# print(x_test['protocol_type'].iloc[0])
# print(x_test.head)
# print(y_test.iloc[0])
# yy_pred = clf.predict(x_test.iloc[0])
yy_pred = clf.predict(x_test.head(1))
print(yy_pred)
print(y_test.head(1))

# %%


# %%


# %%


