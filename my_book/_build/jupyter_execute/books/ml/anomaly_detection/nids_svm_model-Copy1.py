#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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
    
    profile = cProfile.Profile()
    x_train, x_test, y_train, y_test = splitter(x, y, test_size=0.3)
    profile.enable()
    # train and test
    model = SVC()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    profile.disable()
    profile.dump_stats('output.prof')
    stream = open('result/' + model_name + '_profiling.txt', 'w')
    stats = pstats.Stats('output.prof', stream=stream)
    stats.sort_stats('cumtime')
    stats.print_stats()
    os.remove('output.prof')
    conf_matrix = confusion_matrix(y_test, y_pred)
    f = open('result/' + model_name + '_output.txt', 'w')
#     sys.stdout = f
    print(conf_matrix)
    print(classification_report(y_test, y_pred))
    dump(model, 'result/' + model_name + '_model.joblib') 
    return x_test, y_test


# In[3]:


model_name = 'svm_kdd'
print(1)
data = pd.read_csv('./dataset/kdd_prediction.csv', delimiter=',', dtype={'protocol_type': str, 'service': str, 'flag': str, 'result': str})
print(data.head)
print(2)


# In[4]:


x_test, y_test = train_and_test(model_name, data)
# data = pd.read_csv('./dataset/kdd_prediction_NSL.csv', delimiter=',', dtype={'protocol_type': str, 'service': str, 'flag': str, 'result': str})
# train_and_test('svm_nsl_kdd', data)


# In[5]:


clf = load('result/' + model_name + '_model.joblib')
# print(x_test['protocol_type'].iloc[0])
# print(x_test.head)
# print(y_test.iloc[0])
# yy_pred = clf.predict(x_test.iloc[0])
yy_pred = clf.predict(x_test.head(1))
print(yy_pred)
print(y_test.head(1))


# In[ ]:




