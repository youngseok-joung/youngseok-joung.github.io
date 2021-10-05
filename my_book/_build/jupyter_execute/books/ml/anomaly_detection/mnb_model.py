#!/usr/bin/env python
# coding: utf-8

# # Gaussian naive bayes classifier
# 
# In statistics, naive Bayes classifiers are a family of simple "probabilistic classifiers" based on applying Bayes' theorem with strong (naïve) independence assumptions between the features (see Bayes classifier). They are among the simplest Bayesian network models,[1] but coupled with kernel density estimation, they can achieve higher accuracy levels.[2][3]
# 
# Naïve Bayes classifiers are highly scalable, requiring a number of parameters linear in the number of variables (features/predictors) in a learning problem. Maximum-likelihood training can be done by evaluating a closed-form expression,[4]: 718  which takes linear time, rather than by expensive iterative approximation as used for many other types of classifiers.
# 
# In the statistics and computer science literature, naive Bayes models are known under a variety of names, including simple Bayes and independence Bayes.[5] All these names reference the use of Bayes' theorem in the classifier's decision rule, but naïve Bayes is not (necessarily) a Bayesian method.[4][5] (https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
# 
# 
# ## argmin/argmax
# Arguments of min, arguments of max, meaning that domains which make a function maximum or minimum. (https://www.latex4technics.com/?note=PMQWIE)
# (https://kapeli.com/cheat_sheets/LaTeX_Math_Symbols.docset/Contents/Resources/Documents/index)
# (https://www.math-linux.com/latex-26/faq/latex-faq/article/latex-derivatives-limits-sums-products-and-integrals)
# 
# For example, the following equation means the value of x where f(x) has the minimum value.
# 
# $\argmax\limits_x f(x)$ 
# 
# If f(x) is $y = x^2 + 3x - 2$, it has a minimum value of ${\dfrac{10}{4}}$ at ${x = -\dfrac{3}{2}}$.
# 
# $\argmax\limits_x f(x) = -\dfrac{3}{2}$
# 
# ## Constructing a classifier from the probability model
# The discussion so far has derived the independent feature model, that is, the naïve Bayes probability model. The naïve Bayes classifier combines this model with a decision rule. One common rule is to pick the hypothesis that is most probable; this is known as the maximum a posteriori or MAP decision rule. The corresponding classifier, a Bayes classifier, is the function that assigns a class label ${\displaystyle {\hat {y}}=C_{k}}{\hat  {y}}=C_{k}$ for some k as follows:
# 
# $\hat{y} = \argmax\limits_{k \in \mathcal{\{1, ..., K\}}} p(C_k) \prod\limits_{i=1}^{n} p(x_i | C_k)$
# 

# In[1]:


"""gnb_model.py: 

This model is the implementation of Gaussian Naive Bayes Classification of KDD datasets.
"""

__author__ = 'Youngseok Joung'
__copyright__ = "Copyright 2007, The Cogent Project"
__credits__ = ["Youngseok Joung"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Youngseok Joung"
__email__ = "none"
__status__ = "Production"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as splitter
from sklearn.naive_bayes import MultinomialNB
import cProfile
import pstats
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
from joblib import dump, load


# In[2]:


def labelEncoding(model_name, data):
    for column in data.columns:
        # If the data type of the cell is 'object'(Categorical), it will be transformed as a numerical 
        if data[column].dtype == type(object):
            le_file_path = 'result/' + model_name + '/' + model_name + '_' + column + '_encoder.pkl'
#             print(os.path.exists(le_file_path))
            if os.path.exists(le_file_path):
                pkl_file = open(le_file_path, 'rb')
                le = pickle.load(pkl_file) 
                pkl_file.close()
                data[column] = le.transform(data[column])            
            else:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column])
                #exporting the departure encoder
                output = open(le_file_path, 'wb')
                pickle.dump(le, output)
                output.close()
            if column == 'result':
                le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                print(le_name_mapping)
                
    return data, le


# In[3]:


def Preprocessing(model_name, data):
    y = data.result
    x = data.drop('result', axis=1)
    
    # Preprocessing: Split 7:3 Train: Test
    x_train, x_test, y_train, y_test = splitter(x, y, test_size=0.3)
    
    return x_train, x_test, y_train, y_test


# In[4]:


def train_and_test(model_name, x_train, x_test, y_train, y_test):
    # Profile: Start 
    profile = cProfile.Profile()
    profile.enable()
    
    # train and test
    model = MultinomialNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

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


# In[5]:


model_name = 'mnb_kdd'
# model_name = 'xgboost_nsl_kdd'
dataset_name = 'kdd_prediction'
# dataset_name = 'kdd_prediction_NSL'

data = pd.read_csv('./dataset/' + dataset_name + '.csv', delimiter=',', dtype={'protocol_type': str, 'service': str, 'flag': str, 'result': str})
print(data.head)


# In[6]:


# labeling
data, _ = labelEncoding(model_name, data)


# In[7]:


# Preprocessing
x_train, x_test, y_train, y_test = Preprocessing(model_name, data)


# In[8]:


# Train and Test
cm, cr = train_and_test(model_name, x_train, x_test, y_train, y_test)
print('\n-----Confusion Matrix-----\n')
print(cm)
print('\n-----Classification Report-----\n')
print(cr)


# In[ ]:


def production(model_name, data):
    real_data, le = labelEncoding(model_name, data)
    real_y = real_data.result
    real_x = real_data.drop('result', axis=1)
#     print(real_y)
#     print(real_x)

    clf = load('result/' + model_name + '/' + model_name + '_model.joblib')
    yy_pred = clf.predict(real_x)
    pred_label = le.inverse_transform(yy_pred)
    real_label = le.inverse_transform(real_y)

    return pred_label, real_label


# In[ ]:


# Production
real_data = pd.read_csv('./dataset/kdd_prediction.csv', delimiter=',', dtype={'protocol_type': str, 'service': str, 'flag': str, 'result': str})
real_data = real_data.head(1)

pred_label, real_label = production(model_name, real_data)
print(pred_label, real_label)


# In[ ]:




