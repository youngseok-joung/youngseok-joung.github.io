#!/usr/bin/env python
# coding: utf-8

# # Support-vector machine
# 
# ## I want to classify something well!
# > I can classify real objects in real world by my eyes and hands.
# > How about entangible things like numbers, positions, attributes, types
# 
# ### Which shape is the best in a specific Dimension?
# 
# 1. How could we devide two points in a 1D by a point?
# Imagine that if there are different two points in a line.
# Which position of a new point is proper to divide the two points below?
# 
# ![1d.PNG](./assets/images/1d.PNG)
# 
# 2. How could we devide two points in a 2D by a line?
# 
# ![2d.PNG](./assets/images/2d.PNG)
# 
# 
# 3. How could we devide two points in a 3D by a plane?
# 
# ![3d.PNG](./assets/images/3d.PNG)
# 
# ### Where is the best position of the shape? 
# 
# 1. The place Where it is the middle of the two points.
# 
# 
# ## Support-vector classification
# More formally, a support-vector machine **constructs a hyperplane** or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks like outliers detection.[3] 
# 
# > Intuitively, a good separation is achieved by *he hyperplane that has the largest distance to the nearest training-data point of any class (so-called functional margin), since in general the larger the margin, the lower the generalization error of the classifier.[4]
# 
# The objective of the support vector machine algorithm is to find a hyperplane(N-1 D Subspace) in an N-dimensional space(N — the number of features) that distinctly classifies the data points. (https://en.wikipedia.org/wiki/Support-vector_machine)
# 
# ### Hyperplane 
# In geometry, a hyperplane is a subspace whose dimension is **one less** than that of its ambient space. If a space is 3-dimensional then its hyperplanes are the 2-dimensional planes, while if the space is 2-dimensional, its hyperplanes are the 1-dimensional lines. (https://en.wikipedia.org/wiki/Hyperplane)
# 
# ### Maximum-margin 
# Maximum-margin hyperplane and margins for an SVM trained with samples from two classes. Samples on the margin are called the support vectors. (https://en.wikipedia.org/wiki/Support-vector_machine)
# 
# ![SVM_margin.PNG](./assets/images/SVM_margin.PNG)
# 
# 
# ### Problem: Not linearly separable in that space (Curved like below)
# 1. Kernal function is to keep the computational load reasonable, the mappings used by SVM schemes are designed to ensure that dot products of pairs of input data vectors may be computed easily in terms of the variables in the original space
# 
# ![kernal_function.PNG](./assets/images/1920px-Kernel_Machine.svg.PNG)
# 
# 2.  a set of vectors is an orthogonal (and thus minimal) set of vectors that defines a hyperplane. The vectors defining the hyperplanes can be chosen to be linear combinations with parameters {\displaystyle \alpha _{i}}\alpha _{i} of images of feature vectors {\displaystyle x_{i}}x_{i} that occur in the data bas
# 
# ### Hinge Loss function
# To extend SVM to cases in which the data are not linearly separable, the hinge loss function is helpful
# 
# ![hinge_loss.PNG](./assets/images/hinge_loss.PNG)
# 
# The goal of the optimization then is to minimize
# 
# ![optimization_function.PNG](./assets/images/optimization_function.PNG)
# 

# In[1]:


"""svm_model.py: 

This model is the implementation of Classification of KDD datasets.
"""

__author__ = 'Youngseok Joung'
__copyright__ = "Copyright 2021, The Cogent Project"
__credits__ = ["Youngseok Joung"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Youngseok Joung"
__email__ = "none"
__status__ = "Production"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as splitter
from sklearn.svm import SVC
import cProfile
import pstats
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot
import pickle
from joblib import dump, load


# In[2]:


def labelEncoding(model_name, data):
    for column in data.columns:
        # If the data type of the cell is 'object'(Categorical), it will be transformed as a numerical 
        if data[column].dtype == type(object):
            le_file_path = 'result/' + model_name + '/' + model_name + '_' + column + '_encoder.pkl'
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
    model = SVC()
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
    
    # Freezing model for production 
    dump(model, 'result/' + model_name + '/' + model_name + '_model.joblib') 
    
    return model, y_pred


# In[5]:


def report(model_name, y_test, y_pred, le=None):
    # Estimation: Confusion Matrix & classification-report 
    _confusion_matrix = confusion_matrix(y_test, y_pred)
    _classification_report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=False)
    _classification_report_dict = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

    # For Multiclass AUC
    _auc_dict = roc_auc_score_multiclass(y_test, y_pred)
    _auc_dict = dict((le.classes_[key], value) for (key, value) in _auc_dict.items())
#     _auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
#     _fpr, _tpr, _thresholds = roc_curve(y_test, y_pred)

    with open('result/' + model_name + '/' + model_name + '_output.txt', 'w') as f:
        f.write("\n---Confusion Matrix---\n")
        f.write(np.array2string(_confusion_matrix, separator=', '))
        f.write("\n---Classification Report---\n")
        f.write(_classification_report)
        f.write("\n---ROC AUC Score---\n")
        f.write(str(_auc_dict))
#         f.write(_auc)
    
    print('\n-----Confusion Matrix-----\n')
    print(_confusion_matrix)
    print('\n-----Classification Report-----\n')
    print(_classification_report)
    print('\n-----AUC Dictionary-----\n')
    print(str(_auc_dict))
    
    metrix = ['precision', 'recall', 'f1-score', 'support']
    xKeys = le.classes_
    for met in metrix:
        xValues = []
        for target_name in le.classes_:
            xValues += [_classification_report_dict[target_name][met]]

        pyplot.title(met)
        pyplot.bar(range(len(xValues)), list(xValues), align='center')
        pyplot.xticks(range(len(xKeys)), list(xKeys))
        pyplot.show()

    pyplot.title(met)
    pyplot.bar(range(len(xValues)), list(xValues), align='center')
    pyplot.xticks(range(len(xKeys)), list(xKeys))
    pyplot.show()
    
    pyplot.title('AUC')
    pyplot.bar(range(len(_auc_dict)), list(_auc_dict.values()), align='center')
    pyplot.xticks(range(len(_auc_dict)), list(_auc_dict.keys()))
    pyplot.show()
    
#     # plot the roc curve for the model
#     # pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
#     pyplot.plot(_fpr, _tpr, marker='.', label=model_name)
#     # axis labels
#     pyplot.xlabel('False Positive Rate')
#     pyplot.ylabel('True Positive Rate')
#     # show the legend
#     pyplot.legend()
#     # show the plot
#     pyplot.show()
    
    return _confusion_matrix, _classification_report, _auc_dict, _classification_report_dict
    


# In[6]:


def roc_auc_score_multiclass(y_test, y_pred, average = "macro"):
    #creating a set of all the unique classes using the actual class list
    unique_class = set(y_test)
    roc_auc_dict = {}
    for per_class in unique_class:
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_y_test = [0 if x in other_class else 1 for x in y_test]
        new_y_pred = [0 if x in other_class else 1 for x in y_pred]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_y_test, new_y_pred, average = average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict


# In[7]:


model_name = 'svm_kdd'
# model_name = 'svm_nsl_kdd'
dataset_name = 'kdd_prediction'
# dataset_name = 'kdd_prediction_NSL'

data = pd.read_csv('./dataset/' + dataset_name + '.csv', delimiter=',', dtype={'protocol_type': str, 'service': str, 'flag': str, 'result': str})
print(data.head)


# In[8]:


# labeling
data, le = labelEncoding(model_name, data)


# In[9]:


# Preprocessing
x_train, x_test, y_train, y_test = Preprocessing(model_name, data)


# In[10]:


# Train and Test
model, y_pred = train_and_test(model_name, x_train, x_test, y_train, y_test)
# Report
cm, cr, auc, _ = report(model_name, y_test, y_pred, le)


# # Test in Product 

# In[11]:


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


# In[12]:


# Production
real_data = pd.read_csv('./dataset/kdd_prediction.csv', delimiter=',', dtype={'protocol_type': str, 'service': str, 'flag': str, 'result': str})
real_data = real_data.head(1)

pred_label, real_label = production(model_name, real_data)
print(pred_label, real_label)


# In[ ]:



