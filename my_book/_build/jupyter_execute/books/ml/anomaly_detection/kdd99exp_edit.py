#!/usr/bin/env python
# coding: utf-8

# # KDD99 Test code 

# In[1]:


import pandas
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss


# In[2]:


# Must declare data_dir as the directory of training and test files
data_dir = "./dataset/"
train_data = data_dir + "kddcup.data.corrected"
train_labels = data_dir + "train_labels.txt"
test_data = data_dir + "corrected"
test_labels = data_dir + "test_labels.txt"


# In[3]:


def process_data(X, y):
    X = X.drop(41, 1)
    X[1], uniques = pandas.factorize(X[1])
    X[2], uniques = pandas.factorize(X[2])
    X[3], uniques = pandas.factorize(X[3])

    num_examples = 10**6
    X = X[0:num_examples]
    y = y[0:num_examples]

    X = numpy.array(X)
    y = numpy.array(y).ravel()

    return X, y


# In[4]:


print("Loading training data")
train_X = pandas.read_csv(train_data, header=None)
train_y = pandas.read_csv(train_labels, header=None)
train_X, train_y = process_data(train_X, train_y)


# In[ ]:


print("Loading test data")
test_X = pandas.read_csv(test_data, header=None)
test_y = pandas.read_csv(test_labels, header=None)
test_X, test_y = process_data(test_X, test_y)


# In[ ]:


print("Training and predicting")
learner = KNeighborsClassifier(1, n_jobs=-1)
learner.fit(train_X, train_y)
pred_y = learner.predict(test_X)


# In[ ]:


results = confusion_matrix(test_y, pred_y)
error = zero_one_loss(test_y, pred_y)
print(results)
print(error)


# In[ ]:





# In[ ]:




