#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


train= pd.read_csv('dataset/1 - 20 Percent Training Set.csv', index_col=False, header=0);
test = pd.read_csv('dataset/Test - KDDTest.csv', index_col=False, header=0);


# In[3]:


train.describe()


# In[4]:


train.head()


# In[5]:


#2 values make no contextual sense
train.su_attempted = train.su_attempted.replace(2,0)


# In[6]:


print('target distribution Training set:')
print(train['target'].value_counts())
print('target distribution Test set:')
print(test['target'].value_counts())


# One-Hot Encoding

# In[7]:


#ONE-HOT ENCODING OF CATEGORICAL VARIABLES
train_target = train["target"]
test_target = test["target"]
train = train.drop("target" , axis =1)
test = test.drop('target', axis =1)
train = pd.get_dummies(train)
test = pd.get_dummies(test)


print('Training Features shape: ', train.shape)
print('Testing Features shape: ', test.shape)


# In[8]:


#aligning train and test sets to have same features
train['target'] = train_target
test['target'] = test_target


train, test = train.align(test, join = 'inner', axis = 1)


print('Training Features shape: ', train.shape)
print('Testing Features shape: ', test.shape)


# In[9]:


# take target column
targetdf=train['target']
targetdf_test=test['target']
# change the target column
newtargetdf=targetdf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newtargetdf_test=targetdf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
# put the new target column back
train['target'] = newtargetdf
test['target'] = newtargetdf_test
print(train['target'].head())


# In[10]:


to_drop_DoS = [2,3,4]
to_drop_Probe = [1,3,4]
to_drop_R2L = [1,2,4]
to_drop_U2R = [1,2,3]
to_drop_normal = [1,2,3,4]
DoS_df=train[~train['target'].isin(to_drop_DoS)];
Probe_df=train[~train['target'].isin(to_drop_Probe)];
R2L_df=train[~train['target'].isin(to_drop_R2L)];
U2R_df=train[~train['target'].isin(to_drop_U2R)];

#test
DoS_df_test=test[~test['target'].isin(to_drop_DoS)];
Probe_df_test=test[~test['target'].isin(to_drop_Probe)];
R2L_df_test=test[~test['target'].isin(to_drop_R2L)];
U2R_df_test=test[~test['target'].isin(to_drop_U2R)];


print('Train:')
print('Dimensions of DoS:' ,DoS_df.shape)
print('Dimensions of Probe:' ,Probe_df.shape)
print('Dimensions of R2L:' ,R2L_df.shape)
print('Dimensions of U2R:' ,U2R_df.shape)

print('Test:')
print('Dimensions of DoS:' ,DoS_df_test.shape)
print('Dimensions of Probe:' ,Probe_df_test.shape)
print('Dimensions of R2L:' ,R2L_df_test.shape)
print('Dimensions of U2R:' ,U2R_df_test.shape)


# Finding Correlations for Feature Engineering

# In[11]:



# Find correlations with the target and sort
DoS_correlations = DoS_df.corr()['target'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', DoS_correlations.tail(15))
print('\nMost Negative Correlations:\n', DoS_correlations.head(15))


# In[12]:


# Find correlations with the target and sort
Probe_correlations = Probe_df.corr()['target'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', Probe_correlations.tail(10))
print('\nMost Negative Correlations:\n', Probe_correlations.head(10))


# In[13]:


# Find correlations with the target and sort
U2R_correlations = U2R_df.corr()['target'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', U2R_correlations.tail(10))
print('\nMost Negative Correlations:\n', U2R_correlations.head(10))


# In[14]:


# Find correlations with the target and sort
R2L_correlations = R2L_df.corr()['target'].sort_values()

# Display correlations
print('Most Positive Correlations:\n', R2L_correlations.tail(10))
print('\nMost Negative Correlations:\n', R2L_correlations.head(10))


# In[15]:


def polynomial_terms_maker(a,b):
    
    corr = a.corr()['target'].sort_values()
    poly_features_head = [*dict(corr.head(15))]
    poly_features_tail = [*dict(corr.tail(15))]
    poly_features = poly_features_head+(poly_features_tail)
    poly_features_test=b[poly_features]
    poly_features = a[poly_features]    
    poly_features = poly_features
    if "target" in poly_features:
        poly_features_test = poly_features_test.drop("target",axis =1)   
        poly_features = poly_features.drop("target",axis =1)    
    poly_names= list(poly_features.columns)
    poly_target = a["target"]
    poly_target_test =b["target"]
    from sklearn.preprocessing import PolynomialFeatures
                                  
    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree = 3)

    # Train the polynomial features 
    poly_transformer.fit(poly_features)

    # Transform the features
    poly_features = poly_transformer.transform(poly_features)
    poly_features_test = poly_transformer.transform(poly_features_test)
    #print('Polynomial Features shape: ', poly_features.shape)

    poly_transformer.get_feature_names(input_features = poly_names)
    
    # Create a dataframe of the features 
    poly_features = pd.DataFrame(poly_features, 
                             columns = poly_transformer.get_feature_names(poly_names))
    # Put test features into dataframe
    poly_features_test = pd.DataFrame(poly_features_test, 
                                  columns = poly_transformer.get_feature_names(poly_names))
    # Add in the target
    poly_features['target'] = poly_target

    a['index_col'] = a.index
    b['index_col'] = b.index

    # Merge polynomial features into training dataframe
    poly_features['index_col'] = a['index_col']
    poly = a.merge(poly_features, on = 'index_col', how = 'left')

    # Merge polnomial features into testing dataframe
    poly_features_test['index_col'] = b['index_col']
    test_poly = b.merge(poly_features_test, on = 'index_col', how = 'left')
    
    

    # Align the dataframes
    poly, test_poly = poly.align(test_poly, join = 'inner', axis = 1)
    
        
    #Drop NaN columns
    poly= poly.dropna(axis=1,how="any")
    test_poly= test_poly.dropna(axis=1,how="any")
    
    return poly,test_poly,poly_target,poly_target_test
    # Print out the new shapes
    #print('Training data with polynomial features shape: ', DoS_df.shape)
    #print('Testing data with polynomial features shape:  ', DoS_df_test.shape)

print(polynomial_terms_maker(DoS_df,DoS_df_test))


# In[16]:


# Split dataframes into X & Y
# assign X as a dataframe of feautures and Y as a series of outcome variables
X_DoS, X_DoS_test, Y_DoS, Y_DoS_test= polynomial_terms_maker(DoS_df,DoS_df_test)
X_Probe,X_Probe_test,Y_Probe,Y_Probe_test =polynomial_terms_maker(Probe_df,Probe_df_test)
X_R2L,X_R2L_test,Y_R2L,Y_R2L_test =polynomial_terms_maker(R2L_df,R2L_df_test)
X_U2R,X_U2R_test,Y_U2R,Y_U2R_test =polynomial_terms_maker(U2R_df,U2R_df_test)
X_DoS.shape


# In[17]:


print(X_R2L.shape)
print(X_DoS.shape)


# In[18]:


colNames=list(X_DoS)
colNames_test=list(X_DoS_test)


# In[19]:


print(len(colNames))
print(len(colNames_test))


# Scaling The Data

# In[20]:



from sklearn import preprocessing
scaler1 = preprocessing.StandardScaler().fit(X_DoS)
X_DoS=scaler1.transform(X_DoS) 
scaler2 = preprocessing.StandardScaler().fit(X_Probe)
X_Probe=scaler2.transform(X_Probe) 
scaler3 = preprocessing.StandardScaler().fit(X_R2L)
X_R2L=scaler3.transform(X_R2L) 
scaler4 = preprocessing.StandardScaler().fit(X_U2R)
X_U2R=scaler4.transform(X_U2R) 
# test data
scaler5 = preprocessing.StandardScaler().fit(X_DoS_test)
X_DoS_test=scaler5.transform(X_DoS_test) 
scaler6 = preprocessing.StandardScaler().fit(X_Probe_test)
X_Probe_test=scaler6.transform(X_Probe_test) 
scaler7 = preprocessing.StandardScaler().fit(X_R2L_test)
X_R2L_test=scaler7.transform(X_R2L_test) 
scaler8 = preprocessing.StandardScaler().fit(X_U2R_test)
X_U2R_test=scaler8.transform(X_U2R_test)


# In[21]:


print(X_Probe.std(axis=0))


# RFE for Feature Selection

# In[22]:


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_jobs=2)
rfe = RFE(estimator=clf, n_features_to_select=15, step=1)
rfe.fit(X_DoS, Y_DoS)
X_rfeDoS=rfe.transform(X_DoS)
true=rfe.support_
rfecolindex_DoS=[i for i, x in enumerate(true) if x]
rfecolname_DoS=list(colNames[i] for i in rfecolindex_DoS)


# In[23]:


rfe.fit(X_Probe, Y_Probe)
X_rfeProbe=rfe.transform(X_Probe)
true=rfe.support_
rfecolindex_Probe=[i for i, x in enumerate(true) if x]
rfecolname_Probe=list(colNames[i] for i in rfecolindex_Probe)


# In[24]:


rfe.fit(X_R2L, Y_R2L)
X_rfeR2L=rfe.transform(X_R2L)
true=rfe.support_
rfecolindex_R2L=[i for i, x in enumerate(true) if x]
rfecolname_R2L=list(colNames[i] for i in rfecolindex_R2L)


# In[25]:


rfe.fit(X_U2R, Y_U2R)
X_rfeU2R=rfe.transform(X_U2R)
true=rfe.support_
rfecolindex_U2R=[i for i, x in enumerate(true) if x]
rfecolname_U2R=list(colNames[i] for i in rfecolindex_U2R)


# In[26]:


print('Features selected for DoS:',rfecolname_DoS)
print('Features selected for Probe:',rfecolname_Probe)
print('Features selected for R2L:',rfecolname_R2L)
print('Features selected for U2R:',rfecolname_U2R)


# In[27]:


print(X_rfeDoS.shape)
print(X_rfeProbe.shape)
print(X_rfeR2L.shape)
print(X_rfeU2R.shape)


# In[28]:


# reduce test dataset to 15 features, use only features described in rfecolname_DoS etc.
X_DoS_test2=X_DoS_test[:,rfecolindex_DoS]
X_Probe_test2=X_Probe_test[:,rfecolindex_Probe]
X_R2L_test2=X_R2L_test[:,rfecolindex_R2L]
X_U2R_test2=X_U2R_test[:,rfecolindex_U2R]
print(X_DoS_test2.shape)
print(X_Probe_test2.shape)


# Algorithms after RFE

# In[32]:


from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.mixture import GaussianMixture

clfs = {
    'gnb': GaussianNB(),
    'gmm': GaussianMixture(),
#Not     'mnb': MultinomialNB(),
#Not     'cnb': CategoricalNB(),
#    'svm1': SVC(kernel='linear'),
#    'svm2': SVC(kernel='rbf'),
#     'svm3': SVC(kernel='sigmoid'),
#     'mlp1': MLPClassifier(),
#     'mlp2': MLPClassifier(hidden_layer_sizes=[100, 100]),
#    'ada': AdaBoostClassifier(),
#     'dtc': DecisionTreeClassifier(),
#     'rfc': RandomForestClassifier(),
#     'gbc': GradientBoostingClassifier(),
#     'lr': LogisticRegression()
}


# In[43]:


from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
def report(model_name, y_test, y_pred, le=None):
    """report function evaluates the quality of the output of a classifier on this data set.
    We can get the value of Precision, Recall,, F1-Score, Support, accuracy by Lables 
    And it can get Multiclass AUC score multiclass using roc_auc_score_multiclass function
    Additionally, it draws Bar graph about comparison between labels in each metrics (precision, recall, f1-score, AUC)
    All are saved as a file
    
    :param model_name: model name used in this project (e.g. "SVM")
    :param y_test: test label    
    :param y_pred: test label    
    :param le: None or Label encoder    
    :return: _confusion_matrix, _classification_report, _auc_dict, _classification_report_dict
    """
    
    # Estimation: Confusion Matrix & classification-report 
    _confusion_matrix = confusion_matrix(y_test, y_pred)
    _classification_report = classification_report(y_test, y_pred, output_dict=False)
#     print(_confusion_matrix)
    print(_classification_report)
    return _confusion_matrix, _classification_report
    


# In[44]:


f1_scores = dict()
for clf_name in clfs:
    print(clf_name)
    clf = clfs[clf_name]
    clf.fit(X_rfeDoS, Y_DoS)
    y_pred = clf.predict(X_DoS_test2)
    f1_scores[clf_name] = f1_score(y_pred, Y_DoS_test,average="weighted")
    cm, cr = report(clf_name, Y_DoS_test, y_pred)


# In[45]:


f1_scores


# In[46]:


accuracy={}
for i in clfs:
    accuracy[i] = clfs[i].score(X_DoS_test2, Y_DoS_test)
accuracy


# In[47]:


f1_scores = dict()
for clf_name in clfs:
    print(clf_name)
    clf = clfs[clf_name]
    clf.fit(X_rfeProbe, Y_Probe)
    y_pred = clf.predict(X_Probe_test2)
    f1_scores[clf_name] = f1_score(y_pred, Y_Probe_test,average="weighted")
    cm, cr = report(clf_name, Y_Probe_test, y_pred)


# In[48]:


f1_scores


# In[49]:


f1_scores = dict()
for clf_name in clfs:
    print(clf_name)
    clf = clfs[clf_name]
    clf.fit(X_rfeR2L, Y_R2L)
    y_pred = clf.predict(X_R2L_test2)
    f1_scores[clf_name] = f1_score(y_pred, Y_R2L_test,average="weighted")
    cm, cr = report(clf_name, Y_R2L_test, y_pred)


# In[50]:


f1_scores


# In[51]:


f1_scores = dict()
for clf_name in clfs:
    print(clf_name)
    clf = clfs[clf_name]
    clf.fit(X_rfeU2R, Y_U2R)
    y_pred = clf.predict(X_U2R_test2)
    f1_scores[clf_name] = f1_score(y_pred, Y_U2R_test,average="weighted")
    cm, cr = report(clf_name, Y_U2R_test, y_pred)


# In[52]:


f1_scores


# In[ ]:




