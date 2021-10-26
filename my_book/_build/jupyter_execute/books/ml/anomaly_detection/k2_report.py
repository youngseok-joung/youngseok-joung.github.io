#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


# Import Network Security Dataset

# In[2]:


dataset = pd.read_csv("dataset/kddcup.data_10_percent_corrected", header = None)


# In[3]:


dataset.head()


# In[4]:


dataset.shape


# In[5]:


dataset.describe()


# In[6]:


#Missing values in the data
column_null = [col for col in dataset.columns if dataset[col].isnull().any()]
dataset[column_null].isnull().sum()

#There are no missing data in the dataset


# In[7]:


#Create column names
feature_names = pd.read_csv("KDD_Feature_Names.txt", header = None)

name_list = []
length = range(len(feature_names))

for index in length:
    name = feature_names.iloc[index]
    name = str(name).split(": ")[0]
    name = name.split("    ")[1]
    name_list.append(name)

name_list.append("attack type")

name_list

dataset.columns = name_list


# In[8]:


#New Column Names
dataset.columns


# In[9]:


#The frequency of normal and attacks
sns.catplot(data = dataset, y = "attack type", kind = "count")
plt.ylabel("Attack Types")
plt.show()


# In[10]:


dataset["attack type"].value_counts()


# In[11]:


#Place all the classes that have less counts than normal into one class "Other"
classes_high_count = ["smurf.", "neptune.", "normal."]
other_classes = list(set(dataset["attack type"]) - set(classes_high_count))
dataset["attack type"] = dataset["attack type"].replace(other_classes, "other")


# In[12]:


#The frequency of normal and attacks
sns.catplot(data = dataset, y = "attack type", kind = "count")
plt.ylabel("Attack Types")
plt.show()


# In[13]:


#Pie Chart
x = dataset["attack type"].value_counts()
attack_type_labels = dataset["attack type"].value_counts().index
attack_type_explode = [0, 0, 0.2, 0,]

plt.pie(x, labels = attack_type_labels, explode = attack_type_explode, shadow = True)
plt.show()


# In[14]:


dataset["attack type"].value_counts()


# In[15]:


#Correlation Heatmap of the numerical features
new_dataset = dataset.copy()
new_dataset = new_dataset.drop(["num_outbound_cmds", "is_host_login"], axis = 1)

corr_matrix = new_dataset.corr()
sns.heatmap(corr_matrix)
plt.plot()


# In[16]:


#Separate Features and Target variable
X = dataset[dataset.columns[0:-1]]
y = dataset["attack type"]


# In[17]:


#Get numerical feature names
numerical_variables = X.select_dtypes(exclude=['object'])

#Get categorical feature names
categorical_variables = [col for col in X.columns if X[col].dtype == "object"]
categorical_variables


# In[18]:


#The number unique values in each categorical feature
object_nunique = list(map(lambda col: X[col].nunique(), categorical_variables))
d = dict(zip(categorical_variables, object_nunique))

#Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])


# In[19]:


#Remove categorical variables that have more than 10 unique values (Cardinality) in train dataset
low_cardinality_cols = [col for col in categorical_variables if X[col].nunique() < 10] #These are the features we will keep

#Features that will be dropped from the dataset
high_cardinality_cols = list(set(categorical_variables)-set(low_cardinality_cols))
#Service feature will be dropped from the dataset


# In[20]:


#These are the features we will keep
low_cardinality_cols


# In[21]:


#Apply OneHotEncoder function
OH_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
OH_imputed_categorical_variables = pd.DataFrame(OH_encoder.fit_transform(X[low_cardinality_cols]))

#OneHotEncoder function removes index. Give it back
OH_imputed_categorical_variables.index = X.index


# In[22]:


#Combine both numerical and categorical features
X = pd.concat([numerical_variables, OH_imputed_categorical_variables], axis=1)
X.head()


# In[23]:


#Train and split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[24]:


#Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators = 100, random_state = 0)
rf_model.fit(X_train, y_train)

#Create predictions
y_pred = rf_model.predict(X_test)


# In[25]:


#Evaluate the model
accuracy = metrics.accuracy_score(y_test, y_pred) * 100
accuracy


# In[26]:


#Confusion Matrix
plotcm = metrics.plot_confusion_matrix(rf_model, X_test, y_test)
plt.show()


# In[ ]:




