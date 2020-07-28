
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import scale, StandardScaler
from sklearn.datasets import load_boston
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# In[4]:


data = pd.read_csv('svm-data.csv', header = None)

data


# In[8]:


Y = data[0]

Y


# In[9]:


X = data.loc[:,1:]

X


# In[10]:


clf = SVC(random_state = 241, C = 100000, kernel = 'linear')

clf.fit(X, Y)


# In[11]:


clf.support_


# In[25]:


str(clf.support_).split('[')[1].split(']')[0]


# In[23]:


file_answer = open("week 3 svm 1.txt", "w")
file_answer.write(repr(str(clf.support_).split('[')[1].split(']')[0]))
file_answer.close()

