
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import scale, StandardScaler
from sklearn.datasets import load_boston
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


data = pd.read_csv('data-logistic.csv', header = None)

data


# In[3]:


Y_ = data.values[:,:1].T[0]

Y_


# In[4]:


X_ = data.values[:,1:]

X_


# In[5]:


def sigm(x):
    
    return 1.0/(1+math.exp(-x))


# In[6]:


def dist(a,b):
    
    return np.sqrt(np.square(a[0]-b[0]) + np.square(a[1]-b[1]))


# In[13]:


def logregr(X,Y,k,w1,w2,C,eps,max_iter):
    
    for i in range(max_iter):
        
        w1n = w1 + k * np.mean(Y * X[:,0] * (1-(1.0 / (1+np.exp(-Y*(w1 * X[:,0] + w2 * X[:,1])))))) - k * C * w1
        
        w2n = w2 + k * np.mean(Y * X[:,1] * (1-(1.0 / (1+np.exp(-Y*(w1 * X[:,0] + w2 * X[:,1])))))) - k * C * w2
        
        if dist((w1n, w2n), (w1,w2)) < eps:
            break
        w1, w2 = w1n, w2n
        
    predicts = []
    
    print(w1,w2)
    
    for i in range(len(X)):
        
        t1 = w1*X[i, 0] + w2 * X[i,1]
        
        s = sigm(t1)
        
        predicts.append(s)
        
    return predicts


# In[14]:


p0 = logregr(X_, Y_, 0.1, 0.0, 0.0, 0, 0.00001, 10000)

p0


# In[15]:


p1 = logregr(X_, Y_, 0.1, 0.0, 0.0, 10, 0.00001, 10000)

p1


# In[18]:


round(roc_auc_score(Y_, p0), 3)


# In[19]:


round(roc_auc_score(Y_, p1), 3)

