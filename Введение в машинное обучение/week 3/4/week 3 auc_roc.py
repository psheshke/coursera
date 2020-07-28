
# coding: utf-8

# In[55]:


import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import scale, StandardScaler
from sklearn.datasets import load_boston
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score,precision_recall_curve,confusion_matrix
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer


# In[3]:


classification = pd.read_csv('classification.csv')

classification


# In[84]:


TP = len(classification[(classification['true'] == 1) & (classification['pred'] == 1)].true)

TP


# In[85]:


FP = len(classification[(classification['true'] == 0) & (classification['pred'] == 1)].true)

FP


# In[86]:


FN = len(classification[(classification['true'] == 1) & (classification['pred'] == 0)].true)

FN


# In[87]:


TN = len(classification[(classification['true'] == 0) & (classification['pred'] == 0)].true)

TN


# In[90]:


TP, FP, FN, TN


# In[61]:


confusion_matrix(classification['true'], classification['pred'], labels=[1,0])


# In[23]:


round(accuracy_score(classification.true, classification.pred), 2)


# In[20]:


round(precision_score(classification.true, classification.pred), 2)


# In[21]:


round(recall_score(classification.true, classification.pred), 2)


# In[22]:


round(f1_score(classification.true, classification.pred), 2)


# In[24]:


scores = pd.read_csv('scores.csv')

scores


# In[40]:


roc_auc = 0
n = 0
for i in range(len(scores.columns[1:])):
    rc = roc_auc_score(scores.true, scores[scores.columns[1:][i]])
    
    if rc > roc_auc:
        roc_auc = rc
        n = i


# In[83]:


scores.columns[1:][n], roc_auc


# In[80]:


maxpr = 0
n = 0
for i in range(len(scores.columns[1:])):
    
    precision, recall, thresholds = precision_recall_curve(scores.true, scores[scores.columns[1:][i]])
    
    d = {'precision': precision,'recall': recall}
    
    pr = max(df[df['recall'] >= 0.7].precision)
    
    if rc > maxpr:
        maxpr = pr
        n = i


# In[82]:


scores.columns[1:][n], maxpr

