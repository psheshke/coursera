
# coding: utf-8

# In[47]:


import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import scale, StandardScaler
from sklearn.datasets import load_boston
from sklearn.linear_model import Perceptron, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score,precision_recall_curve,confusion_matrix
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.decomposition import PCA
from scipy.stats.stats import pearsonr


# In[29]:


close_prices = pd.read_csv('close_prices.csv')

close_prices


# In[30]:


close_prices.drop('date', axis=1, inplace=True)


# In[32]:


pca = PCA(n_components=10)

pca.fit(close_prices)


# In[33]:


print(pca.explained_variance_ratio_)  


# In[34]:


pca.explained_variance_ratio_


# In[69]:


s = 0

n = 0

for i in range(len(pca.explained_variance_ratio_)):
    
    s = s + pca.explained_variance_ratio_[i]
    
    if s>=0.9:
        
        n = i + 1
        
        break
        
print(s, n)


# In[44]:


Xtr = pd.DataFrame(pca.transform(close_prices))[0]

Xtr


# In[45]:


djia_index = pd.read_csv('djia_index.csv')

djia_index


# In[58]:


np.round(np.corrcoef(djia_index['^DJI'],Xtr),2)[0][1]


# In[52]:


pca.components_[0]


# In[62]:


s = 0

k = 0

for i in range(len(pca.components_[0])):
    
    if abs(pca.components_[0][i]) > s:
        
        s = abs(pca.components_[0][i])
        
        k = i
        
print(s, k)


# In[63]:


close_prices.columns[k]

