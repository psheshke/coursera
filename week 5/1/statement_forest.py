
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import scale, StandardScaler
from sklearn.datasets import load_boston
from sklearn.linear_model import Perceptron, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score,     precision_recall_curve,confusion_matrix, r2_score

from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.decomposition import PCA
from scipy.stats.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor 


# In[17]:


data = pd.read_csv('abalone.csv')

data


# In[18]:


data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

data


# In[38]:


Y = data[data.columns[len(data.columns)-1]]

Y


# In[39]:


X = data[[x for x in data.columns if x != data.columns[len(data.columns)-1]]]

X


# In[40]:


kf = KFold(n_splits=5, random_state=1, shuffle=True)


# In[42]:


RF = RandomForestRegressor(random_state=1, n_estimators=1)

RF.fit(X,Y)


# In[77]:


n = 0

r2cv = 0.52

for i in range(50):
    
    RF = RandomForestRegressor(random_state=1, n_estimators=i+1)
    
    r2s = []
    
    for train_index, test_index in kf.split(data):
        
        RF.fit(X.loc[train_index],Y[train_index])
        
        r2s.append( r2_score(Y[test_index],RF.predict(X.loc[test_index])) )
        
    print("i = ", i, " r2 = ", np.mean(r2s))
    
    if np.mean(r2s) > r2cv:
        
        n = i+1
        
        break
    


# In[78]:


n

