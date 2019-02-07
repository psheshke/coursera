
# coding: utf-8

# In[3]:


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


# In[4]:


salary_train = pd.read_csv('salary-train.csv')

salary_train


# In[5]:


salary_test_mini = pd.read_csv('salary-test-mini.csv')

salary_test_mini


# In[6]:


salary_train['FullDescription'] = salary_train['FullDescription'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex = True)

salary_train['LocationNormalized'] = salary_train['LocationNormalized'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex = True)

salary_train['ContractTime'] = salary_train['ContractTime'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex = True)

salary_train['LocationNormalized'].fillna('nan', inplace=True)

salary_train['ContractTime'].fillna('nan', inplace=True)

salary_train


# In[7]:


vectorizer = TfidfVectorizer(min_df = 5)

X1 = vectorizer.fit_transform(salary_train['FullDescription'])

X1


# In[8]:


enc = DictVectorizer()

X2 = enc.fit_transform(salary_train[['LocationNormalized', 'ContractTime']].to_dict('records'))

X2


# In[9]:


X = hstack([X1,X2])

X


# In[10]:


Y = salary_train['SalaryNormalized']

Y


# In[11]:


clf = Ridge(alpha=1, random_state=241)

clf.fit(X, Y)


# In[12]:


Ypred = clf.predict(X)

Ypred


# In[24]:


salary_test_mini['FullDescription'] = salary_test_mini['FullDescription'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex = True)

salary_test_mini['LocationNormalized'] = salary_test_mini['LocationNormalized'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex = True)

salary_test_mini['ContractTime'] = salary_test_mini['ContractTime'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex = True)

salary_test_mini['LocationNormalized'].fillna('nan', inplace=True)

salary_test_mini['ContractTime'].fillna('nan', inplace=True)

salary_test_mini


# In[26]:


X1test = vectorizer.transform(salary_test_mini['FullDescription'])

X1test


# In[31]:


X2test = enc.transform(salary_test_mini[['LocationNormalized', 'ContractTime']].to_dict('records'))

X2test


# In[32]:


Xtest = hstack([X1test,X2test])

Xtest


# In[35]:


Ypredtest = clf.predict(Xtest)

np.round(Ypredtest,2)

