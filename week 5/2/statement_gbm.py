
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import scale, StandardScaler
from sklearn.datasets import load_boston
from sklearn.linear_model import Perceptron, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score,     precision_recall_curve,confusion_matrix, r2_score, log_loss

from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.decomposition import PCA
from scipy.stats.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,     GradientBoostingRegressor
from sklearn.cross_validation import train_test_split


# In[7]:


data = pd.read_csv('gbm-data.csv')

data


# In[16]:


Y = data[data.columns[0]]

Y


# In[17]:


X = data[[x for x in data.columns if x != data.columns[0]]]

X


# In[24]:


X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y.values, test_size = 0.8, random_state = 241)


# In[32]:


gbc = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate = 0.1)


# In[33]:


gbc.fit(X_train, Y_train)


# In[128]:


original_params = {'n_estimators': 250, 'verbose': False, 'random_state': 241}

learning_rate = [1, 0.5, 0.3, 0.2, 0.1]

plt.figure()

for label, color, label2, color2, setting in [ ('test' , 'blue', 'train', 'Crimson',{'learning_rate': 1.0}),
                             ('test' , 'green', 'train', 'Crimson',{'learning_rate': 0.5}),
                             ('test' , 'red', 'train', 'Crimson',{'learning_rate': 0.3}),
                             ('test' , 'orange', 'train', 'Crimson',{'learning_rate': 0.2}),
                             ('test' , 'gray', 'train', 'Crimson',{'learning_rate': 0.1}),]:
    
    params = dict(original_params)
    params.update(setting)
    
    gbc = GradientBoostingClassifier(**params)
    
    gbc.fit(X_train, Y_train)
    
    test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)
    
    train_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)
    
    for i, y_pred in enumerate(gbc.staged_decision_function(X_test)):
        
        u = 1 / (1 + math.e**(-y_pred))
        
        test_deviance[i] = log_loss(Y_test, u)
        
    for i, y_pred in enumerate(gbc.staged_decision_function(X_train)):
        
        u = 1 / (1 + math.e**(-y_pred))
        
        train_deviance[i] = log_loss(Y_train, u)    
        
    minerr = test_deviance[0]

    n = 1

    for i in range(len(test_deviance)):

        if test_deviance[i] < minerr:

            minerr = test_deviance[i]

            n = i+1
            
    print(round(minerr,3), n)
        
    plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
            '-', color=color, label=label)
    
    plt.plot((np.arange(train_deviance.shape[0]) + 1)[::5], train_deviance[::5],
            '-', color=color2, label=label2)
    
    plt.legend(loc='upper left')
    plt.xlabel('Boosting Iterations')
    plt.title('learning-rate = ' + str(params['learning_rate']))
    

    plt.show()


# In[109]:


learning_rate = [1, 0.5, 0.3, 0.2, 0.1]

for label, color, setting in [ ('learning_rate=1' , 'blue',{'learning_rate': 0.2})]:
    
    params = dict(original_params)
    params.update(setting)
    
    gbc = GradientBoostingClassifier(**params)
    
    gbc.fit(X_train, Y_train)
    
    test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)
    
    for i, y_pred in enumerate(gbc.staged_decision_function(X_test)):
        
        u = 1 / (1 + math.e**(-y_pred))
        
        test_deviance[i] = log_loss(Y_test, u)


# In[112]:


minerr = test_deviance[0]

n = 1

for i in range(len(test_deviance)):
    
    if test_deviance[i] < minerr:
        
        minerr = test_deviance[i]
        
        n = i+1
        
print(round(minerr,2), n)


# In[115]:


gbc = GradientBoostingClassifier(random_state=241, n_estimators=37)

gbc.fit(X_train, Y_train)


# In[116]:


gbc.predict_proba(X_test)


# In[118]:


round(log_loss(Y_test, gbc.predict_proba(X_test)),2)

