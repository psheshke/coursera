
# coding: utf-8

# In[18]:

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import scale, StandardScaler
from sklearn.datasets import load_boston
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


# In[36]:

train = pd.read_csv('perceptron-train.csv', header=None)

train


# In[37]:

test = pd.read_csv('perceptron-test.csv', header=None)

test


# In[38]:

X_train = train.loc[:,1:]

X_train


# In[39]:

y_train = train[0]

y_train


# In[40]:

X_test = test.loc[:,1:]

X_test


# In[41]:

y_test = test[0]

y_test


# In[42]:

clf = Perceptron(random_state=241)

clf.fit(X_train, y_train)

predictions = clf.predict(X_test)

predictions


# In[43]:

a1 = accuracy_score(y_test, predictions)

a1


# In[44]:

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# In[45]:

#clf = Perceptron(random_state=241)

clf.fit(X_train_scaled, y_train)

predictions_scaled = clf.predict(X_test_scaled)

predictions_scaled


# In[46]:

a2 = accuracy_score(y_test, predictions_scaled)

a2


# In[47]:

a2 - a1

