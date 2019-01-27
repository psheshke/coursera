
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


# In[72]:


data = pd.read_csv('titanic.csv', index_col = 'PassengerId')


# In[73]:


data['Sex'] = data['Sex'].replace('male', 1)
data['Sex'] = data['Sex'].replace('female', 0)


# In[76]:


data = data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]
data = data.dropna()
data


# In[77]:


X = data[['Pclass', 'Fare', 'Age', 'Sex']].values
X


# In[78]:


y = data[['Survived']].values
y


# In[79]:


clf = DecisionTreeClassifier(random_state=241)


# In[80]:


clf.fit(X, y)


# In[81]:


clf.feature_importances_


# In[82]:


['Pclass', 'Fare', 'Age', 'Sex']

