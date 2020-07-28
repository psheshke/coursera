
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np


# In[3]:

data = pd.read_csv('titanic.csv', index_col = 'PassengerId')


# In[4]:

data


# In[6]:

data['Sex'].value_counts()


# In[13]:

round(data['Survived'].value_counts(1)*100,2)


# In[14]:

round(data['Pclass'].value_counts(1)*100,2)


# In[25]:

data['Age'].mean()


# In[18]:

data['Age'].median()


# In[27]:

corrmat = data.corr(method='pearson')
corrmat


# In[35]:

data[data['Sex'] == 'female']['Name']


# In[133]:

s = pd.Series()
qq = data[data['Sex'] == 'female']['Name'].str.split('Miss. ')
for i in range(len(qq)):
    if len(qq[qq.index[i]]) == 2:
        s = pd.concat([s, pd.Series([qq[qq.index[i]][1]])])
        
dd = data[data['Sex'] == 'female']['Name'].str.split('(')
for i in range(len(dd)):
    if len(dd[dd.index[i]]) ==2:
        s = pd.concat([s, pd.Series([dd[dd.index[i]][1].split()[0]])])


# In[135]:

s.value_counts()

