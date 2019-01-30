
# coding: utf-8

# In[110]:

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale


# In[11]:

data = pd.read_csv('wine.data', header=None)


# In[12]:

data


# In[18]:

Y = data[0]
Y


# In[31]:

X = data.loc[:,1:]
X


# In[103]:

kf = KFold(n_splits=5, random_state=42, shuffle=True)


# In[104]:

neigh = KNeighborsClassifier(n_neighbors=1)


# In[105]:

neigh.fit(X, Y)


# In[106]:

cross_val_score(estimator = neigh, cv = kf, X = X, y = Y,scoring = 'accuracy')


# In[107]:

np.mean(cross_val_score(estimator = neigh, cv = kf, X = X, y = Y,scoring = 'accuracy'))


# In[116]:

neigh = KNeighborsClassifier(n_neighbors=1)

neigh.fit(X, Y)

ac = np.mean(cross_val_score(estimator = neigh, cv = kf, X = X, y = Y,scoring = 'accuracy'))

k = 1

for i in range(1,50):
    
    neigh = KNeighborsClassifier(n_neighbors=i+1)
    
    neigh.fit(X, Y)
    
    ac2 = np.mean(cross_val_score(estimator = neigh, cv = kf, X = X, y = Y,scoring = 'accuracy'))
    
    if ac2 > ac:
        
        ac = ac2
        
        k = i


# In[117]:

ac, k


# In[115]:

Xs = scale(X = X)

pd. DataFrame(Xs)


# In[118]:

neigh = KNeighborsClassifier(n_neighbors=1)

neigh.fit(Xs, Y)

ac = np.mean(cross_val_score(estimator = neigh, cv = kf, X = Xs, y = Y,scoring = 'accuracy'))

k = 1

for i in range(1,50):
    
    neigh = KNeighborsClassifier(n_neighbors=i+1)
    
    neigh.fit(Xs, Y)
    
    ac2 = np.mean(cross_val_score(estimator = neigh, cv = kf, X = Xs, y = Y,scoring = 'accuracy'))
    
    if ac2 > ac:
        
        ac = ac2
        
        k = i


# In[119]:

ac, k


# In[ ]:



