
# coding: utf-8

# In[16]:

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston


# In[11]:

data = load_boston()['data']

pd.DataFrame(data)


# In[22]:

target = load_boston()['target']

Y = target

pd.DataFrame(target)


# In[14]:

X = scale(data)

pd.DataFrame(X)


# In[15]:

kf = KFold(n_splits=5, random_state=42, shuffle=True)


# In[47]:

ps = np.linspace(1.0, 10.0, num=200)

pd.DataFrame(ps)


# In[45]:

neigh = KNeighborsRegressor(n_neighbors=5, weights = 'distance', p = ps[0])

neigh.fit(X, Y)

ac = np.mean(cross_val_score(estimator = neigh, cv = kf, X = X, y = Y,scoring = 'neg_mean_squared_error'))

k = 1

for i in range(1,len(ps)):
    
    neigh = KNeighborsRegressor(n_neighbors=5, weights = 'distance', p = ps[i])

    neigh.fit(X, Y)

    ac2 = np.mean(cross_val_score(estimator = neigh, cv = kf, X = X, y = Y,scoring = 'neg_mean_squared_error'))
    
    if ac2 > ac:
        
        ac = ac2
        
        k = i    


# In[46]:

ac, k

