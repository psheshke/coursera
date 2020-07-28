
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import scale, StandardScaler
from sklearn.datasets import load_boston
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV


# In[3]:


newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )


# In[19]:


Y = newsgroups.target

Y


# In[8]:


newsgroups.target_names


# In[17]:


texts = newsgroups.data


# In[20]:


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)


# In[26]:


grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, Y)


# In[27]:


gs.grid_scores_


# In[35]:


clf = SVC(random_state = 241, C = 1.0, kernel = 'linear')

clf.fit(X, Y)


# In[49]:


coef = clf.coef_
coef


# In[51]:


q = pd.DataFrame(coef.toarray()).transpose()

q


# In[52]:


top10=abs(q).sort_values([0], ascending=False).head(10)
top10


# In[57]:


indices=[]

indices=top10.index

words=[]

for i in indices:

    feature_mapping=vectorizer.get_feature_names()

    words.append(feature_mapping[i])

print(" ".join(sorted(words)))


# In[61]:


file_answer = open("week 3 txtanalysis 1.txt", "w")
file_answer.write("atheism atheists bible god keith moon religion sci sky space")
file_answer.close()

