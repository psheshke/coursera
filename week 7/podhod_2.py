
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import time
import datetime
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold, cross_val_score, train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# # Подход 2: логистическая регрессия

# In[2]:


train = pandas.read_csv('features.csv', index_col='match_id')
test = pandas.read_csv('features_test.csv', index_col='match_id')


# In[2]:


train.drop(['duration', 'tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire'], axis=1, inplace=True)


# In[2]:


y_train = train['radiant_win']
del train['radiant_win']


# In[2]:


X_train = StandardScaler().fit_transform(train.fillna(0))
X_test = test.fillna(0)


# ## Подбор параметра регуляризации - C

# In[ ]:


cv_kf = KFold(y_train.size, n_folds=5, shuffle=True, random_state=42)
scores = []
C_pow_range = range(-5, 6)
C_range = [10.0 ** i for i in C_pow_range]


# In[3]:


for C in C_range:
    start_time = datetime.datetime.now()
    print('C :', str(C))
    model = LogisticRegression(C=C, random_state=42)
    model_scores = cross_val_score(model, X_train, y_train, cv=cv_kf, scoring='roc_auc', n_jobs=-1)
    print(model_scores)
    print('Time spent:', datetime.datetime.now() - start_time)
    scores.append(np.mean(model_scores))

plot.plot(C_pow_range, scores)
plot.xlabel('log(C)')
plot.ylabel('score')
plot.show()

max_score = max(scores)
max_score_index = scores.index(max_score)


# ##Результаты

# In[4]:


print('C: ', C_range[max_score_index], 'score: ', max_score)


# Написать выводы
# 

# ## Удаление категориальных признаков

# In[5]:


# Выборка для обучения
X_train = train.fillna(0)
del X_train['lobby_type']
for n in range(1, 6):
    del X_train['r{}_hero'.format(n)]
    del X_train['d{}_hero'.format(n)]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Выборка для теста
X_test = X_test.fillna(0)
del X_test['lobby_type']
for n in range(1, 6):
    del X_test['r{}_hero'.format(n)]
    del X_test['d{}_hero'.format(n)]
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

cv_kf = KFold(y_train.size, n_folds=5, shuffle=True, random_state=42)

scores = []
C_pow_range = range(-5, 6)
C_range = [10.0 ** i for i in C_pow_range]
for C in C_range:
    start_time = datetime.datetime.now()
    print('C =', str(C))
    model = LogisticRegression(C=C, random_state=42)
    model_scores = cross_val_score(model, X_train, y_train, cv=cv_kf, scoring='roc_auc', n_jobs=-1)

    print(model_scores)
    print('Time spent ', datetime.datetime.now() - start_time)
    scores.append(np.mean(model_scores))

plot.plot(C_pow_range, scores)
plot.xlabel('log(C)')
plot.ylabel('score')
plot.show()

max_score = max(scores)
max_score_index = scores.index(max_score)


# ## Результаты

# In[ ]:


print('C: ', C_range[max_score_index], 'score: ', max_score)


# Выводы
# 

# ## Добавлении "мешка слов" по героям

# In[6]:


heroes = pandas.read_csv('./data/dictionaries/heroes.csv')
print('Всего героев в игре:', len(heroes))

X_train = train.fillna(0)
X_pick = np.zeros((X_train.shape[0], len(heroes)))
for i, match_id in enumerate(X_train.index):
    for p in range(5):
        X_pick[i, X_train.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
        X_pick[i, X_train.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1

X_hero = pandas.DataFrame(X_pick, index=X_train.index)

X_test = test.fillna(0)
X_pick = np.zeros((X_test.shape[0], len(heroes)))
for i, match_id in enumerate(X_test.index):
    for p in range(5):
        X_pick[i, X_test.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
        X_pick[i, X_test.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1

X_test_hero = pandas.DataFrame(X_pick, index=X_test.index)

scaler = StandardScaler()
X_train = pandas.DataFrame(scaler.fit_transform(X_train), index=X_train.index)
X_test = pandas.DataFrame(scaler.transform(X_test), index=X_test.index)

X_train = pandas.concat([X_train, X_hero], axis=1)
X_test = pandas.concat([X_test, X_test_hero], axis=1)

cv_kf = KFold(y_train.size, n_folds=5, shuffle=True, random_state=42)

scores = []
C_pow_range = range(-5, 6)
C_range = [10.0 ** i for i in C_pow_range]
for C in C_range:
    start_time = datetime.datetime.now()
    print('C =', str(C))
    model = LogisticRegression(C=C, random_state=42)
    model_scores = cross_val_score(model, X_train, y_train, cv=cv_kf, scoring='roc_auc', n_jobs=-1)

    print(model_scores)
    print('Time spent ', datetime.datetime.now() - start_time)
    scores.append(np.mean(model_scores))

plot.plot(C_pow_range, scores)
plot.xlabel('log(C)')
plot.ylabel('score')
plot.show()

max_score = max(scores)
max_score_index = scores.index(max_score)


# ## Результаты

# In[8]:


print('C: ', C_range[max_score_index], 'score: ', max_score)


# #### Какое качество получилось у логистической регрессии над всеми исходными признаками? Как оно соотносится с качеством градиентного бустинга? Чем можно объяснить эту разницу? Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?
# * Наилучшее значение показателя AUC-ROC при C = 0.01 и равно 0.71. Это сравнимо с градиентным бустингом по 2110 деревьями, при этом логистическая регрессия работает в ~25 быстрее.
# 
# #### Как влияет на качество логистической регрессии удаление категориальных признаков (укажите новое значение метрики качества)? Чем можно объяснить это изменение?
# * Удаление категориальных признаков не изменило качество предсказания. Наилучшее значение показателя AUC-ROC не изменились. Следовательно, вес этих параметров в предыдушей модели был близок к нулю.
# 
# #### Сколько различных идентификаторов героев существует в данной игре?
# * Всего героев в игре: 112
# 
# #### Какое получилось качество при добавлении "мешка слов" по героям? Улучшилось ли оно по сравнению с предыдущим вариантом? Чем можно это объяснить?
# * Добавление "мешка слов" улучшило качество предсказания. Значение AUC-ROC улучшилось  до 0.75 при C = 0.1. Это объясняется тем, что вместо отсутствия данных о героях или случайного шума из id мы имеем осмысленную разреженную матрицу для построения предсказания.
# 
# #### Какое минимальное и максимальное значение прогноза на тестовой выборке получилось у лучшего из алгоритмов?
# * Минимальное значение показателя AUC-ROC у лучшего алгоритма равно 0.69 при C=0.00001 
# * Максимальное значение показателя AUC-ROC у лучшего алгоритма алгоритма равно 0.751 при C=0.1 
# 
