
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import time
import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# In[2]:

features = pd.read_csv('features.csv', index_col='match_id')

features.shape


# In[3]:

features.head()


# # Подход 1: градиентный бустинг "в лоб"

# ## Какие признаки имеют пропуски среди своих значений (приведите полный список имен этих признаков)? Что могут означать пропуски в этих признаках (ответьте на этот вопрос для двух любых признаков)?

# In[4]:

for col in features.columns[pd.DataFrame(features.count())[0]<features.shape[0]].tolist() :
    print('Признак: {} должно быть {:d} значений, по факту {:d}, не хватает {:d}'.format(col,
                                            features.shape[0],
                                            features[col].count(),
                                            features.shape[0]-features[col].count()))


# - `first_blood*` Судя по описанию задачи, существуют такие игровые моменты, что first_blood в течении первых 5 минут не случилось и признаки приняли пропущенное значение 
# - `*bottle_time` команды (radiant|dire) не приобретали объект bottle
# - `*courier_time` команды (radiant|dire) не приобретали объект courier
# - `*flying_courier_time` команды (radiant|dire) не приобретали объект flying_courier
# - `*first_ward_time` команды (radiant|dire) не приобретали объект first_ward

# In[5]:

features.fillna(0,inplace=True)
y = features['radiant_win'].to_frame()
features.drop(['duration', 'radiant_win', 'tower_status_radiant', 
               'tower_status_dire', 'barracks_status_radiant', 
               'barracks_status_dire'], 
              axis=1, inplace=True)
X = features.copy()


# Проверяем, что отсутствуют пропуски

# In[6]:

print(X.columns[X.isna().any()].tolist())
print(y.columns[y.isna().any()].tolist())


# Столбец содержащий целевую переменную `radiant_win`

# In[7]:

trees_cnt = [10, 20, 30, 100, 200]


# In[8]:

kf = KFold(n_splits=5, shuffle=True, random_state=241)
scores = []
for n_est in trees_cnt:
    print('n_estimators={}'.format(n_est))
    model = GradientBoostingClassifier(n_estimators=n_est, random_state=241)
    start_time = datetime.datetime.now()
    score = cross_val_score(model, X, y, cv=kf, scoring='roc_auc', n_jobs=-1)
    scores.append(np.mean(score))
    print('scores={} \ntime={}\n'.format(scores, (datetime.datetime.now() - start_time)))


# In[46]:

model = GradientBoostingClassifier(n_estimators=30, random_state=241)
model.fit(X, y)


# Значимые признаки

# In[71]:

cols = pd.DataFrame(list(zip(X.columns, model.feature_importances_)),columns=['f_name','f_imp'])
a = cols[cols['f_imp']>0]
a.sort_values(by=['f_imp'], ascending=False)


# Бустинг с 30 деревьями завершился за ~30 секунд, показатель AUC_ROC при этом

# In[11]:

round(dict(zip(trees_cnt, scores))[30], 2)


# Скорость бустинг на большем кол-ве деревьев падает (см. вывод)
# Для увеличения производительности можно сократить обучающую выборку, уменьшить кол-во уровней деревьев (хотя и так деревья не большой глубины)

# # Подход 2: логистическая регрессия

# In[12]:

kf = KFold(n_splits=5, shuffle=True, random_state=241)
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)


# In[13]:

kfl = KFold(n_splits=5, shuffle=True, random_state=17)
scores = []
for C_idx in [10.0 ** i for i in range(-5,3)]:
    print('C={}'.format(C_idx))
    start_time = datetime.datetime.now()
    model = LogisticRegression(C=C_idx, random_state=17, n_jobs=-1)
    score = cross_val_score(model, X_sc, y, cv=kfl, scoring='roc_auc', n_jobs=-1)
    scores.append([C_idx,np.mean(score)])
    print('scores={} \ntime={}\n'.format(scores, (datetime.datetime.now() - start_time)))


# In[14]:

scores


# In[15]:

print('Лучшее С={:.2f} AUC_ROC={:.4f}'.format(pd.DataFrame(scores).sort_values(by=[1],ascending=False).iloc[0,:][0],
               pd.DataFrame(scores).sort_values(by=[1],ascending=False).iloc[0,:][1]))


# Это лучше бустинга с 200 деревьями и быстрее в несколько раз

# In[16]:

X_sc = X.drop([col for col in X.columns if 'hero' in col],axis=1)
X_sc.drop('lobby_type', axis=1, inplace=True)


# In[17]:

scaler = StandardScaler()
X_sc = scaler.fit_transform(X_sc)
kfl = KFold(n_splits=5, shuffle=True, random_state=17)
scores = []
for C_idx in [10.0 ** i for i in range(-5,3)]:
    print('C={}'.format(C_idx))
    start_time = datetime.datetime.now()
    model = LogisticRegression(C=C_idx, random_state=17, n_jobs=-1)
    score = cross_val_score(model, X_sc, y, cv=kfl, scoring='roc_auc', n_jobs=-1)
    scores.append([C_idx,np.mean(score)])
    print('scores={} \ntime={}\n'.format(scores, (datetime.datetime.now() - start_time)))


# In[18]:

scores


# In[19]:

print('Лучшее С={:.2f} AUC_ROC={:.4f}'.format(pd.DataFrame(scores).sort_values(by=[1],ascending=False).iloc[0,:][0],
               pd.DataFrame(scores).sort_values(by=[1],ascending=False).iloc[0,:][1]))


# Удаление категориальных признаков ничтожно мало повлияло на качество предсказания. 
# Наилучшее значение показателя AUC-ROC так же достигается при `C = 0.01 и равно 0.7164`. 
# В предыдущем эксперименте `Лучшее С=0.01 AUC_ROC=0.7163`
#  В предыдущей модели признаки `*hero` никак не влияли на результат, возможно, модель посчитала признаки не значимыми (см вывод "Значимые признаки")

# In[20]:

df_unique = pd.DataFrame(np.unique(X[[col for col in X.columns if 'hero' in col]].values))
h_unq_cnt = df_unique.count()[0]
h = pd.read_csv('./dictionaries/heroes.csv')
h_all_cnt = h.shape[0]


# In[21]:

print('Различных (уникальных) геров в данных = {}'.format(h_unq_cnt))


# In[34]:

print('Всего геров = {}'.format(h_all_cnt))


# In[22]:

X_pick = np.zeros((X.shape[0], h_all_cnt))

for i, match_id in enumerate(X.index):
    for p in range(1,5):
        X_pick[i, X.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, X.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1


# In[97]:

XX_pick = pd.concat([X.reset_index(drop=True), pd.DataFrame(X_pick).reset_index(drop=True)], axis=1, ignore_index=True)
scaler = StandardScaler()
XX_pick_sc = scaler.fit_transform(XX_pick)


# In[24]:

kfl = KFold(n_splits=5, shuffle=True, random_state=241)
scores = []
for C_idx in [10.0 ** i for i in range(-5,3)]:
    print('C={}'.format(C_idx))
    start_time = datetime.datetime.now()
    model = LogisticRegression(C=C_idx, random_state=241, n_jobs=-1)
    score = cross_val_score(model, XX_pick_sc, y, cv=kfl, scoring='roc_auc', n_jobs=-1)
    scores.append([C_idx,np.mean(score)])
    print('scores={} \ntime={}\n'.format(scores, (datetime.datetime.now() - start_time)))


# In[25]:

scores


# In[26]:

print('Лучшее С={:.2f} AUC_ROC={:.4f}'.format(pd.DataFrame(scores).sort_values(by=[1],ascending=False).iloc[0,:][0],
               pd.DataFrame(scores).sort_values(by=[1],ascending=False).iloc[0,:][1]))


# Добавления "мешка слов" улучшает качество модели. 
# Наилучшее значение показателя AUC-ROC достигается при C = 0.1 и равно 0.7432. 
# Это объясняется тем, что добавились признаки о героях.

# In[27]:

df_test = pd.read_csv('./features_test.csv', index_col='match_id')


# In[90]:

df_test.fillna(0,inplace=True)
X_test_pick = np.zeros((X_test.shape[0], h_all_cnt))

for i, match_id in enumerate(X_test.index):
    for p in range(1,5):
        X_test_pick[i, X_test.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_test_pick[i, X_test.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
print(X_test.shape,X_test_pick.shape)


# In[99]:

XX_test_pick = pd.concat([X_test.reset_index(drop=True), pd.DataFrame(X_test_pick).reset_index(drop=True)], axis=1, ignore_index=True)


# In[103]:

XX_test_pick_sc = scaler.transform(XX_test_pick)
model = LogisticRegression(C=0.01, random_state=241, n_jobs=-1)
model.fit(XX_pick_sc, y)
y_test = model.predict_proba(XX_test_pick_sc)[:, 1]


# In[104]:

res = pd.DataFrame({'radiant_win': y_test}, index=X_test.index)
res.head()


# In[105]:

print('Минимальное значение прогноза = {:.4f}\nМаксимальное значение прогноза = {:.4f}'.format(res.min()[0], res.max()[0]))


# In[106]:

res.to_csv('kaggle_dota2.csv')


# In[107]:

get_ipython().system('head -10 ./kaggle_dota2.csv')

