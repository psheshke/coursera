
# coding: utf-8

# # Подход 1: градиентный бустинг "в лоб"

# ## Что указать в отчете
# ### В отчете по данному этапу вы должны ответить на следующие вопросы:
# 
# ### Какие признаки имеют пропуски среди своих значений? Что могут означать пропуски в этих признаках (ответьте на этот вопрос для двух любых признаков)?
# #### 'first_blood_time', 'first_blood_team', 'first_blood_player1', 'first_blood_player2', 'radiant_bottle_time', 'radiant_courier_time', 'radiant_flying_courier_time', 'radiant_first_ward_time', 'dire_bottle_time','dire_courier_time','dire_flying_courier_time','dire_first_ward_time'.
# Признак first_blood_time - игровое время первой крови. Первая кровь не была пролита в течение первых 5 минут матча. Возможно обе команды были настроены на пассивную игру и попытки нарастить преимущество в лейте.
# Признак radiant_first_ward_time - время установки командой первого "наблюдателя", т.е. предмета, который позволяет видеть часть игрового поля. Ни один из игроков команды dire не покупал "ward" в течение первых 5 минут матча. Игра без саппортов на низком рейтинге...
# 
# ### Как называется столбец, содержащий целевую переменную?
# radiant_win
# 
# ### Как долго проводилась кросс-валидация для градиентного бустинга с 30 деревьями? Инструкцию по измерению времени можно найти ниже по тексту. Какое качество при этом получилось? Напомним, что в данном задании мы используем метрику качества AUC-ROC.
# Затраченное время:  0:01:23, AUC = 0.69  
# 
# ### Имеет ли смысл использовать больше 30 деревьев в градиентном бустинге? Что бы вы предложили делать, чтобы ускорить его обучение при увеличении количества деревьев?
# С ростом числа деревьев, при сохранении остальных параметров увеличивается общее время расчета. Среднее значение AUC по кросс-валидации увеличивается при росте числа деревьев, что говорит нам о том, что использование большего числа деревьев увеличивает качество модели на обучающей выборке. Однако, с ростом качества на обучении увеличивается риск переобучения
# 

# ## Загрузка обучающей и тестовой выборок

# In[229]:


import pandas as pd

features = pd.read_csv('./features.csv', index_col='match_id')

features.head()


# In[230]:


features_test = pd.read_csv('./features_test.csv', index_col='match_id')

features_test.head()


# ## Найдем признаки, отсутствующие в тестовой выборке.

# In[231]:


itogs = [x for x in features.columns if x not in features_test.columns]

itogs


# ## Целевой признак - 'radiant_win'

# In[232]:


y = features['radiant_win'].as_matrix()

y


# ## Оставим только те признаки, которые есть в тестовой выборке

# In[233]:


features = features[[x for x in features.columns if x in features_test.columns]]

features.head()


# ## Найдем все признаки в которых присутствуют пропуски

# In[234]:


[k for k,v in dict(features.count() / features.shape[0]).items() if v < 1]


# ## Возможная причина пропуска в данных:
# 
# ### Признак first_blood_time - игровое время первой крови.
# 
# #### Первая кровь не была пролита в течение первых 5 минут матча. Возможно обе команды были настроены на пассивную игру и попытки нарастить преимущество в лейте.
# 
# ### Признак radiant_first_ward_time - время установки командой первого "наблюдателя", т.е. предмета, который позволяет видеть часть игрового поля
# 
# #### Ни один из игроков команды dire не покупал "ward" в течение первых 5 минут матча. Игра без саппортов на низком рейтинге...

# ## Заменяем пропуски в обучающей выборке на 0 и проверяем, что не осталось ни одного признака с пропусками.

# In[235]:


features = features.fillna(0)

[k for k,v in dict(features.count() / features.shape[0]).items() if v < 1]


# ## Обучение модели градиентного бустинга

# In[243]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)

X = features.as_matrix()

for n in [10,20,30,40,50,60, 70, 80, 90, 100]:
    
    start_time = datetime.datetime.now()
    
    gbc = GradientBoostingClassifier(n_estimators=n, verbose=False, random_state=42)
    
    model_scores = cross_val_score(gbc, X, y, cv=kf, scoring='roc_auc')
        
    print("Деревьев: ", n, " Среднее значение AUC: ", round(np.mean(model_scores),4), " Затраченное время: ", datetime.datetime.now() - start_time)


# ## С ростом числа деревьев, при сохранении остальных параметров увеличивается общее время расчета. Среднее значение AUC по кросс-валидации увеличивается при росте числа деревьев, что говорит нам о том, что использование большего числа деревьев увеличивает качество модели на обучающей выборке. Однако, с ростом качества на обучении увеличивается риск переобучения

# # Подход 2: логистическая регрессия

# ## Что указать в отчете
# ### В отчете по данному этапу вы должны ответить на следующие вопросы:
# 
# ### Какое качество получилось у логистической регрессии над всеми исходными признаками? Как оно соотносится с качеством градиентного бустинга? Чем вы можете объяснить эту разницу? Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?
# Лучшее значение AUC получилось 0.7166 при С = 0.01. На модели бустинга подобное значение качества не было достигнуто при количестве деревьев меньше 100. Логистическая регрессия значительно быстрее бустинга.
# 
# ### Как влияет на качество логистической регрессии удаление категориальных признаков (укажите новое значение метрики качества)? Чем вы можете объяснить это изменение?
# Их удаление практически не повлияло на итоговые результаты, что говорит о низком влиянии данных показателей на модель. Лучшее значение AUC получилось 0.7166 при С = 0.01.
# 
# ### Сколько различных идентификаторов героев существует в данной игре?
# 112
# 
# ### Какое получилось качество при добавлении "мешка слов" по героям? Улучшилось ли оно по сравнению с предыдущим вариантом? Чем вы можете это объяснить?
# Применение "мешка слов" привело к улучшению результатов модели на кросс-валидации. Лучшее значение AUC получилось 0.7519   при С = 0.1, что может быть объяснено включением в модель информации о составе каждоый из команд, что определенно должно повлиять на результаты матчей
# 
# ### Какое минимальное и максимальное значение прогноза на тестовой выборке получилось у лучшего из алгоритмов?
# минимум = 0.0086, максимум = 0.9965
# 

# In[241]:


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

scaler = StandardScaler()

X = scaler.fit_transform(features.as_matrix())

for C in [10.0 ** i for i in range(-5,5)]:
    
    start_time = datetime.datetime.now()
    
    lr = LogisticRegression(C=C, random_state=42, penalty='l2')
    
    model_scores = cross_val_score(lr, X, y, cv=kf, scoring='roc_auc')
        
    print("C: ", C, " Среднее значение AUC: ", round(np.mean(model_scores),4), " Затраченное время: ", datetime.datetime.now() - start_time)


# ## Удалим категориальные переменные

# In[246]:


X = features[[x for x in features.columns if x not in ['lobby_type', 'r1_hero', 'r2_hero', 
                                                       'r3_hero','r4_hero','r5_hero','d1_hero', 
                                                       'd2_hero', 'd3_hero','d4_hero','d5_hero']]]

X = scaler.fit_transform(X.as_matrix())

for C in [10.0 ** i for i in range(-5,5)]:
    
    start_time = datetime.datetime.now()
    
    lr = LogisticRegression(C=C, random_state=42, penalty='l2')
    
    model_scores = cross_val_score(lr, X, y, cv=kf, scoring='roc_auc')
        
    print("C: ", C, " Среднее значение AUC: ", round(np.mean(model_scores),4), " Затраченное время: ", datetime.datetime.now() - start_time)


# ## Добавим "мешок слов"

# In[252]:


heroes = pd.read_csv('./data/dictionaries/heroes.csv')
print('Всего героев в игре:', len(heroes))


# In[254]:


N = len(heroes)

X_pick = np.zeros((features.shape[0], N))

for i, match_id in enumerate(features.index):
    
    for p in range(5):
        
        X_pick[i, features.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, features.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
        
X_pick


# In[272]:


X = pd.concat([pd.DataFrame(X, index=features.index), pd.DataFrame(X_pick, index=features.index)], axis=1).as_matrix()

X


# In[276]:


for C in [10.0 ** i for i in range(-5,5)]:
    
    start_time = datetime.datetime.now()
    
    lr = LogisticRegression(C=C, random_state=42, penalty='l2')
    
    model_scores = cross_val_score(lr, X, y, cv=kf, scoring='roc_auc')
        
    print("C: ", C, " Среднее значение AUC: ", round(np.mean(model_scores),4), " Затраченное время: ", datetime.datetime.now() - start_time)


# ## Предскажем вероятность победы radiant на тестовой выборке по лучшей модели

# In[277]:


lr = LogisticRegression(C=0.1, random_state=42, penalty='l2')

lr.fit(X,y)


# In[303]:


features_test = features_test.fillna(0)

X_test = features_test[[x for x in features_test.columns if x not in ['lobby_type', 'r1_hero', 'r2_hero', 
                                                       'r3_hero','r4_hero','r5_hero','d1_hero', 
                                                       'd2_hero', 'd3_hero','d4_hero','d5_hero']]]

X_test = scaler.fit_transform(X_test.as_matrix())

X_pick_test = np.zeros((features_test.shape[0], N))

for i, match_id in enumerate(features_test.index):
    
    for p in range(5):
        
        X_pick_test[i, features_test.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick_test[i, features_test.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
        
X_test = pd.concat([pd.DataFrame(X_test, index=features_test.index), pd.DataFrame(X_pick_test, index=features_test.index)], axis=1).as_matrix()

X_test


# In[314]:


pred = lr.predict_proba(X_test)[:,1]

pd.concat([pd.DataFrame(X_test, index=features_test.index), pd.DataFrame({'radiant_win ':pred}, index=features_test.index)], axis=1)


# In[318]:


round(min(pred),4), round(max(pred),4)

