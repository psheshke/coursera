
# coding: utf-8

# # 1: Градиентный бустинг

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from sklearn import ensemble
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler


# 1.1) Считайте таблицу с признаками из файла features.csv с помощью кода, приведенного выше. Удалите признаки, связанные с итогами матча (они помечены в описании данных как отсутствующие в тестовой выборке).

# In[2]:


df = pd.read_csv('C:/Users/alexl/Downloads/data/features.csv', index_col='match_id')
num_rows,num_columns = df.shape
df.index = range(num_rows)
X=df.drop(['duration','radiant_win','tower_status_radiant','tower_status_dire','barracks_status_radiant','barracks_status_dire'],axis=1)
X.head()


# 1.2) Проверьте выборку на наличие пропусков с помощью функции count(), которая для каждого столбца показывает число заполненных значений. Много ли пропусков в данных? Запишите названия признаков, имеющих пропуски, и попробуйте для любых двух из них дать обоснование, почему их значения могут быть пропущены.

# In[3]:


for i in X.columns:
    if(X[i].count()!=num_rows):
        print(i)


# 1.3) Замените пропуски на нули с помощью функции fillna(). На самом деле этот способ является предпочтительным для логистической регрессии, поскольку он позволит пропущенному значению не вносить никакого вклада в предсказание. Для деревьев часто лучшим вариантом оказывается замена пропуска на очень большое или очень маленькое значение — в этом случае при построении разбиения вершины можно будет отправить объекты с пропусками в отдельную ветвь дерева. Также есть и другие подходы — например, замена пропуска на среднее значение признака. Мы не требуем этого в задании, но при желании попробуйте разные подходы к обработке пропусков и сравните их между собой

# In[4]:


X=X.fillna(0)


# 1.4) Какой столбец содержит целевую переменную? Запишите его название.

# In[5]:


y=df['radiant_win']


# 1.5) Забудем, что в выборке есть категориальные признаки, и попробуем обучить градиентный бустинг над деревьями на имеющейся матрице "объекты-признаки". Зафиксируйте генератор разбиений для кросс-валидации по 5 блокам (KFold), не забудьте перемешать при этом выборку (shuffle=True), поскольку данные в таблице отсортированы по времени, и без перемешивания можно столкнуться с нежелательными эффектами при оценивании качества. Оцените качество градиентного бустинга (GradientBoostingClassifier) с помощью данной кросс-валидации, попробуйте при этом разное количество деревьев (как минимум протестируйте следующие значения для количества деревьев: 10, 20, 30). Долго ли настраивались классификаторы? Достигнут ли оптимум на испытанных значениях параметра n_estimators, или же качество, скорее всего, продолжит расти при дальнейшем его увеличении?

# In[6]:


kf = KFold(n_splits=5,shuffle=True)
mean_score=[]
for i in range(10,100,10):
    score=[]
    clf = ensemble.GradientBoostingClassifier(n_estimators=i)
    start_time = datetime.datetime.now()
    for train, test in kf.split(X, y):
        clf.fit(X.iloc[train], y.iloc[train])
        pred = clf.predict_proba(X.iloc[test])[:, 1]
        score.append(roc_auc_score(y.iloc[test], pred))
    mean_score.append(np.mean(score))
    print('Number of trees', i)
    print(score)
    print ('Time elapsed:', datetime.datetime.now() - start_time)
print(mean_score)


# In[7]:


plt.plot(range(10,100,10),mean_score)
plt.xlabel('Number of trees')
plt.ylabel('Score')
plt.show()


# # Отчет:
# 
# #### 1) Какие признаки имеют пропуски среди своих значений? Что могут означать пропуски в этих признаках (ответьте на этот вопрос для двух любых признаков)?
# 
# Признаки first_blood_time, first_blood_team, first_blood_player1, first_blood_player2 могут быть пропущены, если первая кровь так и не пролилась в первые 5 минут матча. Признаки radiant_bottle_time и dire_bottle_time могут быть пропущены, если команда не купила предмет "bottle" в первые 5 минут. Признаки radiant_courier_time, radiant_flying_courier_time, dire_courier_time, dire_flying_courier_time могут быть пропущены, если команда света/тьмы не купила курьера/летающего курьера в первые 5 минут. Признаки dire_first_ward_time и radiant_first_ward_time могут быть пропущены, если соответствующая команда не поставила вард в первые 5 минут.
# 
# #### 2) Как называется столбец, содержащий целевую переменную?
# 
# radiant_win - победа сил света
# 
# #### 3) Как долго проводилась кросс-валидация для градиентного бустинга с 30 деревьями? Инструкцию по измерению времени можно найти ниже по тексту. Какое качество при этом получилось? Напомним, что в данном задании мы используем метрику качества AUC-ROC.
# 
# Время выполнения на 30 деревьях по 5 блокам: 0:01:14.660263 AUC-ROC оценка при этом: 0.6897973787535268
# 
# #### 4) Имеет ли смысл использовать больше 30 деревьев в градиентном бустинге? Что бы вы предложили делать, чтобы ускорить его обучение при увеличении количества деревьев?
# 
# Оценка AUC-ROC монотонно возрастает при увеличении количества деревьев, поэтому имеет смысл использовать их как можно больше. Для ускорения сходимости можно попытаться выспользоваться усечением деревьев и ограичить их глубину

# # 2: Логистическая регрессия

# 2.1) Оцените качество логистической регрессии (sklearn.linear_model.LogisticRegression с L2-регуляризацией) с помощью кросс-валидации по той же схеме, которая использовалась для градиентного бустинга. Подберите при этом лучший параметр регуляризации (C). Какое наилучшее качество у вас получилось? Как оно соотносится с качеством градиентного бустинга? Чем вы можете объяснить эту разницу? Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?

# In[8]:


X = pd.DataFrame(StandardScaler().fit_transform(X))
mean_score=[]
C_range = [0.001, 0.01, 0.1, 1, 10, 100,1000,10000]
for i in C_range:
    score=[]
    start_time = datetime.datetime.now()
    for train, test in kf.split(X, y):
        clf = LogisticRegression(C = i,solver='lbfgs').fit(X.iloc[train], y.iloc[train])
        pred = clf.predict_proba(X.iloc[test])[:, 1]
        score.append(roc_auc_score(y.iloc[test], pred))
    mean_score.append(np.mean(score))
    print('C:', i)
    print(score)
    print ('Time elapsed:', datetime.datetime.now() - start_time)
print('max', max(mean_score), 'min',min(mean_score))

fig,ax = plt.subplots() 
ax.plot(mean_score,marker='o') 
ax.set_xticks([i for i in range(len(mean_score))]) 
ax.set_xticklabels(C_range) 
plt.show() 


# 2.2) Среди признаков в выборке есть категориальные, которые мы использовали как числовые, что вряд ли является хорошей идеей. Категориальных признаков в этой задаче одиннадцать: lobby_type и r1_hero, r2_hero, ..., r5_hero, d1_hero, d2_hero, ..., d5_hero. Уберите их из выборки, и проведите кросс-валидацию для логистической регрессии на новой выборке с подбором лучшего параметра регуляризации. Изменилось ли качество? Чем вы можете это объяснить?

# In[9]:


X=df.drop(['duration','radiant_win','tower_status_radiant','tower_status_dire','barracks_status_radiant','barracks_status_dire','lobby_type','r1_hero','r2_hero','r3_hero','r4_hero','r5_hero','d1_hero','d2_hero','d3_hero','d4_hero','d5_hero'],axis=1)
X=X.fillna(0)
X = pd.DataFrame(StandardScaler().fit_transform(X))
mean_score=[]
C_range = [0.001, 0.01, 0.1, 1, 10, 100,1000,10000]
for i in C_range:
    score=[]
    start_time = datetime.datetime.now()
    for train, test in kf.split(X, y):
        clf = LogisticRegression(C = i,solver='lbfgs').fit(X.iloc[train], y.iloc[train])
        pred = clf.predict_proba(X.iloc[test])[:, 1]
        score.append(roc_auc_score(y.iloc[test], pred))
    mean_score.append(np.mean(score))
    print('C:', i)
    print(score)
    print ('Time elapsed:', datetime.datetime.now() - start_time)
print('max', max(mean_score), 'min',min(mean_score))

fig,ax = plt.subplots() 
ax.plot(mean_score,marker='o') 
ax.set_xticks([i for i in range(len(mean_score))]) 
ax.set_xticklabels(C_range) 
plt.show() 


# 2.3) На предыдущем шаге мы исключили из выборки признаки rM_hero и dM_hero, которые показывают, какие именно герои играли за каждую команду. Это важные признаки — герои имеют разные характеристики, и некоторые из них выигрывают чаще, чем другие. Выясните из данных, сколько различных идентификаторов героев существует в данной игре (вам может пригодиться фукнция unique или value_counts).

# In[10]:


heroes = pd.read_csv('C:/Users/alexl/Downloads/data/dictionaries/heroes.csv',index_col='id')
print(len(heroes))


# 2.4) Воспользуемся подходом "мешок слов" для кодирования информации о героях. Пусть всего в игре имеет N различных героев. Сформируем N признаков, при этом i-й будет равен нулю, если i-й герой не участвовал в матче; единице, если i-й герой играл за команду Radiant; минус единице, если i-й герой играл за команду Dire. Ниже вы можете найти код, который выполняет данной преобразование. Добавьте полученные признаки к числовым, которые вы использовали во втором пункте данного этапа.

# In[11]:


X_pick = np.zeros((df.shape[0], len(heroes)))
for i, match_id in enumerate(df.index):
    for p in range(5):
        X_pick[i, df.loc[match_id, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, df.loc[match_id, 'd%d_hero' % (p+1)]-1] = -1
X_pick = pd.DataFrame(X_pick)
X = pd.concat([X, X_pick], axis=1)
X = pd.DataFrame(StandardScaler().fit_transform(X))


# 2.5) Проведите кросс-валидацию для логистической регрессии на новой выборке с подбором лучшего параметра регуляризации. Какое получилось качество? Улучшилось ли оно? Чем вы можете это объяснить?

# In[12]:


mean_score=[]
C_range = [0.001, 0.01, 0.1, 1, 10, 100,1000,10000]
for i in C_range:
    score=[]
    start_time = datetime.datetime.now()
    for train, test in kf.split(X, y):
        clf = LogisticRegression(C = i,solver='lbfgs').fit(X.iloc[train], y.iloc[train])
        pred = clf.predict_proba(X.iloc[test])[:, 1]
        score.append(roc_auc_score(y.iloc[test], pred))
    mean_score.append(np.mean(score))
    print('C:', i)
    print(score)
    print ('Time elapsed:', datetime.datetime.now() - start_time)
print('max', max(mean_score), 'min',min(mean_score))


fig,ax = plt.subplots() 
ax.plot(mean_score,marker='o') 
ax.set_xticks([i for i in range(len(mean_score))]) 
ax.set_xticklabels(C_range) 
plt.show() 


# # Отчет:
# #### 1) Какое качество получилось у логистической регрессии над всеми исходными признаками? Как оно соотносится с качеством градиентного бустинга? Чем вы можете объяснить эту разницу? Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?
# 
# Логистическая регрессия работает в разы быстрее градиентного бустинга. Качество логистической регрессии на AUC-ROC кривой колеблется в районе 0.71
# 
# #### 2) Как влияет на качество логистической регрессии удаление категориальных признаков (укажите новое значение метрики качества)? Чем вы можете объяснить это изменение?
# 
# Удаление категориальных признаков никак не повлияло на качество модели, это может означать что их вклад в модель сравним с нулевым
# 
# #### 3) Сколько различных идентификаторов героев существует в данной игре?
# 
# 112
# 
# #### 4) Какое получилось качество при добавлении "мешка слов" по героям? Улучшилось ли оно по сравнению с предыдущим вариантом? Чем вы можете это объяснить?
# 
# Применение "мешка слов" слегка улучшило модель, значение AUC-ROC возрасло до 0.75
# 
# #### 5) Какое минимальное и максимальное значение прогноза на тестовой выборке получилось у лучшего из алгоритмов?
# 
# Оптимальный из алгоритмов - с применением "мешка слов". Лучшее значение AUC-ROC = 0.7521256190907969 при C = 0.01, худшее = 0.7514800148504667 при C = 0.001
