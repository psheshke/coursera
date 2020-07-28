
# coding: utf-8

# # Подход 1: градиентный бустинг "в лоб"

# In[10]:

#Загружаем используемые библиотеки
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score,make_scorer
import time
import datetime
from sklearn.linear_model import LogisticRegression


# In[55]:

#Загружаем массив с данными
features=pd.read_csv('features.csv', index_col='match_id' )
features_test=pd.read_csv('features_test.csv', index_col='match_id')


# In[56]:

#1.Считайте таблицу с признаками из файла features.csv с помощью кода, приведенного выше. 
#Удалите признаки, связанные с итогами матча (они помечены в описании данных как отсутствующие в тестовой выборке).
Train=features.filter(items=features_test.columns.values).iloc[:,1:]


# In[13]:

#2.Какие признаки имеют пропуски среди своих значений? Что могут означать пропуски в этих признаках (ответьте на этот вопрос для двух любых признаков)?
df5=pd.DataFrame(Train.count(),columns=['Count'])
df5.loc[df5['Count']!=97230]


# In[57]:

#3.Как называется столбец, содержащий целевую переменную? - Ответ: переменная radiant_win (если победила команда Radiant, 0 — иначе)
Target=features.loc[:,'radiant_win']
Train=Train.fillna(0)
Test=features_test.fillna(0).iloc[:,1:]


# In[32]:

#4.Как долго проводилась кросс-валидация для градиентного бустинга с 30 деревьями? Инструкцию по измерению времени можно найти выше по тексту. 
#Ответ: на моём компьютере обучение с 30 деревьями происходил в течение 1 минуты 01 секунд.
start_time = datetime.datetime.now()
kf = KFold(n_splits=5, shuffle=True, random_state=1)
model = GradientBoostingClassifier(n_estimators=30, random_state=241)
cross=cross_val_score(model,Train, Target, cv=kf,scoring='roc_auc')
print ('Time elapsed:', datetime.datetime.now() - start_time)
#Какое качество при этом получилось?
print(cross.mean())
#Ответ: в среднем площадь под roc кривой получилась равной 0,689


# In[127]:

#5.Имеет ли смысл использовать больше 30 деревьев в градиентном бустинге?
grid = {'n_estimators': np.arange(10,61,10)}
cv = KFold(n_splits=5, shuffle=True, random_state=1)
clf = GradientBoostingClassifier(random_state=241)
gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=cv)
Grid_result=gs.fit(Train, Target)
Grid_result.cv_results_['mean_test_score']
#Результаты: array([0.66483292, 0.68204034, 0.68951367, 0.69402998, 0.69720407, 0.69978775]). То есть, при росте количества 
#деревьев незначительно растёт площадь под  ROC-кривой при кросс-валидации - улучшается качество модели. Но с другой стороны растёт 
#риск переобучения и время обучения. В итоге, смысл наращивать количество деревьев есть, при наличие достаточных мощностей и свободного времени:)


# In[33]:

#Что можно сделать, чтобы ускорить его обучение при увеличении количества деревьев?
start_time = datetime.datetime.now()
kf = KFold(n_splits=5, shuffle=True, random_state=1)
model = GradientBoostingClassifier(n_estimators=30, random_state=241, max_depth=2)
cross=cross_val_score(model,Train, Target, cv=kf,scoring='roc_auc')
print ('Time elapsed:', datetime.datetime.now() - start_time)
#Можно упростить параметры модели. К примеру, уменьшить глубину дерева. По default стоит глубина 3, при уменьшении до 2.
#время обучения уменьшается с 1 минуты 01 секунд до 33 секунд.


# # Подход 2: логистическая регрессия

# In[6]:

#1. Какое качество получилось у логистической регрессии над всеми исходными признаками?
#Для начала определим оптимальное число С (Inverse of regularization strength). Так выборка большая, я сократил выборку при поиске до 5000 наблюдений.
cv = KFold(n_splits=5, shuffle=True, random_state=1)
features_sample=features.sample(n=5000)
Train_sample=features_sample.filter(items=features_test.columns.values).iloc[:,1:]
Target_sample=features_sample.loc[:,'radiant_win']
Train_sample=Train_sample.fillna(0)

def test_roc(kf, X, y):
    scores = list()
    C_range = np.arange(0.01,3,0.2)
    for C in C_range:
        model =LogisticRegression(C=C, penalty='l2', random_state=241)
        scores.append(cross_val_score(model, X, y, cv=kf, scoring='roc_auc'))

    return pd.DataFrame(scores, C_range).mean(axis=1).sort_values(ascending=False)

roc_auc=test_roc(cv,Train_sample,Target_sample)
'''
Результаты поиска ниже:
1.0    0.692860
0.8    0.692741
0.6    0.692647
0.4    0.692604
1.4    0.692576
1.8    0.692536
1.6    0.692507
1.2    0.692277
Оптимальным значением является 1.
'''


# In[31]:

#Ответ: теперь подсчитаем качество на всей выборке 
kf = KFold(n_splits=5, shuffle=True, random_state=1)
model = LogisticRegression(C=1, random_state=241)
cross=cross_val_score(model,Train, Target, cv=kf,scoring='roc_auc')
print(cross.mean())
#В итоге, AUC=0.7164100


# In[ ]:

#Как оно соотносится с качеством градиентного бустинга?
#Ответ: качество логичтической регрессии (AUC=0.71) выше чем качество градиентного бустинга (0.69)


# In[ ]:

#Чем можно объяснить эту разницу? 
#Ответ: В теории градиентный бустинг должжен давать более точные результат, так как он не требователен к выборке и в целом заточен на повышение точности.
#Но с другой стороны бустинг склонен к переобучению, что при большом количестве признаков (как в этой выборке) становится более вероятно.
#А логистическая регрессия учитывает все признаки, при этом невелируя переобучение l2 регулиразацией.


# In[34]:

#Быстрее ли работает логистическая регрессия по сравнению с градиентным бустингом?
start_time = datetime.datetime.now()
kf = KFold(n_splits=5, shuffle=True, random_state=1)
model = LogisticRegression(C=1, random_state=241)
cross=cross_val_score(model,Train, Target, cv=kf,scoring='roc_auc')
print ('Time elapsed:', datetime.datetime.now() - start_time)
#Ответ: регрессия обучилась за 1 минуту 48 секунд,что медленее градиентного бустинга (1 минута 1 секунда)


# In[64]:

#2.Как влияет на качество логистической регрессии удаление категориальных признаков (укажите новое значение метрики качества)? Чем можно объяснить это изменение?
Train_without_categorial=Train.loc[:,~Train.columns.str.contains('hero', case=False)].drop('lobby_type',axis=1)
kf = KFold(n_splits=5, shuffle=True, random_state=1)
model = LogisticRegression(C=1, random_state=241)
cross=cross_val_score(model,Train_without_heros, Target, cv=kf,scoring='roc_auc')
print(cross.mean())
#Качество незначительно повысилось с 0.71641 до 0.71644, что говорит о том, что данные переменные ранее использованные как непрерывные перменные,
#несли слабую предскзательную способность, и даже вносили шум.


# In[83]:

#3.Сколько различных идентификаторов героев существует в данной игре?
#Создаём DataFrame только с полями имен героев
df2=Train.filter(regex='hero').reset_index(drop=True)
heros=list()
for i in range(len(df2.columns)):
    for j in range(len(df2)):
        heros.append(df2.iloc[j,i])

#Создаём функцию делающую уникальными значения листа
def unique(list1):
    unique_list = [] 
    for x in list1: 
        if x not in unique_list: 
            unique_list.append(x) 
    return(unique_list)

print(len(unique(heros)))
print(max(unique(heros)))
#Ответ: всего уникальных 108 героев в массиве, но если предположить, что герои пронумерованы по порядку, то максимальное число 
#в массиве равно 112, то есть 4-ых героев не выберали. В итоге ответ 112.


# In[15]:

#4.Какое получилось качество при добавлении "мешка слов" по героям? 
#Создаём Dataframe с мешком слов
X_pick = np.zeros((Train.shape[0], 112))
for i,j in enumerate(Train.index):
    for p in range(5):
        X_pick[i, Train.ix[j, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, Train.ix[j, 'd%d_hero' % (p+1)]-1] = -1
df3=pd.DataFrame(X_pick,columns=np.arange(0,112,1))
df3.index.name='id'


# In[16]:

#Так как поле в тренирвочной выборке match_id не по порядку (часть массива ушла на тест), следовательно лучше создать новый id 
Train=Train.reset_index()
Train.drop('match_id', axis=1)
Train.index.name='id'
#Соединяем оба массива
Train_with_heroes=pd.merge(df3,Train,on='id', how='inner')


# In[133]:

#В итоге качество (AUC) на кросс-валидации полуслось равным 0.74
kf = KFold(n_splits=5, shuffle=True, random_state=1)
model = LogisticRegression(C=1, random_state=241)
cross=cross_val_score(model,Train_with_heroes, Target, cv=kf,scoring='roc_auc')
print(cross.mean())


# In[ ]:

#Улучшилось ли оно по сравнению с предыдущим вариантом?
#Качество улучшилось (AUC) с 0.716 до 0.74


# In[ ]:

#Чем можно это объяснить?
#Новая информация про героев игроков увеличила предсказательную способность модели, то есть команда, где  игроки, выбреающие определенных "сильных" героев 
#большую вероятность выиграть, и наоборот, если в команде противника вибрали "сильных" героев, то вероятность проиграть увличивается.


# In[17]:

#5.Какое минимальное и максимальное значение прогноза на тестовой выборке получилось у лучшего из алгоритмов?
#Лучший из алгоритмов является логистическая регрессия с инкодингом героев: 
model = LogisticRegression(C=1, random_state=241).fit(Train_with_heroes,Target)


# In[18]:

#Преобразуем тестовую выборку:
X_pick = np.zeros((Test.shape[0], 112))
for i,j in enumerate(Test.index):
    for p in range(5):
        X_pick[i, Test.ix[j, 'r%d_hero' % (p+1)]-1] = 1
        X_pick[i, Test.ix[j, 'd%d_hero' % (p+1)]-1] = -1
df3=pd.DataFrame(X_pick,columns=np.arange(0,112,1))
df3.index.name='id'
Test=Train.reset_index()
Test.drop('match_id', axis=1)
Test.index.name='id'
Test_with_heroes=pd.merge(df3,Train,on='id', how='inner')


# In[69]:

#Применяем модель к тестовой выборке и выгружаем результаты:
Test_result_final=pd.DataFrame(model.predict_proba(Test_with_heroes),index=Test.index, columns=['s', 'radiant_win'])
Test_result_final.index.name='match_id'
Test_2=pd.DataFrame(Test_result_final.loc[:,'radiant_win'])
Test_2.to_csv('TEST_RESULT_FINAL.csv', sep=',', encoding='utf-8')
#Результирующий файл был проверен на сайте kaggle.com. AUC равняется 0.52763


# In[72]:

#Теперь проверим качество на алгоритме без инкодинга героев:
model_3  = LogisticRegression(C=1, random_state=241).fit(Train,Target)


# In[73]:

#Применяем модель к тестовой выборке и выгружаем результаты:
Test_result_final=pd.DataFrame(model_3.predict_proba(Test),index=Test.index, columns=['s', 'radiant_win'])
Test_result_final.index.name='match_id'
Test_2=pd.DataFrame(Test_result_final.loc[:,'radiant_win'])
Test_2.to_csv('TEST_RESULT_FINAL_7.csv', sep=',', encoding='utf-8')
#Результирующий файл был проверен на сайте kaggle.com. AUC равняется 0.72258

