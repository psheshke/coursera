# -*- coding: utf-8 -*-
"""
Редактор Spyder

Это временный скриптовый файл.
"""

import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

def GB(data, result):
    gen = KFold(n_splits=5,shuffle=True,random_state=1)
    for train, test in gen.split(data):
        data_train, data_test = data[train],data[test]
        result_train, result_test = result[train], result[test]
        
    check_res= []
    score = []    
    for i in range(1,4):
        start_time = datetime.datetime.now()
        
        clf = GradientBoostingClassifier(n_estimators=i*10, random_state=1)
        clf.fit(data_train,result_train)
        
        pred = clf.predict_proba(data_test)[:, 1]
        roc_auc = roc_auc_score(result_test,pred)
        score.append(roc_auc)
        
        check = cross_val_score(clf,data_test,result_test,cv=gen,scoring='roc_auc')
        check = (sum(check)/5)
        check_res.append(check)
        
        print ('Время кроссвалидации (',i*10,' деревьев): ', datetime.datetime.now()-start_time)
    return check_res
    
def LR(data, result):
    gen = KFold(n_splits=5,shuffle=True,random_state=1)
    for train, test in gen.split(data):
        data_train, data_test = data[train],data[test]
        result_train, result_test = result[train], result[test]
        
    check_res= []
    score = []    
    for i in range(1,4):
        start_time = datetime.datetime.now()
        
        clf = LogisticRegression(penalty='l2',solver='lbfgs',C=10**(-i))
        clf.fit(data_train,result_train)
        
        pred = clf.predict_proba(data_test)[:, 1]
        roc_auc = roc_auc_score(result_test,pred)
        score.append(roc_auc)
        
        
        check = cross_val_score(clf,data_test,result_test,cv=gen,scoring='roc_auc')
        check = (sum(check)/5)
        check_res.append(check)
        
        print ('Время кроссвалидации (c=',10**(-i),'): ', datetime.datetime.now()-start_time)
    return check_res

    
data_unprepared = pd.read_csv('features.csv')   #Первоначально загруженные данные
result = data_unprepared['radiant_win']         #Целевая переменная



data1 = data_unprepared.copy()                   #Массив для обработки
del data1['duration']
del data1['radiant_win']
del data1['tower_status_radiant']
del data1['tower_status_dire']
del data1['barracks_status_radiant']
del data1['barracks_status_dire']


skips = data1.count()                            #Подсчет пропусков
data1 = data1.fillna(0)                           #Замена пропусков нулями

data2 = data1

scaler = StandardScaler()
data1 = scaler.fit_transform(data1)

gb = GB(data1, result)
lr1 = LR(data1,result)

del data2['r1_hero']
del data2['r2_hero']
del data2['r3_hero']
del data2['r4_hero']
del data2['r5_hero']
del data2['d1_hero']
del data2['d2_hero']
del data2['d3_hero']
del data2['d4_hero']
del data2['d5_hero']
data2 = scaler.fit_transform(data2)
lr2 = LR(data2,result)

heroes_mas = data_unprepared['r1_hero'],data_unprepared['r2_hero'],data_unprepared['r3_hero'], data_unprepared['r4_hero'], data_unprepared['r5_hero'],data_unprepared['d1_hero'], data_unprepared['d2_hero'], data_unprepared['d3_hero'],data_unprepared['d4_hero'],data_unprepared['d5_hero']
unique_heroes = np.unique(heroes_mas)
unique_heroes_len = len(unique_heroes)

# С данного момента начинаются костыли... мешок прикручивал как мог (были проблемы с numpy)
del data_unprepared['duration']
del data_unprepared['radiant_win']
del data_unprepared['tower_status_radiant']
del data_unprepared['tower_status_dire']
del data_unprepared['barracks_status_radiant']
del data_unprepared['barracks_status_dire']
data_unprepared = data_unprepared.fillna(0)

N = max(unique_heroes)
X  = np.zeros((data_unprepared.shape[0], N))
for m, match_id in enumerate(data_unprepared.index):
    for n in range(5):
        X[m, data_unprepared.loc[match_id, 'r%d_hero' % (n+1)]-1] = 1
        X[m, data_unprepared.loc[match_id, 'd%d_hero' % (n+1)]-1] = -1
data3 = np.concatenate([data_unprepared,X],axis=1)

data3 = scaler.fit_transform(data3)
lr3 = LR(data3,result)
 

