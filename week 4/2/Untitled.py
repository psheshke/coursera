
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import scale, StandardScaler
from sklearn.datasets import load_boston
from sklearn.linear_model import Perceptron, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score,precision_recall_curve,confusion_matrix
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.decomposition import PCA


# In[ ]:


salary_train = pd.read_csv('salary-train.csv')

salary_train

