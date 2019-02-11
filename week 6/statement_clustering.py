
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import scale, StandardScaler
from sklearn.datasets import load_boston
from sklearn.linear_model import Perceptron, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score,     precision_recall_curve,confusion_matrix, r2_score, log_loss

from sklearn.svm import SVC
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.decomposition import PCA
from scipy.stats.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier,     GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from skimage.io import imread
from skimage import img_as_float
import pylab
from time import time


# In[2]:


image = imread('parrots.jpg')

image


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
pylab.imshow(image)


# In[10]:


binimage = img_as_float(image)

binimage


# In[11]:


pylab.imshow(binimage)


# In[12]:


len(binimage)*len(binimage[0])


# In[13]:


objects = np.zeros((len(binimage)*len(binimage[0]), 3), dtype=np.float64)

objects


# In[14]:


pd.DataFrame(objects)


# In[15]:


k = 0

for i in range(len(binimage)):
    
    for j in range(len(binimage[i])):
        
        objects[k][0] = binimage[i][j][0]
        objects[k][1] = binimage[i][j][1]
        objects[k][2] = binimage[i][j][2]
        
        k = k+1
        
pd.DataFrame(objects, columns = ['r','g','b'])


# In[16]:


colors = 16

kmeans = KMeans(random_state=241, init='k-means++', n_clusters = colors).fit(objects)

kmeans


# In[17]:


kmeans.cluster_centers_ 


# In[18]:


kmeans.predict(objects)


# In[19]:


df = pd.DataFrame(objects, columns = ['r','g','b'])
df['clust'] = pd.Series(kmeans.predict(objects))

df


# In[20]:


mean_color = np.zeros((colors, 3))

median_color = np.zeros((colors, 3))

for i in range(colors):
    
    mean_color[i] = np.mean(df[df['clust'] == i])[['r','g','b']]
    
    median_color[i][0] = np.median(df[df['clust'] == i]['r'])
    median_color[i][1] = np.median(df[df['clust'] == i]['g'])
    median_color[i][2] = np.median(df[df['clust'] == i]['b'])
    

mean_color


# In[21]:


median_color


# In[22]:


image_mean = np.zeros((len(binimage), len(binimage[0]), 3))

image_median = np.zeros((len(binimage), len(binimage[0]), 3))

image_mean


# In[31]:


k = 0

for i in range(len(binimage)):
    
    for j in range(len(binimage[i])):
        
        cl = int(df.loc[k]['clust'])
        
        image_mean[i][j] = mean_color[cl]
        
        image_median[i][j] = median_color[cl]
        
        k = k+1


# In[32]:


image_mean


# In[37]:


pylab.imshow(image_mean)


# In[38]:


pylab.imshow(image_median)


# In[65]:


def psnr(image_mean, image, image_median):
    
    mse = np.mean((image - image_mean) ** 2)

    psnr1 = 10 * math.log10(float(1) / mse)

    mse = np.mean((image - image_median) ** 2)

    psnr2 = 10 * math.log10(float(1) / mse)

    return psnr1, psnr2


# In[67]:


colors = 0

for colors in range(160):
    
    colors = colors + 1

    kmeans = KMeans(random_state=241, init='k-means++', n_clusters = colors).fit(objects)

    df = pd.DataFrame(objects, columns = ['r','g','b'])
    df['clust'] = pd.Series(kmeans.predict(objects))

    mean_color = np.zeros((colors, 3))

    median_color = np.zeros((colors, 3))

    for i in range(colors):

        mean_color[i] = np.mean(df[df['clust'] == i])[['r','g','b']]

        median_color[i][0] = np.median(df[df['clust'] == i]['r'])
        median_color[i][1] = np.median(df[df['clust'] == i]['g'])
        median_color[i][2] = np.median(df[df['clust'] == i]['b'])


    image_mean = np.zeros((len(binimage), len(binimage[0]), 3))

    image_median = np.zeros((len(binimage), len(binimage[0]), 3))

    k = 0

    for i in range(len(binimage)):

        for j in range(len(binimage[i])):

            cl = int(df.loc[k]['clust'])

            image_mean[i][j] = mean_color[cl]

            image_median[i][j] = median_color[cl]

            k = k+1

    psnr1, psnr2 = psnr(image_mean, binimage, image_median)
    
    print("Colors = ",colors, " PSNR = ", max(psnr1, psnr2))
    
    if max(psnr1, psnr2) >= 20:
        
        break
        
pylab.imshow(image_mean)

print("Минимум цветов = ", color, " PSNR = ", max(psnr1, psnr2))

