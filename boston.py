# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 10:29:28 2019

@author: Waggoner
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
from scipy.stats import mode
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer

boston = load_boston()

boston.data
bos_df = pd.DataFrame(boston.data, columns = boston.feature_names)
bos_df['PRICE'] = boston.target

bos_df['PRICE'].describe()

def bucket_by_5(row):
    if 5 <= row < 10:
        return 0
    elif 10 <= row < 15:
        return 1
    elif 15 <= row < 20:
        return 2
    elif 20 <= row < 25:
        return 3
    elif 25 <= row < 30:
        return 4
    elif 30 <= row < 35:
        return 5
    elif 35 <= row < 40:
        return 6
    elif 40 <= row < 45:
        return 7

def bucket_by_10(row):
    if 5 <= row < 15:
        return 0
    elif 15 <= row < 25:
        return 1
    elif 25 <= row < 35:
        return 2
    elif 35 <= row < 55:
        return 3
    elif 45 <= row < 55:
        return 4
       
np.quantile(bos_df['PRICE'], [0,.2,.4,.6,.8,1])

def bucket_by_quant(row):
    if 5 <= row < 15.3:
        return 0
    elif 15.3 <= row < 19.7:
        return 1
    elif 19.7 <= row < 22.7:
        return 2
    elif 22.7 <= row < 28.2:
        return 3
    elif 28.2 <= row < 51:
        return 4

np.quantile(bos_df['PRICE'], [0,.33,.66,1])

def bucket_by_tris(row):
    if 5 <= row < 18.765:
        return 0
    elif 18.765 <= row < 23.53:
        return 1
    elif 23.53 <= row < 51:
        return 2

       
np.quantile(bos_df['PRICE'], [0,.25,.50,.75,1])

def bucket_by_quad(row):
    if 5 <= row < 17.025:
        return 0
    elif 17.025 <= row < 21.2:
        return 1
    elif 21.2 <= row < 25:
        return 2
    elif 25 <= row < 51:
        return 3

bos_df['PRICE_BUCKET'] = bos_df['PRICE'].apply(bucket_by_quad)


enc = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy = 'quantile')

enc.fit(bos_df['PRICE'].to_numpy().reshape(-1,1))

T = enc.transform(bos_df['PRICE'].to_numpy().reshape(-1,1))

bos_df['PRICE_BUCKET'] = T


bos_df.dtypes

# should be even with quantiles
sns.catplot('PRICE_BUCKET',
            kind = 'count',
            data = bos_df)

sns.distplot(bos_df['PRICE'])

sns.relplot(x = 'DIS',
            y = 'NOX',
            hue = 'PRICE_BUCKET',
            data = bos_df)

sns.catplot(x = 'PRICE_BUCKET',
            y = 'INDUS',
            kind = 'box',
            data = bos_df)

pairs = sns.pairplot(bos_df, diag_kind = 'hist', hue = 'PRICE_BUCKET')
#pairs = pairs.map_diag(plt.hist)
#pairs = pairs.map_offdiag(plt.scatter)

"""
bos_df.iloc[:,0] = stats.boxcox(bos_df.iloc[:,0])[0] 
#bos_df.iloc[:,1] = stats.boxcox(bos_df.iloc[:,1])[0] 
bos_df.iloc[:,2] = stats.boxcox(bos_df.iloc[:,2])[0] 
#bos_df.iloc[:,3] = stats.boxcox(bos_df.iloc[:,3])[0] 
bos_df.iloc[:,4] = stats.boxcox(bos_df.iloc[:,4])[0] 
bos_df.iloc[:,5] = stats.boxcox(bos_df.iloc[:,5])[0] 
bos_df.iloc[:,6] = stats.boxcox(bos_df.iloc[:,6])[0] 
bos_df.iloc[:,7] = stats.boxcox(bos_df.iloc[:,7])[0] 
bos_df.iloc[:,8] = stats.boxcox(bos_df.iloc[:,8])[0] 
bos_df.iloc[:,9] = stats.boxcox(bos_df.iloc[:,9])[0] 
bos_df.iloc[:,10] = stats.boxcox(bos_df.iloc[:,10])[0] 
bos_df.iloc[:,11] = stats.boxcox(bos_df.iloc[:,11])[0] 
bos_df.iloc[:,12] = stats.boxcox(bos_df.iloc[:,12])[0] 
#bos_df.iloc[:,13] = stats.boxcox(bos_df.iloc[:,13])[0] 
"""

x = bos_df.iloc[:,0:-2]
y = bos_df.iloc[:,-1]

#model = KMeans(n_clusters = 5)

model = SpectralClustering(n_clusters = 4,
                           assign_labels="discretize",
                           random_state=0,
                           #eigen_solver = 'lobpcg',
                           affinity = 'linear',
                           #n_init = 6,
                           n_neighbors = 20
                          )
model.fit(x)

#pairs = sns.pairplot(bos_df, diag_kind = 'hist', hue = 'PRED')

#aligning kmean classes w/ actual class labels

labels = np.zeros_like(model.labels_)
for i in range(5):
    mask = (model.labels_ == i)
    labels[mask] = mode(y[mask])[0]

print("kmeans model accuracy: " + str(accuracy_score(y, labels)))

result_mat = confusion_matrix(y,labels)

print(result_mat)

mat_df = pd.DataFrame(result_mat)

sns.heatmap(result_mat,
            square = True,
            annot = True,
            fmt = 'd',
            cbar = False)
        
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.show()

