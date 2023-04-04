from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['figure.figsize'] = (16,9)

plt.style.use('ggplot')

#importing thr dataset

data= pd.read_csv('mallCustomerData.txt', sep=",")

f1 = data['Annual Income (k$)'].values
f2 = data['Spending Score (1-100)'].values


x = np.array(list(zip(f1,f2)))

# wygląd : x = [[1,1], [1,2], [3,3], [4,4]]

plt.scatter(f1,f2, c='black', s=20)
# plt.show()

#number of clusters:
kmeans = KMeans(n_clusters=3)

#fitting the input data

kmeans = kmeans.fit(x)

labels = kmeans.predict(x)

c = kmeans.cluster_centers_

fig = plt.figure()

ax = Axes3D(fig)

ax.scatter(x[:,0],x[:,1],x[:,2], c='y')

ax.scatter(c[:,0], c[:,1], c[:,2], marker='*', c='#050505', s=1000)

# initializing KMeans
kmeans = KMeans(n_clusters=4)

##fitting with inputs
kmeans = kmeans.fit(x)

#Predicting the clusters
labels = kmeans.predict(x)

#getting the cluster centers

c = kmeans.cluster_centers_

print(labels)