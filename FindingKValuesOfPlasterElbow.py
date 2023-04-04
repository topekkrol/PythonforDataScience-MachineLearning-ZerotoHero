import numpy as np
import  matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.datasets import  make_blobs

plt.rcParams['figure.figsize'] = (16,9)

x,y = make_blobs(n_samples=800, n_features=3, centers=4)
print(x,y)
wcss_list = [] #Initializing the list for the values of WCSS

# 10 iterations

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)

plt.plot(range(1,11),wcss_list)
plt.title("the elbow method")
plt.xlabel("number of plaste")
plt.ylabel('wcss_list')
plt.show()