# Hierarchical Clustering

# Data Preprocessing

# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the datasets
dataset  = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3,4]]

# Using the Dendogram to find the optimal number of cluster
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X,method = 'ward'))
plt.title('Dendogram')
plt.xlabel('customers')
plt.ylabel('Euclidean Distance')
plt.show()


#Fitting hierarchial Clustering to the mall datasets
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5,affinity = 'euclidean',linkage = 'ward')
y_hc = hc.fit_predict(X)


#Visualizing the clusters
plt.scatter(X[y_hc == 0,0],X[y_hc == 0,1],s = 100,c = 'red',label = 'category1')
plt.scatter(X[y_hc == 1,0],X[y_hc == 1,1],s = 100,c = 'green',label = 'category2')
plt.scatter(X[y_hc == 2,0],X[y_hc == 2,1],s = 100,c = 'blue',label = 'category3')
plt.scatter(X[y_hc == 3,0],X[y_hc == 3,1],s = 100,c = 'black',label = 'category4')
plt.scatter(X[y_hc == 4,0],X[y_hc == 4,1],s = 100,c = 'magenta',label = 'category5')
plt.title('Cluster of clients')
plt.xlabel('Annual Income')
plt.ylabel('yLabel')
plt.legend()
plt.show()


