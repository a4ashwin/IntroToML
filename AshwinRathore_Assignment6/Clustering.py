from pandas import DataFrame
from pandas import read_csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score
from kneed import KneeLocator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#Loading Iris Data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
#Part 1
array=dataset.values
X=array[:,0:4]
y=array[:,4]

current_inertia=100.000
i=1
print("Part 1: k-Means Clustering")
while i <= 100:
    kmeans = KMeans( n_clusters=3, n_init=i)
    kmeans.fit(X)   
    if kmeans.inertia_ < current_inertia:
        print("Iteration: ", i)
        print("Reconstruction Error: ", kmeans.inertia_)
        if((current_inertia-kmeans.inertia_)*100/current_inertia < 1):
            current_inertia = kmeans.inertia_
            break
        current_inertia = kmeans.inertia_
    i+=1
sse = []
for k in range(2, 21):
    kmeans = KMeans(n_clusters=k, n_init = i)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
plt.plot(range(2, 21), sse)
plt.xticks(range(2, 21))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()
print("elbow_k: ", 3)

k_labels = kmeans.labels_  # Get cluster labels
k_labels_matched = np.empty_like(k_labels)

kmeans = KMeans(n_clusters=3, n_init = i)
kmeans.fit(X)
predictions = kmeans.predict(X)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
z = le.transform(y)

for k in np.unique(k_labels):
    # ...find and assign the best-matching truth label
    match_nums = [np.sum((k_labels==k)*(z==t)) for t in np.unique(z)]
    k_labels_matched[k_labels==k] = np.unique(z)[np.argmax(match_nums)]
print("Accuracy:", accuracy_score(z,k_labels_matched))
print("Confusion matrix:")
print(confusion_matrix(z,k_labels_matched))

print()
print()
print("Part 2: Gaussian Mixture Model GMM Clustering")
from sklearn.mixture import GaussianMixture
i=1
current_lower_bound = -100
while i <= 100:
    gmm = GaussianMixture( n_components=3, n_init=i)
    gmm.fit(X)   
    if gmm.lower_bound_ > current_lower_bound:
        print("Iteration: ", i)
        print(gmm.lower_bound_)
        #print(gmm.n_iter_)0
        if((abs(current_lower_bound-gmm.lower_bound_))*100/abs(current_lower_bound) < 1):
            current_lower_bound = gmm.lower_bound_
            break
        current_lower_bound = gmm.lower_bound_
    i+=1

aics = []
for k in range(2, 21):
    gmm = GaussianMixture(n_components=k, n_init = i, covariance_type = 'diag')
    gmm.fit(X)
    aics.append(gmm.aic(X))
plt.style.use("fivethirtyeight")
plt.plot(range(2, 21), aics)
plt.xticks(range(2, 21))
plt.xlabel("Number of Clusters")
plt.ylabel("aic")
plt.show()
print("aic_elbow_k: ", 11)
print("“Cannot calculate Accuracy Score because the number of classes is not the same as the number of clusters")

print("Confusion Matrix: ")


gmm = GaussianMixture(n_components=11, n_init = i, covariance_type = 'diag')
gmm.fit(X)
g_labels = gmm.predict(X)
print(confusion_matrix(z,g_labels))

bics = []
for k in range(2, 21):
    gmm = GaussianMixture(n_components=k, n_init = i, covariance_type = 'diag')
    gmm.fit(X)
    bics.append(gmm.bic(X))
plt.style.use("fivethirtyeight")
plt.plot(range(2, 21), bics)
plt.xticks(range(2, 21))
plt.xlabel("Number of Clusters")
plt.ylabel("bic")
plt.show()
print("bic_elbow_k: ", 14)
print("“Cannot calculate Accuracy Score because the number of classes is not the same as the number of clusters")
print("Confusion Matrix: ")
gmm = GaussianMixture(n_components=14, n_init = i, covariance_type = 'diag')
gmm.fit(X)
g_labels = gmm.predict(X)
print(confusion_matrix(z,g_labels))

print("For k =  ", 3)
gmm = GaussianMixture(n_components=3, n_init = i, covariance_type = 'diag')
gmm.fit(X)
g_labels = gmm.predict(X)
g_labels_matched = np.empty_like(g_labels)
for k in np.unique(g_labels):
    # ...find and assign the best-matching truth label
    match_nums = [np.sum((g_labels==k)*(z==t)) for t in np.unique(z)]
    g_labels_matched[g_labels==k] = np.unique(z)[np.argmax(match_nums)]
print("Accuracy:", accuracy_score(z,g_labels_matched))
print("Confusion matrix:")
print(confusion_matrix(z,g_labels_matched))
