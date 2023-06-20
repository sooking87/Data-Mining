from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.datasets import make_blobs

# K-means
def k_means():
    n_clusters = int(input("n_clusters: "))
    random_state = int(input('random_state: '))
    
    X, _ = make_blobs(n_samples=100, centers=5, 
            n_features=2, random_state=0)
    clustering = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto').fit(X)

    print('<Clustering result>')
    print(clustering.labels_, '\n')

# Hierarchical clustering
def hierarchical_clustering():
    n_clusters = int(input('n_clusters: '))
    linkage = input('linkage: ')

    X, _ = make_blobs(n_samples=100, centers=5, 
            n_features=2, random_state=0)
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage).fit(X)

    print('<Clustering result>')
    print(clustering.labels_, '\n')

# DBSCAN
def dbscan():
    eps = float(input('eps: '))
    min_samples = int(input('min_samples: '))

    X, _ = make_blobs(n_samples=100, centers=5, 
            n_features=2, random_state=0)
    print(eps)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    print('<Clustering result>')
    print(clustering.labels_, '\n')

# menu
while True:
    print("[ Student ID: 2116313 ]")
    print("[ Name: 손수경 ]\n")

    print("1. K-means")
    print("2. Hierarchical clustering")
    print("3. DBSCAN")
    print("4. Quit")

    menu = int(input())

    if menu == 1:
        k_means()
    elif menu == 2:
        hierarchical_clustering()
    elif menu == 3:
        dbscan()
    elif menu == 4:
        break
