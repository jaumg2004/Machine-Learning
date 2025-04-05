from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

iris = datasets.load_iris()
print(iris)
print(iris.target)

def plot_cluster(data, labels, title):
    colors = ['red', 'green', 'blue', 'black']
    plt.figure(figsize=(8,4))
    for i, c, l in zip(range(-1, 3), colors, ['Noise', 'Setosa', 'Versicolor', 'Virginica']):
        if i == -1:
            plt.scatter(data[labels == i, 0], data[labels == i, 3], c=colors[i], label=l, alpha=0.5, s=50, marker='x')
        else:
            plt.scatter(data[labels == i, 0], data[labels == i, 3], c=colors[i], label=l, alpha=0.5, s=50)
    plt.legend()
    plt.title(title)
    plt.xlabel('Comprimento Sépala')
    plt.ylabel('Largura da Pétala')
    plt.show()

kmeans = KMeans(n_clusters=3, n_init='auto')
kmeans.fit(iris.data)
print(kmeans.labels_)
resultados = confusion_matrix(iris.target, kmeans.labels_)
print(resultados)

plot_cluster(iris.data, kmeans.labels_, 'Cluster KMeans')

dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan_labels = dbscan.fit_predict(iris.data)
print(dbscan_labels)

plot_cluster(iris.data, dbscan_labels, 'Cluster DBSCAN')

agglo = AgglomerativeClustering(n_clusters=3)
agglo_labels = agglo.fit_predict(iris.data)
print(agglo_labels)
resultados = confusion_matrix(iris.target, agglo_labels)
print(resultados)

plot_cluster(iris.data, agglo_labels, 'Cluster Agglo')

plt.figure(figsize=(12,6))
plt.title('Cluster Hierárquico: Dendrograma')
plt.xlabel('Índice')
plt.ylabel('Distância')
linkage_matix = linkage(iris.data, method='ward')
dendrogram(linkage_matix, truncate_mode='lastp', p=15)
plt.axhline(y=7, c='gray', lw=1, linestyle='dashed')
plt.show()