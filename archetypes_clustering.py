"""
A group is defined by ['Housing type', 'Energy', 'DPE']
An archetype is define by ['Wall', 'Floor', 'Roof', 'Windows', 'Efficiency']

For each group, we want to obtain the centroid of the main groups, and the number of individuals in each group.
The centroid will determine the additional properties of its cluster.

We run a hierarchical clustering, and use Ward method and Euclidian metric.
"""

import pandas as pd

from scipy.spatial import cKDTree
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler


def find_k_closest(centroids, data, k=1, distance_norm=2):
    """
    Arguments:
    ----------
        centroids: (M, d) ndarray
            M - number of clusters
            d - number of data dimensions
        data: (N, d) ndarray
            N - number of data points
        k: int (default 1)
            nearest neighbour to get
        distance_norm: int (default 2)
            1: Hamming distance (x+y)
            2: Euclidean distance (sqrt(x^2 + y^2))
            np.inf: maximum distance in any dimension (max((x,y)))

    Returns:
    -------
        indices: (M,) ndarray
        values: (M, d) ndarray
    """

    kdtree = cKDTree(data)
    distances, indices = kdtree.query(centroids, k, p=distance_norm)
    if k > 1:
        indices = indices[:,-1]
    values = data[indices]
    return indices, values


def find_archetypes(df, n_clusters=None, distance_threshold=None):
    """Create clusters from data and return representative individuals for each cluster.

    Use AgglomerativeClustering method.

    Parameters
    ----------
    df: pd.DataFrame
        Initial data to cluster
    n_clusters: int, default None
    distance_threshold: float, default None

    Returns
    -------
    pd.DataFrame
        Representative individuals and weight.
    """
    data = StandardScaler().fit_transform(df.to_numpy())
    print('Start clustering')
    """clusters = AgglomerativeClustering(n_clusters=n_clusters, distance_threshold=distance_threshold, linkage='ward', affinity='euclidean').fit_predict(data)
    # calculate centroids
    print('Calculate centroids')
    clf = NearestCentroid().fit(data, clusters)"""
    clusters = KMeans(n_clusters=n_clusters).fit_predict(data)

    print('End clustering')

    df_cluster = pd.concat((df, pd.Series(clusters, index=df.index, name='clusters')), axis=1)


    # find closest individual from centroids
    idx, values = find_k_closest(clusters.cluster_centers_, data, k=1, distance_norm=2)

    df_centers = df_cluster.iloc[idx, :]

    # calculate weight for each representative
    scale = df_cluster['clusters'].value_counts().sort_index()
    scale = scale / scale.sum()
    scale.index = df_centers.index

    return pd.concat((df_centers, scale.rename('Weight')), axis=1).sort_values('Weight', ascending=False)


if __name__ == '__main__':

    from sklearn.cluster import KMeans
    import numpy as np

    X = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    print(X)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    kmeans.labels_

    kmeans.predict([[0, 0], [12, 3]])

    kmeans.cluster_centers_



    data = pd.read_csv('data_parsed.csv', index_col=[0, 1, 2, 3])

    var_group = ['Housing type', 'Energy', 'DPE']
    var_archetype = ['Wall', 'Floor', 'Roof', 'Windows', 'Efficiency']

    archetypes = dict()
    k = 0
    for n, g in data.groupby(var_group)[var_archetype]:
        print(n)
        print(g.shape)
        archetypes[n] = find_archetypes(g, n_clusters=3)
        k += 1
        print(k)

    archetypes = pd.concat(list(archetypes.values()))

    archetypes.to_csv('archetypes.csv')
    print('end')

