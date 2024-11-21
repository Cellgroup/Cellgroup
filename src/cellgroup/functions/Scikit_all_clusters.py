import matplotlib.pyplot as plt
import pandas as pd
import hdbscan
from sklearn.cluster import Birch, KMeans, StandardScaler, SpectralClustering, DBSCAN, AgglomerativeClustering
from sklearn.cluster import OPTICS, AffinityPropagation

def scikit_all_clusters(df, cluster_N, h_cluster_sie):
    # Drop the 'label' column if it's causing issues
    if 'Label' in df.columns:
        df = df.drop(columns=['Label'])
    df = df.dropna()
    X = df[["X", "Y"]].copy()

    # Obtain kmeans clustering
    kmeans = KMeans(n_clusters=cluster_N).fit(X)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=h_cluster_sie)
    h_cluster_labels = clusterer.fit_predict(X)
    brc = Birch(branching_factor=50, n_clusters=None, threshold=0.5, compute_labels=True)
    brc_labels = brc.fit_predict(X)

    optics = OPTICS()
    opt_labels = optics.fit_predict(X)

    db = DBSCAN(eps=10, min_samples=50).fit(X)
    db_labels = db.labels_

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clustering = SpectralClustering(n_clusters=3, assign_labels="discretize", random_state=0)
    spec_labels = clustering.fit_predict(X_scaled)

    clustering = AffinityPropagation()
    clustering.fit(X)
    cluster_centers_indices = clustering.cluster_centers_indices_
    AP_labels = clustering.labels_

    clustering = AgglomerativeClustering(n_clusters=7, linkage='ward')
    AC_labels = clustering.fit_predict(X)

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    axs = axs.ravel()

    scatter_args = {'cmap': 'viridis', 's': 20}

    axs[0].scatter(X['X'], X['Y'], c=kmeans.labels_, **scatter_args)
    axs[0].set_title('KMeans')

    axs[1].scatter(X['X'], X['Y'], c=h_cluster_labels, **scatter_args)
    axs[1].set_title('HDBSCAN')

    axs[2].scatter(X['X'], X['Y'], c=brc_labels, **scatter_args)
    axs[2].set_title('Birch')

    axs[3].scatter(X['X'], X['Y'], c=db_labels, **scatter_args)
    axs[3].set_title('DBSCAN')

    axs[4].scatter(X['X'], X['Y'], c=spec_labels, **scatter_args)
    axs[4].set_title('Spectral Clustering')

    axs[5].scatter(X['X'], X['Y'], c=opt_labels, **scatter_args)
    axs[5].set_title('OPTICS')

    axs[6].scatter(X['X'], X['Y'], c=AP_labels, **scatter_args)
    axs[6].set_title('Affinity Propagation')

    axs[7].scatter(X['X'], X['Y'], c=AC_labels, **scatter_args)
    axs[7].set_title('Agglomerative Clustering')

    plt.tight_layout()
    plt.show()
