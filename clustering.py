import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import AgglomerativeClustering, DBSCAN

# ==============================
# PREPARATION DES DONNEES
# ==============================
def preparer_clustering(df):
 
    # garder colonnes numériques
    X = df.select_dtypes(include=["number"]).copy()

    # CONVERSION EN FLOAT 
    X = X.astype(float)

    # remplacer NaN par moyenne
    X = X.fillna(X.mean())

    return X

# ==============================
# K-MEANS
# ==============================
def appliquer_kmeans(X, k):
   

    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    centers = model.cluster_centers_

    # ajouter les clusters au dataset
    X_result = X.copy()
    X_result["Cluster"] = labels

    return X_result, centers


# ==============================
# K-MEDOIDS
# ==============================
def appliquer_kmedoids(X, k):
   

    model = KMedoids(n_clusters=k, random_state=42)
    labels = model.fit_predict(X)
    centers = model.cluster_centers_

    # ajouter les clusters au dataset
    X_result = X.copy()
    X_result["Cluster"] = labels

    return X_result, centers

from sklearn.cluster import AgglomerativeClustering, DBSCAN

# ==============================
# AGNES
# ==============================
def appliquer_agnes(X, k):
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X)

    X_result = X.copy()
    X_result["Cluster"] = labels

    return X_result, None


# ==============================
# DBSCAN
# ==============================
def appliquer_dbscan(X, eps=0.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    X_result = X.copy()
    X_result["Cluster"] = labels

    return X_result, None


# ==============================
# DIANA (approximation)
# ==============================
def appliquer_diana(X, k):
    model = AgglomerativeClustering(n_clusters=k, linkage='complete')
    labels = model.fit_predict(X)

    X_result = X.copy()
    X_result["Cluster"] = labels

    return X_result, None