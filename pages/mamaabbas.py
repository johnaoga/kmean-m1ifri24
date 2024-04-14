import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

def load_data(file_path):
    """
    Charge les données à partir d'un fichier CSV.

    Args:
        file_path (str): Chemin vers le fichier CSV.

    Returns:
        pandas.DataFrame: Les données chargées depuis le fichier CSV.
    """
    data = pd.read_csv(file_path)
    return data

def k_means(data, k, max_iterations=100):
    """
    Implémente l'algorithme K-means.

    Args:
        data (pandas.DataFrame): Les données d'entrée.
        k (int): Le nombre de clusters.
        max_iterations (int): Nombre maximal d'itérations.

    Returns:
        numpy.ndarray: Les centroids finaux.
        numpy.ndarray: Les clusters finaux.
    """
    # Initialisation des centroids de manière aléatoire
    centroids = data.sample(n=k, replace=True).values

    for _ in range(max_iterations):
        # Attribution des points aux clusters les plus proches
        clusters = np.argmin(np.linalg.norm(data.values[:, None] - centroids, axis=2), axis=1)

        # Mise à jour des centroids
        new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])

        # Vérification de la convergence
        if np.array_equal(centroids, new_centroids):
            break

        centroids = new_centroids

    return centroids, clusters

def predict_cluster(point, centroids):
    """
    Prédit le cluster auquel un point appartient en fonction des centroids.

    Args:
        point (list): Les valeurs des caractéristiques du point.
        centroids (numpy.ndarray): Les centroids calculés par K-means.

    Returns:
        int: Le numéro du cluster prédit.
    """
    distances = np.linalg.norm(centroids - point, axis=1)
    return np.argmin(distances)

def compute_wcss(data, centroids, clusters):
    """
    Calcule la somme des carrés intra-cluster.

    Args:
        data (pandas.DataFrame): Les données d'entrée.
        centroids (numpy.ndarray): Les centroids finaux.
        clusters (numpy.ndarray): Les clusters finaux.

    Returns:
        float: La somme des carrés intra-cluster (WCSS).
    """
    wcss = 0
    for i, centroid in enumerate(centroids):
        wcss += np.sum((data[clusters == i] - centroid) ** 2)
    return wcss

def compute_bcss(data, centroids, clusters):
    """
    Calcule la somme des carrés inter-cluster.

    Args:
        data (pandas.DataFrame): Les données d'entrée.
        centroids (numpy.ndarray): Les centroids finaux.
        clusters (numpy.ndarray): Les clusters finaux.

    Returns:
        float: La somme des carrés inter-cluster (BCSS).
    """
    bcss = 0
    centroid_global = np.mean(data, axis=0)
    for i, centroid in enumerate(centroids):
        bcss += len(data[clusters == i]) * np.sum((centroid - centroid_global) ** 2)
    return bcss

def compute_cluster_metrics(data, centroids, clusters):
    """
    Calcule les métriques pour chaque cluster.

    Args:
        data (pandas.DataFrame): Les données d'entrée.
        centroids (numpy.ndarray): Les centroids finaux.
        clusters (numpy.ndarray): Les clusters finaux.

    Returns:
        dict: Un dictionnaire contenant les métriques pour chaque cluster.
    """


    cluster_metrics = {}
    for i, centroid in enumerate(centroids):
        cluster_data = data[clusters == i]
        cluster_metrics[f'Cluster {i+1}'] = {
            'Nombre d\'éléments': len(cluster_data),
            'Écart-type': np.std(cluster_data, axis=0),
            'Moyenne': np.mean(cluster_data, axis=0)
        }
    return cluster_metrics

def interpret_metric_value(metric_value):
    """
    Interprète la valeur d'une métrique pour donner une indication sur sa qualité.

    Args:
        metric_value (float or pandas.DataFrame): La valeur de la métrique.

    Returns:
        str: Une indication sur la qualité de la métrique.
    """


    if isinstance(metric_value, pd.DataFrame):
        metric_value = metric_value.iloc[0, 0]  # Récupérer la valeur à la première ligne, première colonne
    if metric_value < 100:
        return "Excellent"

    elif metric_value < 500:
        return "Très bon"
    elif metric_value < 1000:

        return "Bon"
    elif metric_value < 2000:

        return "Moyen"
    else:
        return "À améliorer"

def evaluate_clustering_quality(data, max_k=25, max_iterations=100):
    """
    Évalue la qualité de chaque clustering pour différentes valeurs de k.

    Args:
        data (pandas.DataFrame): Les données d'entrée.
        max_k (int): La valeur maximale de k à évaluer.
        max_iterations (int): Nombre maximal d'itérations pour l'algorithme K-means.

    Returns:
        dict: Un dictionnaire contenant les métriques intra et inter pour chaque valeur de k.
    """


    clustering_quality = {}
    for k in range(1, max_k + 1):
        centroids, clusters = k_means(data, k, max_iterations=max_iterations)  # Passer max_iterations
        wcss = compute_wcss(data, centroids, clusters)
        bcss = compute_bcss(data, centroids, clusters)
        clustering_quality[k] = {'WCSS': wcss, 'BCSS': bcss}
    return clustering_quality


def suggest_optimal_k(clustering_quality):
    """
    Suggère un k optimal en fonction de la qualité de clustering.

    Args:
        clustering_quality (dict): Un dictionnaire contenant les métriques intra et inter pour chaque valeur de k.

    Returns:
        int: La valeur de k suggérée.
    """



    quality_difference = {k: clustering_quality[k]['BCSS'] - clustering_quality[k]['WCSS'] for k in clustering_quality}


    quality_difference_float = {k: float(v.iloc[0]) if isinstance(v, pd.Series) else float(v) for k, v in quality_difference.items()}


    suggested_k = max(quality_difference_float.items(), key=lambda x: x[1])[0]

    return suggested_k






def main():
    st.title("K-means Clustering")

    # Etape 1: Chargement des données
    st.header("Etape 1: Chargement des données")
    file_path = st.file_uploader("Charger un fichier CSV", type=['csv'])
    if file_path is not None:
        data = load_data(file_path)
        st.success("Les données ont été chargées avec succès.")
    else:
        st.warning("Veuillez charger un fichier CSV.")

    # Etape 2: Demande de k et du nombre de groupes
    st.header("Etape 2: Paramètres K-means")
    if file_path is not None:
        k = st.number_input("Entrez la valeur de k", min_value=1, max_value=len(data), value=2, step=1)
        max_iterations = st.number_input("Entrez le nombre maximal d'itérations", min_value=1, value=100, step=1)

        if st.button("Exécuter K-means"):
            centroids, clusters = k_means(data, k, max_iterations)
            st.success("K-means a été exécuté avec succès.")

            # Réduction de dimension avec PCA
            pca = PCA(n_components=2)  # Réduire à 2 dimensions pour la visualisation
            data_reduced = pca.fit_transform(data)

            # Ajout des données réduites au DataFrame
            data_reduced_df = pd.DataFrame(data_reduced, columns=['PC1', 'PC2'])


            st.session_state.centroids = centroids
            st.session_state.clusters = clusters

            # Etape 3: Affichage des résultats
            st.header("Etape 3: Résultats")
            st.write("Centroids:")
            st.write(pd.DataFrame(centroids, columns=data.columns))
            st.write("Clusters:")
            st.write(pd.DataFrame({"Data Point": np.arange(len(data)), "Cluster": clusters}))
            st.write("Écart-type de chaque cluster:")
            for i in range(k):
                st.write(f"Cluster {i+1}: {np.std(data[clusters == i], axis=0)}")

            # Etape 4: Affichage visuel des résultats avec données réduites
            st.header("Etape 4: Résultats visuels avec PCA")
            plot_data = pd.concat([data_reduced_df, pd.Series(clusters, name='Cluster')], axis=1)
            fig = px.scatter(plot_data, x='PC1', y='PC2', color='Cluster', title='K-means Clustering with PCA')
            st.plotly_chart(fig)

    # Etape 5: Prédiction du cluster pour un point donné
    st.header("Etape 5: Prédiction du cluster")
    centroids = st.session_state.get('centroids')
    if centroids is not None:
        point_values = []
        for i in range(data.shape[1]):
            value = st.number_input(f"Entrez la valeur de la caractéristique {i+1}", step=0.01)
            point_values.append(value)
        if st.button("Prédire le cluster"):
            cluster_prediction = predict_cluster(point_values, centroids)
            st.write(f"Le point est prédit appartenir au cluster {cluster_prediction + 1}.")

            # Etape 6: Affichage des métriques intra et inter-cluster
            clusters = st.session_state.get('clusters')
            if file_path is not None and centroids is not None and clusters is not None:
                st.header("Etape 6: Métriques intra et inter-cluster")
                wcss = compute_wcss(data, centroids, clusters)
                bcss = compute_bcss(data, centroids, clusters)

                # Extraire la valeur scalaire de la somme des carrés intra-cluster à partir du DataFrame
                wcss_value = wcss.iloc[0]

                st.write(
                    f"Somme des carrés intra-cluster (WCSS): {wcss_value} - {interpret_metric_value(wcss_value)}")
                st.write(f"Somme des carrés inter-cluster (BCSS): {bcss} - {interpret_metric_value(bcss)}")

                # Légende pour WCSS et BCSS
                st.write("Légende:")
                st.write("WCSS - Somme des carrés intra-cluster")
                st.write("BCSS - Somme des carrés inter-cluster")

                cluster_metrics = compute_cluster_metrics(data, centroids, clusters)
                for cluster, metrics in cluster_metrics.items():
                    st.subheader(cluster)
                    st.write("Nombre d'éléments:", metrics['Nombre d\'éléments'])
                    st.write("Écart-type:", metrics['Écart-type'])
                    st.write("Moyenne:", metrics['Moyenne'])

        # Etape 7: Analyse comparative pour différentes valeurs de k et suggestion de k optimal
        if file_path is not None:
            st.header("Etape 7: Analyse comparative de différentes valeurs de k")

            # Afficher un spinner pendant le calcul
            with st.spinner("Calcul en cours..."):
                clustering_quality = evaluate_clustering_quality(data)
                st.write("Qualité de clustering pour différentes valeurs de k:")
                st.write(pd.DataFrame(clustering_quality).T)

                suggested_k = suggest_optimal_k(clustering_quality)
                st.write(f"Un k optimal suggéré est: {suggested_k}")

if __name__ == "__main__":
    main()
