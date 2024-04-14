
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.spatial.distance import cdist

# Définir l'algorithme K-means
class KMeans:
    def __init__(self, num_clusters, max_iterations=100):
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.centroids = None
        self.std_deviation = None

    def fit(self, X):
        # Initialisation aléatoire des centroïdes
        self.centroids = X[np.random.choice(X.shape[0], self.num_clusters, replace=False)]

        for _ in range(self.max_iterations):
            # Assigner chaque point au cluster le plus proche
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)

            # Mettre à jour les centroïdes
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.num_clusters)])

            # Vérifier la convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids
            
        # Calculer l'écart-type pour chaque centroïde
        self.std_deviation = np.array([np.std(X[labels == k], axis=0) for k in range(self.num_clusters)])

        # Calculer l'inertie manuellement
        inertia = 0
        for k in range(self.num_clusters):
            cluster_points = X[labels == k]
            inertia += ((cluster_points - self.centroids[k])**2).sum()

        self.inertia_ = inertia
        
        return labels
    
    def predict_cluster(self, new_point):
        distances = np.sqrt(((new_point - self.centroids[:, np.newaxis])**2).sum(axis=2))
        cluster = np.argmin(distances)
        return cluster


# Charger les données depuis le fichier CSV
data = pd.read_csv('data.csv')
X = data.values




# Réduction de dimensions avec PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


def main():
    st.title('Résultats de l\'algorithme K-means')
    
    st.write('Données chargées:')
    st.write(data)

    # Slider pour choisir le nombre de clusters (K)
    k = st.slider('Nombre de clusters (K)', min_value=1, max_value=10, value=3)
    num_clusters = k  # Nombre de clusters à calculer
    labels = KMeans(num_clusters=num_clusters).fit(X)

    # Calcul de l'inertie pour différents nombres de clusters
    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(num_clusters=k)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    # Affichage du graphique de l'inertie en fonction du nombre de clusters
    fig = px.line(x=range(1, 11), y=inertias, title='Méthode du coude')
    fig.update_xaxes(title='Nombre de clusters (K)')
    fig.update_yaxes(title='Inertie')
    st.plotly_chart(fig)
    
    # Affichage des résultats pour chaque centroïde dans un tableau
    if st.checkbox('Afficher les résultats pour chaque centroïde'):
        centroids_df = pd.DataFrame(kmeans.centroids, columns=data.columns)
        std_deviation_df = pd.DataFrame(kmeans.std_deviation, columns=data.columns, index=[f'Centroïde {i+1}' for i in range(k)])
        st.subheader('Centroïdes')
        st.write(centroids_df)
        st.subheader('Écart-type de chaque centroïde')
        st.write(std_deviation_df)
        
    # Visualisation des clusters avec PCA
    if st.checkbox('Visualiser les clusters avec PCA'):
        clusters_df = pd.DataFrame({'PCA1': X_pca[:, 0], 'PCA2': X_pca[:, 1], 'Cluster': labels})  # Utiliser labels retourné par fit
        fig_pca = px.scatter(clusters_df, x='PCA1', y='PCA2', color='Cluster', title='Clusters avec PCA')
        st.plotly_chart(fig_pca)
        
    # Prédiction du groupe d'un point
    st.subheader('Prédiction du groupe d\'un point')
    new_point_input = st.text_input('Entrer les valeurs du nouveau point (séparées par des virgules)')
    if st.button('Prédire le groupe'):
        try:
            new_point_values = np.array([float(x.strip()) for x in new_point_input.split(',')])
            if len(new_point_values) != len(data.columns):
                st.error('Le nombre de valeurs doit correspondre au nombre de colonnes dans les données.')
            else:
                kmeans_model = KMeans(num_clusters=num_clusters)
                kmeans_model.fit(X)
                predicted_cluster = kmeans_model.predict_cluster(new_point_values)
                st.success(f'Le nouveau point appartient au cluster {predicted_cluster + 1}.')
        except ValueError:
            st.error('Veuillez entrer des valeurs numériques valides.')
            
    # Calculer les métriques pour évaluer les clusters
    cluster_metrics = {}
    silhouette = silhouette_score(X, labels)
    db_index = davies_bouldin_score(X, labels)
    cluster_metrics['Silhouette Score'] = silhouette
    cluster_metrics['Davies-Bouldin Index'] = db_index

    # Calculer l'écart-type des distances intra-cluster
    intra_cluster_distances = cdist(X, kmeans.centroids[labels])
    std_intra_cluster = np.std(intra_cluster_distances)

    cluster_metrics['Écart-type distances intra-cluster'] = std_intra_cluster

    # Afficher les métriques dans l'interface utilisateur
    st.subheader('Évaluation des clusters')
    st.write(cluster_metrics)

if __name__ == '__main__':
    main()
