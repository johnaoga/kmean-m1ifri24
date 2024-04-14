import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Fonction pour initialiser les centroïdes de manière aléatoire
def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], size=k, replace=False)
    selected_records = data[indices]
    return selected_records

# Fonction pour assigner les clusters
def assign_clusters(data, centroids):
    distances = np.zeros((len(centroids), len(data)))  # Initialiser le tableau des distances

    for i in range(len(centroids)):
        for j in range(len(data)):
            # Calculer la distance euclidienne entre le point de donnée et le centroide
            diff = data[j] - centroids[i]
            distances[i, j] = np.sqrt(np.sum(diff**2))

    # Retourner l'indice du centroïde le plus proche pour chaque point de données
    return np.argmin(distances, axis=0)

def recalculate_centroids(data, labels, k):
    # Initialisation du tableau des nouveaux centroïdes
    new_centroids = np.zeros((k, data.shape[1]))
    
    for i in range(k):
        # Initialisation de la liste pour stocker les points de données du centroïde actuel
        centroid_data = []
        for j in range(len(data)):
            if labels[j] == i:
                # Ajouter le point de données à la liste du centroïde
                centroid_data.append(data[j])
        
        if centroid_data:
            centroid_data = np.array(centroid_data)
            new_centroids[i] = np.mean(centroid_data, axis=0)
    return new_centroids

# Fonction pour calculer l'écart-type de chaque cluster
def calculate_std_dev(data, labels, k):
    std_devs = []
    for i in range(k):
        cluster_data = []
        for j in range(len(data)) :
            if labels[j] == i:
                cluster_data.append(data[j])
        if cluster_data :
            cluster_data = np.array(cluster_data)
            std_dev = cluster_data.std(axis=0).mean()  # Écart-type moyen des caractéristiques
            std_devs.append(std_dev)
    return std_devs

# Fonction pour calculer la Somme des carrés intra-classe (WCSS) afin d'évaluer la qualité intra des clusters
def calculate_wcss(data, labels, centroids, k):
    wcss = 0
    for i in range(k):
        cluster_data = []
        for j in range(len(data)) :
            if labels[j]  == i :
                cluster_data.append(data[j])
        if cluster_data :
            cluster_data=np.array(cluster_data)
            wcss += np.sum((cluster_data - centroids[i])**2)
    return wcss

def calculate_silhouette(data, labels, k):
    silhouette_values = []
    for i in range(len(data)):
        same_cluster= []
        for j in range(len(data)) :
            if i != j and labels[j] == labels[i] :
                same_cluster.append(data[j])
        if same_cluster :
            same_cluster= np.array(same_cluster)
            # Calculer a : la distance moyenne intra-cluster
            a = np.mean(np.sqrt(np.sum((same_cluster - data[i])**2, axis=1)))

        # Calculer b : la distance moyenne au cluster voisin le plus proche
        b = np.inf
        for j in range(k):
            if j != labels[i]:
                other_cluster = []
                for f in range(len(data)) :
                    if labels[f] == j :
                        other_cluster.append(data[f])
                other_cluster = np.array(other_cluster)
                b = min(b, np.mean(np.sqrt(np.sum((other_cluster - data[i])**2, axis=1))))

        # Calculer la silhouette pour le point de données
        silhouette = (b - a) / max(a, b) if max(a, b) > 0 else 0
        silhouette_values.append(silhouette)

    # Retourner la silhouette moyenne pour tous les points de données
    return np.mean(silhouette_values)

# Fonction pour l'algorithme K-means
def kmeans(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)
    for i in range(max_iters):
        old_centroids = centroids
        labels = assign_clusters(data, centroids)
        centroids = recalculate_centroids(data, labels, k)
        if np.all(centroids == old_centroids):
            break
    std_devs = calculate_std_dev(data, labels, k)
    wcss = calculate_wcss(data, labels, centroids, k)
    silhouette = calculate_silhouette(data, labels, k)
    return centroids, labels, std_devs, wcss, silhouette

# Fonction pour afficher les résultats avec les données originales (<=3 dimensions)
def display_clustering_results_2d(data, labels, centroids, k):
    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    for i in range(k):
        cluster_data = data[labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], label=f'Cluster {i}')

        plt.scatter(centroids[i, 0], centroids[i, 1], c=colors[i], marker='x', s=100)

    plt.title('Résultats du clustering')
    plt.xlabel('Caractéristique 1')
    plt.ylabel('Caractéristique 2')
    plt.legend()
    st.pyplot(plt)

# Fonction pour afficher les résultats avec l'ACP (>3 dimensions)
def display_clustering_results_pca(data, labels, centroids, k):
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)

    plt.figure(figsize=(8, 6))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    for i in range(k):
        cluster_data = data_2d[labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c=colors[i], label=f'Cluster {i}')

        centroids_2d = pca.transform(centroids[i].reshape(1, -1))
        plt.scatter(centroids_2d[0, 0], centroids_2d[0, 1], c=colors[i], marker='x', s=100)

    plt.title('Résultats du clustering (ACP)')
    plt.xlabel('Composante Principale 1')
    plt.ylabel('Composante Principale 2')
    plt.legend()
    st.pyplot(plt)

# Fonction pour prédire le cluster d'un nouveau point
def predict_cluster(new_data_point, centroids):
    distances = np.sqrt(np.sum((centroids - new_data_point) ** 2, axis=1))
    return np.argmin(distances)

# Interface Streamlit
st.title("K-means personnalisé sans scikit-learn")

# Chargement des données
uploaded_file = st.file_uploader("Choisissez un fichier CSV")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)
    k = st.number_input("Entrez le nombre de clusters (K)", min_value=2, value=3, step=1)
    
    if st.button("Effectuer K-means"):
        data_values = data.values
        centroids, labels, std_devs, wcss, silhouette = kmeans(data_values, k)
        st.write("Centroïdes:")
        st.write(centroids)
        st.write("Écart-type de chaque cluster:")
        st.write(std_devs)

        
        # Afficher les résultats du clustering sous forme visuelle
        if data_values.shape[1] <= 3:  # Si le nombre de dimensions est <= 3
            display_clustering_results_2d(data_values, labels, centroids, k)
        else:
            display_clustering_results_pca(data_values, labels, centroids, k)


    if st.button("Trouver le K optimal"):
        data_values = data.values
        wcss_values = []
        silhouettes_values = []
        K_range = range(2, 10)  # Par exemple, tester de 2 à 9 clusters
        for k in K_range:
            centroids, labels, std_devs, wcss, silhouette = kmeans(data_values, k)
            wcss_values.append(wcss)
            silhouettes_values.append(silhouette)

        # Afficher les graphiques de WCSS et Silhouette avec les étiquettes appropriées
        wcss_data = {f"K={k}": wcss for k, wcss in zip(K_range, wcss_values)}
        st.line_chart(wcss_data)

        silhouette_data = {f"K={k}": silhouette for k, silhouette in zip(K_range, silhouettes_values)}
        st.line_chart(silhouette_data)

            # Trouver le K optimal en utilisant la méthode du coude pour WCSS
        wcss_gradients = np.gradient(wcss_values)
        k_optimal_wcss = K_range[np.argmin(wcss_gradients[1:])] + 2  # +2 car np.gradient décale les valeurs

        # Trouver le K optimal en utilisant la silhouette moyenne la plus élevée
        k_optimal_silhouette = K_range[np.argmax(silhouettes_values)]

        st.write(f"Le K optimal suggéré en utilisant WCSS (méthode du coude) est: {k_optimal_wcss}")
        st.write(f"Le K optimal suggéré en utilisant la silhouette moyenne est: {k_optimal_silhouette}")

    with st.form("prediction_form"):
        st.write("Prédire le cluster pour un point")
        data_values = data.values
        centroids, labels, std_devs, wcss, bss = kmeans(data_values, k)

        # L'utilisateur doit fournir les valeurs pour chaque caractéristique
        new_point_data = []
        for i in range(data_values.shape[1]):
            feature_value = st.number_input(f"Valeur de la caractéristique {i}", value=0.000, key=f"feature_{i}")
            new_point_data.append(feature_value)

        submitted = st.form_submit_button("Prédire le cluster")

        if submitted:
            new_point_data = np.array(new_point_data).reshape(1, -1)
            predicted_cluster = predict_cluster(new_point_data, centroids)
            st.write(f"Ce point appartient au cluster: {predicted_cluster}")


