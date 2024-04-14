import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ajouter cette ligne pour désactiver l'avertissement
st.set_option('deprecation.showPyplotGlobalUse', False)

def assign_random_centroids(data, k):
    """
    Assigns random centroids to each of the k clusters.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        k (int): Number of clusters.
        
    Returns:
        centroids (np.array): Array containing the random centroids.
    """
    # Shuffle the data indices to get random centroids
    shuffled_indices = np.random.choice(data.shape[0], k, replace=False)
    
    # Select random data points as centroids
    centroids = data.iloc[shuffled_indices].values
    
    return centroids

def calculate_distances(data, centroids):
    """
    Calculates the distances of all observations from each of the k centroids.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        centroids (np.array): Array containing the centroids.
        
    Returns:
        distances (np.array): 2D array containing the distances of each observation from each centroid.
    """
    # Initialize an empty array to store distances
    distances = np.zeros((data.shape[0], centroids.shape[0]))

    # Calculate distances manually
    for i in range(data.shape[0]):
        for j in range(centroids.shape[0]):
            distances[i, j] = np.sqrt(np.sum((data.iloc[i].values - centroids[j])**2))
    
    return distances

def assign_clusters(data, centroids):
    """
    Assigns observations to the nearest centroid.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        centroids (np.array): Array containing the centroids.
        
    Returns:
        cluster_labels (np.array): Array containing the assigned cluster labels for each observation.
    """
    # Calculate distances
    distances = calculate_distances(data, centroids)
    
    # Assign each observation to the nearest centroid
    cluster_labels = np.argmin(distances, axis=1)
    
    return cluster_labels

def update_centroids(data, cluster_labels, k):
    """
    Updates the centroids by taking the mean of all observations in each cluster.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the data.
        cluster_labels (np.array): Array containing the assigned cluster labels for each observation.
        k (int): Number of clusters.
        
    Returns:
        centroids (np.array): Updated array containing the centroids.
    """
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_data = data[cluster_labels == i]
        centroids[i] = np.mean(cluster_data, axis=0)
    return centroids

# Titre de l'application Streamlit
st.title("Bienvenue sur mon application")

st.title("Algorithme K-Means avec Streamlit")

# Section pour importer les données CSV
st.subheader("Importer les données (vous y trouverez un exemple de dataset sample_data_set_of_kmeans_josue_agboton.csv)")
uploaded_file = st.file_uploader("Importer un fichier CSV", type=["csv"])

# Si un fichier est téléchargé, lire le fichier CSV
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Voici les premières lignes du DataFrame :")
    st.write(data.head())

    # Demander à l'utilisateur le nombre de clusters (k)
    k = st.number_input("Entrez le nombre de clusters (k)", min_value=2, max_value=10, value=2, step=1)

    # Assigner aléatoirement les centroïdes initiaux
    centroids = assign_random_centroids(data, k)

    # Afficher les centroïdes assignés
    st.subheader("Centroïdes assignés initialement :")
    st.write(centroids)

    # Algorithme K-Means
    max_iterations = 100  # Nombre maximal d'itérations
    for iteration in range(max_iterations):
        # Étape 3: Assigner les observations aux clusters les plus proches
        cluster_labels = assign_clusters(data, centroids)
        
        # Étape 4: Mettre à jour les centroïdes
        new_centroids = update_centroids(data.values, cluster_labels, k)
        
        # Vérifier si les centroïdes ont convergé
        if np.allclose(centroids, new_centroids):
            st.write(f"Les centroïdes ont convergé après {iteration+1} itérations.")
            break
        
        centroids = new_centroids

    # Afficher les centroïdes finaux
    st.subheader("Centroïdes finaux :")
    st.write(centroids)

    # Afficher les étiquettes de cluster attribuées
    st.subheader("Étiquettes de cluster attribuées :")
    st.write(cluster_labels)

    # Visualisation des clusters
    df_clusters = pd.DataFrame(data.values, columns=[f'Feature {i}' for i in range(data.shape[1])])
    df_clusters['Cluster'] = cluster_labels

    # Visualisation des clusters avec Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(k):
        cluster_data = df_clusters[df_clusters['Cluster'] == i]
        ax.scatter(cluster_data['Feature 0'], cluster_data['Feature 1'], label=f'Cluster {i}')
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='black', label='Centroids')
    ax.set_xlabel('Feature 0')
    ax.set_ylabel('Feature 1')
    ax.set_title('K-Means Clustering')
    ax.legend()
    st.pyplot(fig)
