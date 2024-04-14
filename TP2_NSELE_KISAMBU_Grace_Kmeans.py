import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Désactiver l'avertissement de l'utilisation de pyplot global
st.set_option('deprecation.showPyplotGlobalUse', False)

def generate_data(num_points, num_features):
    return pd.DataFrame(np.random.randn(num_points, num_features), columns=[f'Feature {i+1}' for i in range(num_features)])

def k_means_clustering(data, k):
    # Initialisation aléatoire des centroids
    centroids = data.sample(n=k, random_state=0)
    prev_centroids = centroids.copy()
    
    # Assigner chaque point au centroid le plus proche
    while True:
        distances = np.zeros((len(data), k))
        for i in range(k):
            distances[:, i] = np.linalg.norm(data.values - centroids.iloc[i].values, axis=1)
        labels = np.argmin(distances, axis=1)
        
        # Calculer les nouveaux centroids
        centroids = data.groupby(labels).mean()
        
        # Vérifier la convergence
        if prev_centroids.equals(centroids):
            break
        
        prev_centroids = centroids.copy()
    
    std_devs = [np.std(data[labels == i], axis=0) for i in range(k)]
    
    return centroids.values, std_devs, labels

def custom_pca(data, n_components):
    # Centrer les données
    
    # ceci est pour center les données 
    centered_data = data - np.mean(data, axis=0)
    
    # Calculer la matrice de covariance
    covariance_matrix = np.cov(centered_data, rowvar=False)
    
    # Calculer les valeurs propres et vecteurs propres
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    # Trier les vecteurs propres selon les valeurs propres décroissantes
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Sélectionner les n_components premiers vecteurs propres
    components = sorted_eigenvectors[:, :n_components]
    
    # Réduire la dimensionnalité
    reduced_data = np.dot(centered_data, components)
    
    return reduced_data

st.title('K-means Clustering with Streamlit')

num_points = st.slider('Number of Data Points:', min_value=10, max_value=1000, value=100, step=10)
num_features = st.slider('Number of Features:', min_value=2, max_value=10, value=2)

data = generate_data(num_points, num_features)

st.subheader('Generated Data:')
st.write(data)

k = st.slider('Number of Clusters (K):', min_value=2, max_value=10, value=3)

centroids, std_devs, labels = k_means_clustering(data, k)

st.subheader('Results:')
result_df = pd.DataFrame({'Centroid': [f'Centroid {i+1}' for i in range(k)],
                          'Coordinates': [centroid for centroid in centroids],
                          'Standard Deviation': [std for std in std_devs]})
st.write(result_df)

# Visualization
if num_features >= 2:
    reduced_data = custom_pca(data.values, 2)  # Réduction à 2 composantes principales
    reduced_centroids = custom_pca(centroids, 2)
    
    fig, ax = plt.subplots()
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k', 'purple', 'orange', 'brown']  # Ajout de couleurs pour 10 clusters
    for i in range(k):
        if np.sum(labels == i) > 0:
            ax.scatter(reduced_data[labels == i][:, 0], reduced_data[labels == i][:, 1], c=colors[i], label=f'Cluster {i+1}')
    ax.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], c='black', marker='x', label='Centroids')
    ax.legend()
    st.pyplot(fig)
else:
    st.error('Cannot visualize data with less than 2 features.')

# Prediction
point_to_predict = st.text_input('Enter a point to predict its cluster (comma-separated values):')
if point_to_predict:
    point = np.array([float(x.strip()) for x in point_to_predict.split(',')])
    point = point.reshape(1, -1)
    distances_to_centroids = np.sqrt(((centroids - point)**2).sum(axis=1))
    predictions = np.argmin(distances_to_centroids)
    st.write(f'The predicted cluster for the point {point_to_predict} is Cluster {predictions+1}.')


