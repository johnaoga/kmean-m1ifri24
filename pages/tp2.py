import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import matplotlib.pyplot as plt

# Désactiver le warning PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Chargement des données depuis le fichier CSV
@st.cache_data
def load_data(filename):
    data = pd.read_csv(filename)
    # Sélectionner uniquement les colonnes numériques
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    # Remplacer les valeurs manquantes uniquement dans les colonnes numériques
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())
    return data

# Fonction pour lancer l'algorithme K-means
@st.cache_data
def run_kmeans(data, k):
    # Sélectionner uniquement les colonnes numériques
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    # Remplacer les valeurs manquantes par la moyenne des colonnes numériques
    data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].mean())

    # Initialiser et exécuter l'algorithme K-means
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(data[numerical_columns])

    # Récupérer les centroids et les labels
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    return centroids, labels

# Affichage des résultats dans un tableau
def display_results(centroids, labels, data):
    # Sélectionner uniquement les colonnes numériques
    numerical_columns = data.select_dtypes(include=[np.number]).columns
    
    # Créer un DataFrame avec les colonnes numériques uniquement
    centroids_df = pd.DataFrame(centroids, columns=numerical_columns)
    
    # Afficher les centroids
    st.write(centroids_df)

# Affichage visuel des clusters
def visualize_clusters(data, labels):
    # Sélectionner uniquement les caractéristiques numériques
    numerical_data = data.select_dtypes(include=['number'])
    
    # Gérer les valeurs manquantes si nécessaire
    numerical_data.fillna(0, inplace=True)  # Remplacer les valeurs manquantes par 0
    
    # Réduction de dimensionnalité avec PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(numerical_data)
    
    # Tracer les clusters dans un nuage de points
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title('Visualisation des clusters')
    plt.xlabel('Composante principale 1')
    plt.ylabel('Composante principale 2')
    plt.colorbar(label='Cluster')
    st.pyplot()
    

# Prédiction du groupe d'un point en fonction des valeurs fournies
def predict_cluster(centroids, point):
    distances = [np.linalg.norm(point - centroid) for centroid in centroids]
    return np.argmin(distances)

# Évaluation de la qualité des clusters
def evaluate_clusters(data, labels):
    # Sélectionner uniquement les caractéristiques numériques
    numerical_data = data.select_dtypes(include=['number'])
    
    # Calculer la métrique de silhouette
    silhouette_score = metrics.silhouette_score(numerical_data, labels)

# Interface utilisateur avec Streamlit
def main():
    st.title('K-means Clustering for Coffee Quality')
    st.sidebar.title('Options')

    # Initialisation de la variable data
    data = None

    # Option pour choisir entre le fichier par défaut et un fichier personnalisé
    data_option = st.sidebar.radio("Choose data option:", ("Default Dataset", "Custom Dataset"))

    if data_option == "Custom Dataset":
        # Télécharger un fichier CSV personnalisé
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.sidebar.write("Custom dataset loaded successfully!")
    else:
        # Charger le jeu de données par défaut
        data = load_data("Coffee_Qlty.csv")
        st.sidebar.write("Default dataset loaded successfully!")

    if data is not None:
        # Afficher les données
        st.sidebar.write(data)

        # Demander le nombre de clusters K
        k = st.sidebar.slider('Select K', min_value=2, max_value=10, value=3)

        # Exécuter l'algorithme K-means
        centroids, labels = run_kmeans(data, k)

        # Affichage des résultats
        st.header('Results')
        display_results(centroids, labels, data)

        # Visualisation des clusters (bonus)
        if st.sidebar.checkbox('Visualize Clusters'):
            visualize_clusters(data, labels)

        # Prédiction du groupe d'un point (bonus)
        if st.sidebar.checkbox('Predict Cluster for a Point'):
            point = st.sidebar.text_input('Enter values for a point (comma-separated)')
            if point:
                point = np.array([float(val) for val in point.split(',')])
                cluster = predict_cluster(centroids, point)
                st.sidebar.write(f'The point belongs to Cluster {cluster}')

        # Évaluation de la qualité des clusters
        st.sidebar.subheader('Cluster Quality Evaluation')
        evaluate_clusters(data, labels)

        # Proposer une analyse pour différentes valeurs de K (bonus)
        if st.sidebar.checkbox('Analyze Different K Values'):
            st.write("Analyzing different values of K...")
            # Ici, vous pouvez ajouter votre analyse pour différentes valeurs de K

if __name__ == "__main__":
    main()
