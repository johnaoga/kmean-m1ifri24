import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

def custom_kmeans(data, k, max_iterations=100):
    # Initialisation aléatoire des centroids
    centroids = data[np.random.choice(range(len(data)), k, replace=False)]

    for _ in range(max_iterations):
        # Attribution des points au cluster le plus proche
        labels = np.argmin(np.linalg.norm(data[:, None] - centroids, axis=2), axis=1)

        # Mise à jour des centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Vérifier la convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids


def custom_predict_cluster(new_point, centroids):
    # Calculer la distance de new_point à chaque centroïde
    distances = np.linalg.norm(new_point - centroids, axis=1)
    
    # Trouver l'indice du centroïde le plus proche (cluster)
    predicted_cluster = np.argmin(distances)
    
    return predicted_cluster

def main():
    st.title("K-means Clustering (implémentation personnalisée)")

    # Ajouter le composant pour le téléchargement du fichier CSV
    uploaded_file = st.file_uploader("Télécharger un fichier CSV", type=["csv"])

    if uploaded_file is not None:
        # Lire le fichier CSV téléchargé
        df = pd.read_csv(uploaded_file)

        colone_a = df.columns[0]
        colone_b = df.columns[1]
        data = df[[colone_a, colone_b]].values

        # Sélectionner le nombre de clusters
        num_clusters = st.slider("Choisissez le nombre de clusters :", min_value=2, max_value=10, value=3, step=1)

        # Appliquer K-means
        labels, centroids = custom_kmeans(data, num_clusters)

        # Créer un DataFrame avec les données, les centroids et les labels de cluster
        result_df = pd.DataFrame(data=data, columns=[colone_a, colone_b])
        result_df['Cluster Label'] = labels

        # Ajouter les centroids au DataFrame
        centroids_df = pd.DataFrame(data=centroids, columns=[colone_a, colone_b])
        centroids_df['Cluster Label'] = np.arange(num_clusters)  # Numéro de cluster pour les centroids

        # Concaténer les données et les centroids
        final_df = pd.concat([result_df, centroids_df])

        # Afficher le DataFrame dans Streamlit
        st.subheader("Résultats :")
        st.write(final_df)

        # Initialiser une liste pour stocker les écarts-types
        std_devs = []

        # Calculer l'écart-type pour chaque centroïde
        for i, centroid in enumerate(centroids):
            cluster_points = final_df[final_df['Cluster Label'] == i][[colone_a, colone_b]]
            std_dev = np.sqrt(np.mean((cluster_points - centroid) ** 2))
            std_devs.append(std_dev)

        # Afficher les résultats dans un tableau
        st.subheader("Résultats pour chaque centroïde et écart-type de chacun")
        result_df = pd.DataFrame({'Centroide': range(1, num_clusters + 1),
                                  'Coordonnée Age': centroids[:, 0],
                                  'Coordonnée Annual Income (k$)': centroids[:, 1],
                                  'Écart-type': std_devs})
        st.write(result_df)

        if st.button('Visualiser en deux dimensions'):

            # Visualiser les clusters en 2D
            fig, ax = plt.subplots(figsize=[10, 7])
            for i in range(num_clusters):
                cluster_data = final_df[final_df['Cluster Label'] == i]
                plt.scatter(cluster_data[colone_b], cluster_data[colone_a], label=f'Cluster {i+1}', s=100, alpha=0.7)
            plt.xlabel(colone_b)
            plt.ylabel(colone_a)
            plt.legend()
            plt.title('Visualisation des clusters')
            st.pyplot(fig)
        

        if st.button("Prédire le cluster d'un point"):
            
            # Entrer les valeurs pour le nouveau point
            st.subheader("Entrez les valeurs pour le nouveau point :")
            age = st.number_input("Âge :", min_value=0)
            income = st.number_input("Revenu annuel (k$) :", min_value=0)

            # Standardiser les valeurs du nouveau point
            new_point_std = np.array([age, income])  # Les valeurs de votre nouveau point
            predicted_cluster = custom_predict_cluster(new_point_std, centroids)

            st.subheader("Prédiction du groupe :")
            st.write(f"Le nouveau point est prédit comme appartenant au groupe {predicted_cluster}.")


if __name__ == "__main__":
    main()
