import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def kmeans(data, k, max_iterations=100):
    # Initialisation aléatoire des centroids
    centroids = data[np.random.choice(range(len(data)), k, replace=False)]

    for _ in range(max_iterations):
        # Attribution des points au cluster le plus proche
        distances = np.linalg.norm(data[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Mise à jour des centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Vérifier la convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

def predict_cluster(new_point, centroids):
    # Calculer la distance de new_point à chaque centroïde
    distances = np.linalg.norm(new_point - centroids, axis=1)
    
    # Trouver l'indice du centroïde le plus proche (cluster)
    predicted_cluster = np.argmin(distances)
    
    return predicted_cluster



def main():
    st.title("Implémentation de l'algorithme K-means ")

    # Ajouter le composant pour le téléchargement du fichier CSV
    uploaded_file = st.file_uploader("Télécharger un fichier CSV", type=["csv"])

    if uploaded_file is not None:
         # Lire le fichier CSV téléchargé
        df = pd.read_csv(uploaded_file)

        # Afficher les données
        st.subheader("Aperçu des données :")
        st.write(df)

        # Sélection du nombre de colonnes à utiliser
        num_cols = st.number_input("Nombre de colonnes à utiliser :", min_value=1, max_value=len(df.columns), value=2, step=1)

        # Sélection des colonnes à utiliser
        selected_cols = []
        for i in range(num_cols):
            selected_col = st.selectbox(f"Colonne {i+1} :", df.columns)
            selected_cols.append(selected_col)

        # Sélectionner les données à utiliser
        data = df[selected_cols].values

        # Sélectionner le nombre de clusters
        num_clusters = st.slider("Choisissez le nombre de clusters :", min_value=2, max_value=10, value=3, step=1)

        
        # Appliquer K-means
        labels, centroids = kmeans(data, num_clusters)

        # Remplacer les numéros de cluster par "Cluster 1", "Cluster 2", etc.
        cluster_labels = [f"Cluster {i+1}" for i in range(num_clusters)]
        labeled_labels = [cluster_labels[label] for label in labels]

        # Créer un DataFrame avec les données, les centroids et les labels de cluster
        result_df = pd.DataFrame(data=data, columns=selected_cols)
        
        result_df['Cluster'] = labeled_labels 
        result_df[""] = labels

        # Afficher les résultats
        st.subheader("Résultats :")
        st.write(result_df, index=False)
    

        # Visualisation des clusters
        if st.button('Visualiser en deux dimensions'):
            fig, ax = plt.subplots(figsize=[10, 7])
            for i in range(num_clusters):
                cluster_data = result_df[result_df[''] == i]
                plt.scatter(cluster_data[selected_cols[0]], cluster_data[selected_cols[1]], label=f'Cluster {i+1}', s=100, alpha=0.7)
            plt.xlabel(selected_cols[0])
            plt.ylabel(selected_cols[1])
            plt.legend()
            plt.title('Visualisation des clusters')
            st.pyplot(fig)
        
        
        # Initialiser une liste pour stocker les écarts-types
        centroids_ecart = []

        # Calculer l'écart-type pour chaque centroïde
        for i, centroid in enumerate(centroids):
            cluster_points = result_df[result_df[''] == i][selected_cols]  # Correction ici
            centroid_ecart = np.sqrt(np.mean((cluster_points - centroid) ** 2))
            centroids_ecart.append(centroid_ecart)

        # Afficher les résultats dans un tableau
        st.subheader("Résultats pour chaque centroïde et écart-type de chacun")
        result_df = pd.DataFrame({'Centroide': range(1, num_clusters + 1),
                                'Colone 1': centroids[:, 0],
                                'Colone 2': centroids[:, 1],
                                'Écart-type': centroids_ecart})
        st.write(result_df)

        # Prédiction du cluster pour un nouveau point
        st.subheader("Entrez les valeurs pour le nouveau point :")
        var1 = st.number_input("Valeur 1 :", min_value=0 )
        var2 = st.number_input("Valeur 2 :", min_value=0 )
        new_point = np.array([var1, var2])  # Nouveau point

        # Prédiction du cluster
        predicted_cluster = predict_cluster(new_point, centroids)
        st.write(f"Le nouveau point est prédit comme appartenant au groupe {predicted_cluster + 1}.")


        if st.button("Voir proposition du K optimal"):

            st.write("Ici nous avons utilisé la methode de courde afin de suggérer un k optimal pouvant nous permetre d'avoir un bon clustering à la fin ")

            # Définir la plage de clusters à tester
            cluster_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]
            inertie = []

            # Calculer l'inertie pour différentes valeurs de clusters
            for c in cluster_range:
                # Appliquer K-means
                _, centroids = kmeans(data, c)
                
                # Calculer les distances des échantillons à leur centroïde le plus proche
                distances = np.min(np.linalg.norm(data[:, None] - centroids, axis=2), axis=1)
                
                # Calculer l'inertie comme la somme des carrés des distances
                inertie.append(np.sum(distances ** 2))

            # Afficher le graphique d'inertie
            st.subheader("Inertie pour différentes valeurs de clusters")
            st.line_chart(pd.DataFrame({'Cluster Range': cluster_range, 'Inertie': inertie}).set_index('Cluster Range'))

        
        

        # Appliquer K-means
        labels, centroids = kmeans(data, num_clusters)

        # Calculer le score de silhouette
        silhouette_avg = silhouette_score(data, labels)

        # Afficher le score de silhouette
        st.subheader("Score de silhouette (Métrique d'évaluation) :")
        st.write("Silhouette score : C'est une mesure de la cohésion intra-cluster et de la séparation inter-cluster. Il varie de -1 à 1, où une valeur proche de 1 indique que les échantillons sont bien séparés, une valeur proche de 0 indique un chevauchement entre les clusters, et une valeur proche de -1 indique que les échantillons sont mal attribués.")
        st.write("Dans notre cas présent cette valeur est : ", silhouette_avg)





if __name__ == "__main__":
    main()
