import streamlit as st
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Fonction pour charger les données à partir d'un fichier CSV
def charger_donnees_depuis_csv(chemin_fichier):
    return pd.read_csv(chemin_fichier)

# Fonction pour saisir manuellement les données
def saisie_manuelle():
    st.subheader("Saisie manuelle des données")
    
    # Demander le nombre de lignes et de colonnes
    num_rows = st.number_input("Nombre de lignes", min_value=1, value=1, step=1)
    num_cols = st.number_input("Nombre de colonnes", min_value=1, value=1, step=1)
    
    # Initialiser une liste vide pour stocker les données saisies
    data = []
    
    # Boucle pour saisir les données pour chaque ligne et chaque colonne
    for i in range(num_rows):
        row = []
        st.write(f"Entrez les données pour la ligne {i+1}:")
        for j in range(num_cols):
            value = st.number_input(f"Colonne {j+1}", key=f"value_{i}_{j}")
            row.append(value)
        data.append(row)
    
    # Créer un DataFrame à partir des données saisies
    df = pd.DataFrame(data, columns=[f"Colonne {j+1}" for j in range(num_cols)])
    
    return df

def distance(point1, point2):
    # Calcul de la distance euclidienne entre deux points
    return sum((x - y) ** 2 for x, y in zip(point1, point2)) ** 0.5

def kmeans(data, k, max_iterations=100):
    # Initialisation aléatoire des centroids
    centroids = random.sample(data, k)

    for _ in range(max_iterations):
        # Attribution des points au cluster le plus proche
        clusters = [[] for _ in range(k)]
        for point in data:
            # Calcul des distances entre le point et tous les centroids
            distances = [distance(point, centroid) for centroid in centroids]
            # Attribution du point au cluster avec le centroid le plus proche
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(point)

        # Mise à jour des centroids
        new_centroids = [[sum(dim) / len(cluster) for dim in zip(*cluster)] for cluster in clusters]

        # Vérifier la convergence
        if centroids == new_centroids:
            break

        centroids = new_centroids

    # Calcul des écarts-types
    std_devs = []
    for cluster in clusters:
        std_devs.append(np.std(cluster, axis=0))

    return clusters, centroids, std_devs
def predict_cluster(point, centroids):
    # Calculer la distance entre le point et chaque centroid
    distances = [distance(point, centroid) for centroid in centroids]
    # Trouver l'indice du centroid le plus proche
    closest_centroid_index = np.argmin(distances)
    return closest_centroid_index

# Définir l'interface utilisateur
def main():
    st.title("Chargement des données")
    
    # Widget pour choisir la méthode de chargement
    methode_chargement = st.radio("Méthode de chargement des données", ("Manuelle", "Fichier"))
    
    # Initialiser la variable donnees à None
    donnees = None
    
    # Charger les données en fonction de la méthode choisie
    if methode_chargement == "Manuelle":
        donnees = saisie_manuelle()
    else:
        fichier = st.file_uploader("Charger un fichier CSV", type=["csv"])
        if fichier is not None:
            donnees = charger_donnees_depuis_csv(fichier)
            if donnees is None:
                st.warning("Aucune donnée trouvée dans le fichier.")
        else:
            st.warning("Veuillez charger un fichier CSV.")
        
    num_k = st.number_input("Nombre de groupes [K]", min_value=1, value=1, step=1)


    # Afficher les données
    if donnees is not None:
        st.subheader("Données chargées :")
        st.write(donnees)

        # Appliquer K-means
        clusters, centroids, std_devs = kmeans(donnees.values.tolist(), num_k)
        
        # Afficher les résultats des centroïdes dans un tableau
        # Afficher les résultats des centroïdes dans un tableau
        st.subheader("Centroïdes et Écarts-types:")
        centroid_data = []
        for i, (centroid, std_dev) in enumerate(zip(centroids, std_devs)):
            centroid_label = f"Centroïde {i+1}"
            centroid_values = [f"{val:.2f}" for val in centroid]  # Convertir les valeurs en chaînes avec deux décimales
            std_dev_values = [f"{val:.2f}" for val in std_dev]  # Convertir les écarts-types en chaînes avec deux décimales
            centroid_data.append([centroid_label] + centroid_values + std_dev_values)

        # Créer un DataFrame pour le tableau
        columns = ["Centroïde Label"] + [f"Colonne {j+1}" for j in range(len(centroids[0]))] + [f"Écart-type {j+1}" for j in range(len(std_devs[0]))]
        centroid_df = pd.DataFrame(centroid_data, columns=columns)

        # Afficher le tableau
        st.table(centroid_df)

        

       # Créer la figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Afficher les clusters dans un scatterplot
        for i, cluster in enumerate(clusters):
            cluster_label = f"Cluster {i+1}"
            cluster_data = np.array(cluster)
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], label=cluster_label)

        ax.set_xlabel("Variable 1")
        ax.set_ylabel("Variable 2")
        ax.set_title("Résultat du clustering")
        ax.legend()
        ax.grid(True)

        # Afficher la figure avec Streamlit
        st.pyplot(fig)

        num_un = st.number_input("Point 1", min_value=0.0, value=0.0, step=0.1)
        num_deux = st.number_input("Point 2", min_value=0.0, value=0.0, step=0.1)



        # Exemple de prédiction pour un nouveau point
        new_point = [num_un,num_deux]  # Remplacez ces valeurs par celles de votre nouveau point

        # Prédire le groupe du nouveau point
        predicted_cluster = predict_cluster(new_point, centroids)
        st.write(f"Le nouveau point {new_point} est prédit appartenir au groupe {predicted_cluster + 1}.")
        
        # Prédire les étiquettes de cluster pour tous les points
        predicted_labels = [predict_cluster(point, centroids) for point in donnees.values.tolist()]

        # Calcul du coefficient de silhouette pour évaluer la cohérence intra-cluster et la séparation inter-cluster
        silhouette_avg = silhouette_score(donnees, predicted_labels)
        st.write(f"Score de silhouette moyen : {silhouette_avg}")
        # Plus le score de silhouette est proche de 1, meilleure est la cohésion intra-cluster et la séparation inter-cluster.

        # Calcul du score de Davies-Bouldin pour évaluer la séparation entre les clusters
        davies_bouldin = davies_bouldin_score(donnees, predicted_labels)
        st.write(f"Score de Davies-Bouldin : {davies_bouldin}")
        # Un score de Davies-Bouldin plus bas indique une meilleure séparation entre les clusters.

        
        # Tester différentes valeurs de K
        for k in range(2, 11):  # Tester de 2 à 10 clusters
            # Appliquer K-means
            clusters, centroids, std_devs = kmeans(donnees.values.tolist(), k)
            
            # Prédire les étiquettes de cluster pour tous les points
            predicted_labels = [predict_cluster(point, centroids) for point in donnees.values.tolist()]

            # Calcul du coefficient de silhouette
            silhouette_avg = silhouette_score(donnees, predicted_labels)
            
            # Calcul du score de Davies-Bouldin
            davies_bouldin = davies_bouldin_score(donnees, predicted_labels)
            
            # Afficher les résultats pour chaque valeur de K
            st.write(f"Pour K={k}:")
            st.write(f"   Score de silhouette moyen : {silhouette_avg}")
            st.write(f"   Score de Davies-Bouldin : {davies_bouldin}")

# Lancer l'application
if __name__ == "__main__":
    main()
