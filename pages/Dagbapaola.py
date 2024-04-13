import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Setting up the title and header
st.title("TP2: Clustering k means")
st.header('👨🏾‍💻Travaux Pratique')

# Subheader
st.subheader('Données à utiliser')

# Affichage de l'uploader dans l'application Streamlit
file = st.file_uploader("Importer vos données ici", type=["csv"])

# Vérification si un fichier a été téléchargé
if file is not None:
    # Traitement du fichier téléchargé (par exemple, afficher les 5 premières lignes d'un DataFrame Pandas)
    df = pd.read_csv(file)
    
    # Vérification si le DataFrame contient au moins deux colonnes numériques
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) >= 2:
        # Sélectionner automatiquement les deux premières colonnes numériques du DataFrame
        dfa = df[num_cols[:2]]
        
        # Afficher le DataFrame dans une table (optionnel)
        st.dataframe(dfa)

        # Titre du graphique
        st.header('Graphique des données')
        # Créer le graphique avec Matplotlib
        fig, ax = plt.subplots()
        ax.scatter(dfa.iloc[:, 0], dfa.iloc[:, 1])
        ax.set_xlabel(dfa.columns[0])  # Utiliser le nom de la première colonne
        ax.set_ylabel(dfa.columns[1])  # Utiliser le nom de la deuxième colonne
        ax.set_title('Représentation des données')

        # Afficher le graphique dans Streamlit
        st.pyplot(fig)
        
        # Demander le nombre de groupes à l'utilisateur
        k = st.number_input("Saisissez le nombre de groupes (k) à faire", min_value=0, value=1, step=1)
        
        # Transformer k en int
        k = int(k)
        st.write(f"Vous avez saisi : **{k}**")

        # Initialiser le nombre d'itérations
        iterations = 200

        # Fonction pour initialiser les centroides
        def initialiser_centroides(X, k):
            indices = np.random.choice(X.index, k, replace=False)
            centroides = X.loc[indices]
            return centroides

        # Fonction pour assigner les clusters
        def assigner_clusters(X, centroides):
            clusters = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                point = X.iloc[i].values.reshape(1, -1)
                distances = np.linalg.norm(point - centroides, axis=1)
                clusters[i] = np.argmin(distances)
            return clusters

        # Fonction pour calculer les nouveaux centroides
        def calculer_centroides(X, clusters, k):
            centroides = np.zeros((k, X.shape[1]))
            for i in range(k):
                centroides[i] = np.mean(X[clusters == i], axis=0)
            return centroides

        # Fonction K-means
        def k_means(X, k, iterations):
            centroides = initialiser_centroides(X, k)
            for _ in range(iterations):
                clusters = assigner_clusters(X, centroides)
                nouveaux_centroides = calculer_centroides(X, clusters, k)
                if np.array_equal(centroides, nouveaux_centroides):
                    break
                centroides = nouveaux_centroides
            return clusters, centroides

        # Afficher les résultats de l'algorithme K-means
        clusters, centroides = k_means(dfa, k, iterations)
        dfac = dfa.assign(clusters=clusters)
        centroides = pd.DataFrame(centroides, columns=num_cols[:2])
        st.write("Centroides obtenues:")
        st.write(centroides)
        # Calculer la distance entre chaque point de données et le centroïde de son cluster
        dfac["distance_centroide"] = np.sqrt(
        (dfac.iloc[:, 0] - dfac.groupby("clusters").transform("mean").iloc[:, 0])**2 +
        (dfac.iloc[:, 1] - dfac.groupby("clusters").transform("mean").iloc[:, 1])**2)

        # Calculer l'écart-type des distances calculer
        ecart_types_centroide = dfac.groupby("clusters")["distance_centroide"].std()
        st.dataframe(dfac)
        # Afficher les résultats dans un tableau
        resultats_ecart_types = pd.DataFrame({"clusters": ecart_types_centroide.index, "ecart_type_centroide": ecart_types_centroide.values})
        centroidess = pd.concat([centroides, resultats_ecart_types], axis=1)
        #affiche les nouveaux centroides obtenue, leurs eécarts type calculer et leurs clusters
        st.write("Centroides obtenues:")
        st.write(centroidess)
        # Défini les couleurs pour chaque cluster
        colors = {0: 'red', 1: 'blue', 2:'yellow', 3:'green' ,4:'pink' ,5:'purple'}

        # Créer le graphique avec Matplotlib
        fig, ax = plt.subplots()

        # Boucle pour parcourir chaque cluster
        for cluster, color in colors.items():
            cluster_data = dfac[dfac['clusters'] == cluster]
            centroid_data = centroidess[centroidess['clusters'] == cluster]  # Récupérer les centroides du cluster
            
            for idx, point in cluster_data.iterrows():
                # Marquer les points de cluster_data en cercle ('o')
                marker = 'o'
                # Trace le point sur le graphe avec la couleur spécifique du cluster
                ax.scatter(point[dfac.columns[0]], point[dfac.columns[1]], color=color, label=f'Cluster {cluster}', marker=marker)
            
            # Marquer les centroides du cluster en carré ('s') avec la même couleur que les points du cluster
            ax.scatter(centroid_data[dfac.columns[0]], centroid_data[dfac.columns[1]], color=color, label=f'Centroides {cluster}', marker='s')

        # Affiche les labels et le titre
        ax.set_xlabel(dfac.columns[0])
        ax.set_ylabel(dfac.columns[1])
        ax.set_title('Représentation des résultats du k-means dans un graphe')

        # Affiche le graphique 
        st.pyplot(fig)

        # Affiche les informations
        st.info("Les couleurs associées à chaque groupe de clusters dans nos données : Groupe 0 : Rouge , Groupe 1 : Bleu  , Groupe 2 : Jaune ,Groupe 3 : Vert , Groupe 4 : Rose , Groupe 5 : Violet ")
        st.info("⚠️Les centroides sont représentés en carrés.")
        # Fonction pour prédire le groupe ou le cluster d'un nouveau point
        def prediction(new_point, centroides):
            distances = np.sqrt(np.sum((centroides - new_point)**2, axis=1))
            predicted_cluster = np.argmin(distances)
            return predicted_cluster
        # Demander à l'utilisateur d'entrer les caractéristiques du nouveau point
        #demander age
        age = st.number_input("Saisissez la pemière information:", 
        min_value=0.0, max_value=90.0, value=5.0, step=1.0)
        #mettre age en float
        age = float(age)
        #demander revenu annuel
        revenu = st.number_input("Saisissez la deuxieme information:", 
        min_value=0.0, max_value=100.0, value=20.0, step=2.0)
        #mettre revenu en float
        revenu = float(revenu)
        #créer le nouveau point qui prend mes variables age et revenu
        new_point = np.array([age, revenu])
        # Prédiction du groupe pour le nouveau point
        predicted_cluster = prediction(new_point, centroides)
        #si on appuie sur le boutton prediction
        if st.button('Prédiction :'):
            #affiche la prediction
            st.write("Cluster ou groupe prédit pour le nouveau point :", predicted_cluster)
        st.info("Pour évaluer la qualité intra et inter de chaque cluster dans un algorithme de clustering comme K-means, on peut utiliser des métriques telles que l'inertie intra-cluster : Une faible inertie intra-cluster indique que les points à l'intérieur d'un cluster sont proches les uns des autres, ce qui souhaitable et la distance inter-cluster : Une grande distance inter-cluster suggère que les clusters sont bien séparés les uns des autres. Ces métriques aident à déterminer à quel point les données sont similaires à l'intérieur d'un cluster (inertie intra-cluster) et à quel point les clusters sont séparés les uns des autres (distance inter-cluster).")
        st.write("Pour évaluer la qualité intra de chaque cluster, nous allons utiliser l'inertie intra-cluster : C'est la somme des carrés des distances entre chaque point de données et le centroïde de son cluster, voilà les résultats que nous trouvons:")
        #calcule la somme des carrés des distances qu'on avait deja dans distance centroides
        inertie_intra_cluster = dfac.groupby("clusters")["distance_centroide"].apply(lambda x: np.sum(x ** 2))
        #Affiche les résultats du calcul de l'inertie 
        st.write(inertie_intra_cluster)
        # Création de la Series pour le graphique
        barchat = pd.Series(inertie_intra_cluster.values, index=inertie_intra_cluster.index, name='distance_centroide')
        bar_colors = [colors.get(idx, 'gray') for idx in barchat.index]

        # Création du graphique Matplotlib avec les couleurs spécifiées
        fig, ax = plt.subplots()
        barchat.plot(kind='bar', color=bar_colors, ax=ax)
        ax.set_xlabel('Clusters')
        ax.set_ylabel('Distance Centroïde')
        ax.set_title('Inertie Intra-Cluster par Cluster')
        plt.xticks(rotation=0)
        # Affichage du graphique avec Streamlit
        st.pyplot(fig)
        st.write("Pour évaluer la qualité inter de chaque cluster, nous allons utiliser la distance inter-clusters, c'est la distance entre les centroïdes de chaque paire de clusters, voilà les résultats que nous trouvons:")
        # Fonction pour calculer la distance euclidienne entre deux points
        def distance_euclidienne(point1, point2):
            return np.sqrt(np.sum((point1 - point2) ** 2))

        # Calcul de la distance inter-cluster
        distances_inter_cluster = pd.DataFrame(index=centroides.index, columns=centroides.index)
        for i in centroides.index:
            for j in centroides.index:
                #calcule la distance euclidienne de chaque paire de centroides
                distances_inter_cluster.loc[i, j] = distance_euclidienne(centroides.loc[i], centroides.loc[j])
        # Affichage des distances inter-cluster
        st.write(distances_inter_cluster)
        #afficher information
        st.info(" Une distance inter-cluster plus élevée, associée à une faible distance intra-cluster (distance entre les points d'un même cluster), indique généralement une bonne qualité de clustering.")
        st.write("Proposons une analyse qui montre la qualité intra et inter de chaque cluster pour des valeurs de K différent et suggérer un K à l’utilisateur.")
        st.write("Prenons k une liste de valeur de 1 à 6:")# Liste des valeurs de K à tester 
        k_valeur = [1, 2, 3, 4, 5, 6]
        # Initialisation des listes pour stocker les métriques 
        inertie_valeur = []
        # Calcul des métriques pour chaque valeur de K
        for k in k_valeur:
            # Clustering K-means
            clusters, centroids = k_means(dfa, k, iterations)
            dfak= dfa.assign(clusters=clusters)
            # Calculer la distance entre chaque point de données et le centroïde de son cluster
            dfak["distance_centroide"] = np.sqrt(
            (dfak.iloc[:, 0] - dfak.groupby("clusters").transform("mean").iloc[:, 0])**2 +
            (dfak.iloc[:, 1] - dfak.groupby("clusters").transform("mean").iloc[:, 1])**2)

            inertia = dfak.groupby("clusters")["distance_centroide"].apply(lambda x: np.sum(x ** 2))   
            # Calcul de l'inertie intra-cluster (somme des carrés des distances)
            inertie_valeur.append(inertia)
            #st.write(inertie_valeur)
            derniere_liste = inertie_valeur[-1]
            st.write(derniere_liste)
            # Dernière liste d'inerties contenue dans inertie_valeur
            derniere_liste = inertie_valeur[-1]

            # Nombre de clusters (K) pour la dernière liste
            k_derniere_liste = derniere_liste.shape[0]

            # Création du graphique de l'inertie en fonction de K
            fig, ax = plt.subplots()
            ax.plot(range(1, k_derniere_liste + 1), derniere_liste, marker='o')
            ax.set_xlabel('Nombre de clusters (K)')
            ax.set_ylabel('Inertie intra-cluster')
            ax.set_title('Méthode du coude pour choisir K')

            # Affichage du graphique avec Streamlit
            st.pyplot(fig)

        st.info("Concentez-vous sur la courbe qui descend généralement à mesure que k augmente et ensuite identifiez le point du coude dans le graphique. C'est le point où l'inertie intra-cluster commence à diminuer de manière plus lente après avoir diminué rapidement. C'est souvent le meilleur choix pour k, cette méthode s'appelle la méthode de coude")




    else:
        st.warning("Le fichier doit contenir au moins deux colonnes numériques pour effectuer le clustering k-means.")
else:
    st.info("Veuillez importer un fichier CSV pour commencer.")
