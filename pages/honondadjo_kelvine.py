import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def rdm_centroid(data, k):
    rand_k_data = [np.random.randint(len(data)) for i in range(k)]
    centroids = [data[i] for i in rand_k_data]
    return centroids

def distance_eucli(point1, point2):
    distance_carre = 0
    for c1, c2 in zip(point1, point2):
        distance_carre += (c1 - c2) ** 2
    return np.sqrt(distance_carre)

def iteration(data, centroids, k):
    distances = [[] for j in range(k)]
    for i, centroid in enumerate(centroids):
        for d in data:
            distance = distance_eucli(d, centroid)
            distances[i].append(distance)
    clusters = [[] for j in range(k)]
    for i in range(len(data)):
        min_distance = float('inf')
        min_cluster = None
        for j in range(k):
            if distances[j][i] < min_distance:
                min_distance = distances[j][i]
                min_cluster = j
        clusters[min_cluster].append(data[i])
    
    columns = [f'x{i+1}' for i in range(len(data[0]))]
    dataframe = pd.DataFrame(data, columns=columns)
    
    for i in range(k):
        cen = [value for value in centroids[i]]
        chaine_cen = ', '.join(str(x) for x in cen)
        dataframe[f'Distance(*, Cen{i+1} = ({chaine_cen}))'] = distances[i]
        
    dataframe['Cluster'] = pd.Series([None] * len(data))
    for i, cluster in enumerate(clusters):
        for point in cluster:
            condition = ' & '.join([f"(dataframe['x{i+1}'] == {point[i]})" for i in range(len(point))])
            index = dataframe[eval(condition)].index.values[0]
            dataframe.at[index, 'Cluster'] = f'C{i+1}'
    
    return dataframe, centroids

def k_means(data, centroids, k=2):
    min_iteration = 100
    max_iteraton = max(min_iteration, len(data) * k)
    
    dataframes = []
    dataframe, centroids = iteration(data, centroids, k)
    dataframes.append(dataframe)
    for i in range(max_iteraton):
        new_centroids = []
        for j in range(k):
            is_cluster = dataframe['Cluster'] == f'C{j+1}'
            new_centroid = [dataframe[is_cluster][f'x{l+1}'].mean() for l in range(len(data[0]))]
            new_centroids.append(new_centroid)
            
        new_dataframe, new_centroids = iteration(data, new_centroids, k)
        
        is_convergence = dataframe['Cluster'].equals(new_dataframe['Cluster'])     
        
        dataframe = new_dataframe
        centroids = new_centroids 
        dataframes.append(dataframe)
        
        if is_convergence:
            break
    
    return dataframe, centroids, dataframes

def predict_with_dataframe(point, dataframe):
    min_distance = float('inf')
    predicted_cluster = None
    
    for cluster in dataframe['Cluster'].unique():
        cluster_points = dataframe[dataframe['Cluster'] == cluster][[f'x{i+1}' for i in range(len(point))]]
        cluster_centroid = cluster_points.mean()
        distance = distance_eucli(point, cluster_centroid)
        
        if distance < min_distance:
            min_distance = distance
            predicted_cluster = cluster
        
    return predicted_cluster

def calcul_inertia(data, centroids, k):
    inertia = 0
    dataframe, _ = iteration(data, centroids, k)
    for i in range(k):
        is_cluster = dataframe['Cluster'] == f'C{i+1}'
        cluster_points = dataframe[is_cluster][[f'x{j+1}' for j in range(len(data[0]))]]
        centroid = np.array(centroids[i])
        inertia += np.sum(np.linalg.norm(cluster_points - centroid, axis=1))
    return inertia

def suggest_optimal_k_with_intra(data, max_k=10):
    inertias = []
    k_values = range(2, max_k)
    for k in k_values:
        centroids = rdm_centroid(data, k)
        _, final_centroids, _ = k_means(data, centroids, k)
        inertia = calcul_inertia(data, final_centroids, k)
        inertias.append(inertia)
        
    inert = np.gradient(inertias)
    optimal_k = k_values[np.argmin(inert[1:])] + 2
   
    fig = plt.figure(figsize=(8, 6))
    plt.plot(k_values, inertias, marker='o')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Inertie')
    plt.title('Méthode de coude')
    st.pyplot(fig)
    
    return optimal_k   
 
def main():
    st.title("K-Means Clustering with Streamlit")
    
    file = st.file_uploader("Téléchargez un fichier CSV contenant vos données", type="csv")

    
    if file is not None:
        data = pd.read_csv(file)
        
        check_s = [True for col in data.columns if data[col].dtype == 'object' ]
        if data.isnull().values.any() or check_s:
            st.warning(' Votre fichier CSV contient des valeurs NaN. Veuillez nettoyer vos données.')
            return
        
        if st.checkbox('Afficher les données de votre data'):
            st.write(data.head())
        
        data_list = data.values.tolist()

        k = st.slider("Selectionner le nombre de  clusters (k)", min_value=2, max_value=10, value=3)
        affiche_centroid = st.checkbox('Afficher les centroides')

        if k is not None :
            centroids = rdm_centroid(data_list, k)
            for i,value in enumerate(centroids):
                if affiche_centroid:
                   st.write(f'Centroide{i+1}:',';'.join(str(x) for x in value)) 
            #st.write(rdm_centroid(data, k))
            
            st.markdown("### Effectuer un regroupement de K-Means")
    
                
            dataframe, centroids, dataframes = k_means(data_list, centroids, k)
            
            for index, value in enumerate(dataframes, 1):
                st.markdown(f"<h4 style='font-weight:bold;'>{index}</h4>", unsafe_allow_html=True)
                st.dataframe(value)

            colors = ['blue', 'red', 'violet', 'cyan', 'orange', 'magenta', 'black', 'orange', 'aqua', 'pink', 'plum', 'beige']
            palette_colors = {}

            for index, row in dataframe.iterrows():
                for cluster in dataframe.columns:
                    if cluster == 'Cluster':
                        c = dataframe.loc[index, cluster]
                        for j in range(k):
                            c = f'C{j+1}'
                            palette_colors[c] = colors[j]
            
            pca = PCA(n_components=3)  
            pca_result = pca.fit_transform(dataframe.drop('Cluster', axis=1))
            
            dataframe_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])

            dataframe_pca['Cluster'] = dataframe['Cluster']

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            for cluster in dataframe_pca['Cluster'].unique():
                data_cluster = dataframe_pca[dataframe_pca['Cluster'] == cluster]
                ax.scatter(data_cluster['PC1'], data_cluster['PC2'], data_cluster['PC3'], label=f'Cluster {cluster}')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.set_zlabel('PC3')
            ax.set_title('Nuage des points des composantes principales (PC1, PC2, PC3)')
            ax.legend()
            st.pyplot(plt)

            new_point = []
            for col in dataframe.columns[:-(k+1)]: 
                new_value = st.number_input(f'Valeur de {col}')
                new_point.append(new_value)
            if st.button('Prédire'):
                predicted_cluster = predict_with_dataframe(new_point, dataframe)
                
                st.success(f"Le nouveau point appartient au cluster {predicted_cluster}.")

            if st.button("Afficher le graphique de la méthode des coudes"):
                suggestion = suggest_optimal_k_with_intra(data_list, max_k=10)
                st.write(f'le k optimal est {suggestion}')
            
            
        
        
main()
