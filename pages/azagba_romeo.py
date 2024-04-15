import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

st.set_page_config(
    page_title="By AZAGBA Roméo",
    page_icon=":house:",
    initial_sidebar_state="auto",
)

def is_string(element):
    return isinstance(element, str)

def rand_centroids(data, k):
    centroids = []
    while len(centroids) < k:
        rand_idx = np.random.randint(len(data))
        centroid = data[rand_idx]
        if centroid not in centroids:
            centroids.append(centroid)
    return centroids

def distance_eucli(point1, point2):
    distance_carre = 0
    for coord1, coord2 in zip(point1, point2):
        distance_carre += (coord1 - coord2) ** 2
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
        cen = []
        for value in centroids[i]:
            cen.append(value)
        chaine_cen = ', '.join(str(x) for x in cen)
        dataframe[f'Distance(*, Cen{i+1} = ({chaine_cen}))'] = distances[i]
        
    dataframe['Cluster'] = pd.Series([None] * len(data))
    for i, cluster in enumerate(clusters):
        for point in cluster:
            condition = ' & '.join([f"(dataframe['x{i+1}'] == {point[i]})" for i in range(len(point))])
            index = dataframe[eval(condition)].index.values[0]
            dataframe.at[index, 'Cluster'] = f'C{i+1}'
    
    return dataframe, centroids

def k_means(data, centroids, k):
    min_iteration = 100
    max_iteraton = max(min_iteration, len(data) * k)
    
    dataframes = []
    
    dataframe, centroids = iteration(data, centroids, k)
    dataframes.append(dataframe)
    for i in range(max_iteraton):
        new_centroids = []
        for j in range(k):
            is_cluster = dataframe['Cluster'] == f'C{j+1}'
            
            new_centroid = []
            for l in range(len(data[0])): 
                coord_new_centroid = dataframe[is_cluster][f'x{l+1}'].mean()
                new_centroid.append(coord_new_centroid)
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
        cluster_points = dataframe[dataframe['Cluster'] == cluster][dataframe.columns[:-1]]
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
        centroids = rand_centroids(data, k)
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
    st.title('Téléversement de fichier CSV')
    
    file = st.file_uploader('Télécharger le fichier CSV comportant vos données', type='csv')
    
    if 'centroids' in st.session_state:
        del st.session_state['centroids']
    if file is not None:
        data = pd.read_csv(file)
        st.markdown("<h4 style='font-weight:bold;'>Affichage des 5 premières lignes </h4>", unsafe_allow_html=True)
        st.dataframe(data.head())
        data_list = data.values.tolist()
        
        selected_columns = st.multiselect("Sélectionnez les colonnes de votre dataset utilisés comme points de données", data.columns.tolist(), placeholder="Choisir les colonnes")
        
        if len(selected_columns) >= 2:
            selected_data = data.loc[:, selected_columns]
            
            nan_values = selected_data.isna().any()
            nan_columns = {column: has_nan for column, has_nan in nan_values.items()}

            null_values = selected_data.isnull().any()
            null_columns = {column: has_null for column, has_null in null_values.items()}

            string_values = selected_data.applymap(is_string)
            string_columns = {column: has_string for column, has_string in string_values.any().items()}
            
            nan_list = [True for col, has_nan in nan_columns.items() if has_nan]
            null_list = [True for col, has_null in null_columns.items() if has_null]
            strings_list = [True for col, has_string in string_columns.items() if has_string]
            
            if nan_list or null_list or strings_list:
                for col, has_nan in nan_columns.items(): 
                    if has_nan:
                        st.warning(f'La colonne {col} contient des valeurs NaN')
                for col, has_null in null_columns.items(): 
                    if has_null:
                        st.warning(f'La colonne {col} contient des valeurs Nulles')
                for col, has_string in string_columns.items(): 
                    if has_string:
                        st.warning(f'La colonne {col} contient des valeurs Strings')
            else:
            
                unique_data = selected_data.drop_duplicates()
                st.dataframe(unique_data.head())
                selected_data_list = unique_data.values.tolist()
                
                option = st.selectbox('Voulez-vous définir le nombre k de cluster ? ', ['Choisir','Oui', 'Non'])
                if option == 'Choisir':
                    st.markdown("<div style='background:rgba(61, 157, 243, 0.2);color:rgb(199, 235, 255);border-radius:10px;padding:15px'><ul><li>Si Oui : Vous allez choisir le nombre k de cluster que vous voulez en fonction de la taille de vos données </li><li>Si Non : Le programme vous suggérera le meilleur k </li></ul></div>", unsafe_allow_html=True)
                elif option == 'Oui':
                    st.markdown("<h3 style='font-weight:bold;'>Définition du k jusqu'à prédiction du groupe d'un new point </h3>", unsafe_allow_html=True)
                    st.markdown("<h4 style='font-weight:bold;'>Choix du nombre k de cluster </h4>", unsafe_allow_html=True)
                    k = st.number_input('Entrer le nombre k de cluster', min_value=2, step=1)
                    
                    if k is not None and k >= 2:
                        if not 'centroids' in st.session_state:
                            st.session_state.centroids = rand_centroids(selected_data_list, k)
                        if 'centroids' in st.session_state:
                            centroids_rand = st.session_state.centroids
                            #centroids = [[1, 4], [1, 5]]
                            
                            st.markdown(f"<h4 style='font-weight:bold;'>Centroïdes choisies aléatoirement</h4>", unsafe_allow_html=True)
                            for i, value in enumerate(centroids_rand):
                                st.write(f'Centroïde {i+1} : ', ', '.join(str(x) for x in value))
                            
                            dataframe, centroids, dataframes = k_means(selected_data_list, centroids_rand, k)
                            
                            for index, value in enumerate(dataframes, 1):
                                st.markdown(f"<h4 style='font-weight:bold;'>Itération {index}</h4>", unsafe_allow_html=True)
                                st.dataframe(value)

                            pca = PCA(n_components=3)  
                            pca_result = pca.fit_transform(dataframe.drop('Cluster', axis=1))
                            
                            dataframe_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2', 'PC3'])

                            dataframe_pca['Cluster'] = dataframe['Cluster']

                            fig = plt.figure(figsize=(8, 6))
                            
                            ax = fig.add_subplot(111, projection='3d')
                            for cluster in dataframe_pca['Cluster'].unique():
                                data_cluster = dataframe_pca[dataframe_pca['Cluster'] == cluster]
                                ax.scatter(data_cluster['PC1'], data_cluster['PC2'], data_cluster['PC3'], label=f'Cluster {cluster}')
                            
                            for i, centroid in enumerate(centroids):
                                if len(centroid) > 2:
                                    ax.scatter(centroid[0], centroid[1], centroid[2], c='black', marker='D', s=100, label=f'Centroïde {i+1}')
                                else:
                                    ax.scatter(centroid[0], centroid[1], c='black', marker='D', s=100, label=f'Centroïde {i+1}')
                    
                            ax.set_xlabel('PC1')
                            ax.set_ylabel('PC2')
                            ax.set_zlabel('PC3')
                            ax.set_title('Nuage des points des composantes principales (PC1, PC2, PC3)')
                            ax.legend()
                            st.pyplot(fig, use_container_width=True)
                            
                            tab1, tab2 = st.tabs(['Prédiction du groupe pour un nouveau point', 'Réinitialisation des données'])
                            with tab1:
                                st.markdown("<h4 style='font-weight:bold;'>Prédiction du groupe pour un nouveau point</h4>", unsafe_allow_html=True)
                                
                                columns_to_rename = dict(zip(dataframe.columns, selected_data.columns))
                                dataframe.rename(columns=columns_to_rename, inplace=True)
                                
                                new_point = []
                                for col in dataframe.columns[:-(k+1)]:
                                    new_value = st.number_input(f'Valeur de {col}', step=1)
                                    new_point.append(new_value)

                                if st.button('Prédire'):
                                    predicted_cluster = predict_with_dataframe(new_point, dataframe)
                                    
                                    st.success(f"Le nouveau point appartient au cluster {predicted_cluster}.")
                            with tab2:
                                if st.button("Réinitialiser les données"):
                                    del st.session_state['centroids']
                elif option == 'Non':
                    st.markdown("<h3 style='font-weight:bold;'>Analyse et Suggestion de k </h3>", unsafe_allow_html=True)
                    
                    max_k = st.number_input("Entrer le max de la plage de k", min_value=2, step=1)
                    
                    if max_k >= 10:
                        st.markdown("<h4 style='font-weight:bold;'>Analyse de la qualité intra de chaque cluster pour des valeurs de k différent en utilisant la métrique Inertie intra-cluster </h4>", unsafe_allow_html=True)
                        
                        k_suggested_intra = suggest_optimal_k_with_intra(selected_data_list, max_k=max_k)       
                        
                        st.markdown(f"<label>Le k optimal suggéré avec la métrique intra-cluster est  : <b>{k_suggested_intra}</b> </label>", unsafe_allow_html=True)
                    else:
                        st.warning('Le max de la plage de k doit être être supérieur ou égal à 10')
        else:
            st.warning(f"Veuillez choisir les colonnes (points de données)\n\n"
                        f"Nombre de colonne >= 2")    
                
if __name__ == '__main__':
    main()