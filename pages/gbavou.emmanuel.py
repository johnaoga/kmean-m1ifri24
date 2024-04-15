import streamlit as st
import numpy as np
import pandas as pd
from math import isnan, sqrt
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Clustering K-means",
    page_icon=":rocket:",
    initial_sidebar_state="expanded",
)

def convert_to_numeric(value):
    try:
        return float(value)
    except ValueError:
        return np.nan


def distance_euclidienne(item1:tuple , item2:tuple):
    
    distance = float(0)
    
    if len(item1)==len(item2):
        for i in range(len(item1)):
            
            try:
                x = float(item1[i])
            except:
                x = 0
                
            try:
                y = float(item2[i])
            except:
                y = 0
                
            distance = float(distance) +  float((x - y)**2)
           
        distance = sqrt(distance)
        
    return distance

def cluster_distance_column_names(k: int) -> list: 
    
    columns = []
    for i in range(k):
        columns.append(f"distance_c{i+1}")
    
    return columns

def cluster_column_names(k: int) -> list: 
    
    columns = {}
    for i in range(k):
        columns[f"distance_c{i+1}"] = f"c{i+1}"
    
    return columns

def define_cluster_center(df : pd.DataFrame, k : int, columns) -> list:
    
    data_frame = df
    centers = []
    
    for i in range(k):
        filter_df = data_frame[data_frame['cluster'] == f"c{i+1}"]
        centers.append(tuple(filter_df[list(columns)].mean()))
                
    return centers

def kmeans_iteration(df : pd.DataFrame, default_centroides: list):
    
    data_frame = df
    df_to_numpy = df.to_numpy()
    k = len(default_centroides)
    
    for i in range(k):
        series = []
        for j in range(len(df.to_numpy())):
            distance = distance_euclidienne(tuple(df_to_numpy[j]),tuple(default_centroides[i]))
            series.append(distance)
               
        data_frame[f"distance_c{i+1}"]  = series
    
    data_frame['cluster'] = data_frame[cluster_distance_column_names(k)].idxmin(axis=1)
    
    data_frame['cluster'] =  data_frame['cluster'].map(cluster_column_names(k))
    
    return data_frame

def define_default_centers(df : pd.DataFrame, k: int):
    
    default_centroides_dataframe = df.sample(k)
    
    default_centers = []

    for i in range(k):
        default_centers.append(tuple(default_centroides_dataframe.iloc[i]))
        
    return default_centers

def kmeans(df : pd.DataFrame, k):
    
    default_centers =  define_default_centers(df, k)
    
    columns = df.columns
    data_frame_iterations = []
    
    centers = []
    centers.append(default_centers)
            
    data_frame = kmeans_iteration(df.copy(), centers[0])
    
    iteration_number = 0
    data_frame_iterations.append(data_frame)
    
    while True:
        
        centers.append(define_cluster_center(data_frame_iterations[iteration_number], k, columns))

        data_frame_iterations.append(kmeans_iteration(data_frame_iterations[iteration_number][columns].copy(), centers[iteration_number+1]))
        
        iteration_number += 1   
        
        # print(data_frame_iterations[iteration_number]['cluster'].to_numpy())
        # print(data_frame_iterations[(iteration_number-1)]['cluster'].to_numpy())
        
        is_convergence = data_frame_iterations[iteration_number]['cluster'].equals(data_frame_iterations[(iteration_number-1)]['cluster'])
        
        if is_convergence:
            break
        
    clusters = data_frame_iterations
    centers = centers
    
    return clusters, centers        
        

def predict_data_clusters(centers : list,  point: tuple):
    
    min_distance  = None
    cluster = 'c1'
    
    for i in range(len(centers)):
        
        distance = distance_euclidienne(centers[i], point)
        
        if min_distance is None:
            min_distance = distance
            # cluster = f"c{i+1}"
             
        if distance < min_distance:
            min_distance = distance
            cluster = f"c{i+1}"
    
    return cluster

def main():
    
    st.title("Clustering K-means")
    
    st.subheader(":blue[Importation de notre data set.]")
    file = st.file_uploader("Importer le fichier CSV", type="csv")
    
    selected_labels = []
            
    if file is not None:
        
        data_frame = pd.read_csv(file)  
        
        st.dataframe(data_frame.head())
        
        st.subheader(":blue[Choix du K.]")

        k = st.number_input(
            'ENTREZ LE NOMBRE K DE GROUPE A FAIRE: ' ,
            step=1, 
            format="%d",
            min_value=2
        )
        
        if k is not None:
            
            std_data_frame = data_frame[data_frame.describe().columns]
            
            with st.spinner('Veuillez patientez le temps que notre algorithme de kmean soit en cours'):
                clusters , centers = kmeans(std_data_frame, k)
                        
            st.subheader(":blue[Affichage des resultats obtenue après chaque itération jusqu'à convergence.]")
            
            for i in range(len(clusters)):  
                
                st.subheader(f"📋 Itération {i+1}")

                with st.expander(f"Cliquez pour afficher le dataframe", i==(len(clusters)-1)):
                    
                    for j in range(len(centers[i])):
                        
                        st.write(f"Centroide du cluster n° {j+1}: {centers[i][j]}")
                    
                    st.dataframe(clusters[i])
               
            point = []     
            for column in std_data_frame.columns:
                value = st.number_input(
                    f'Entrez la valeur de {str(column).upper()}' ,
                    key=column,
                    value=0,
                    step=1
                )
                point.append(value)
            
            predict = st.button('Faire la prédiction')
            
            if predict:
                predicted_cluster = predict_data_clusters(centers[len(centers)-1] , point)

                st.success(f"Le nouveau point appartient au cluster {predicted_cluster}.")

                    
if __name__ == "__main__":
    main()
