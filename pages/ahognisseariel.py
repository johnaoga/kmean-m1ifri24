import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# FONCTIONS

st.header(':blue[K-means Clustering à partir d\'un csv]')

# Création des différents containers
dataframe_container_ = st.empty()

csv_loader = st.container()

k_choice_ = st.container()

point_predict_container_ = st.container()

metrics_ = st.container()

evaluation_ = st.container()

if 'dataframe_' not in st.session_state:
    st.session_state['dataframe_'] = pd.DataFrame()

if 'show_csv_field_' not in st.session_state:
    st.session_state['show_csv_field_'] = True

if 'choose_k_' not in st.session_state:
    st.session_state['choose_k_'] = False

if 'point_predict_container_' not in st.session_state:
    st.session_state['point_predict_container_'] = False

if 'evaluation_container_' not in st.session_state:
    st.session_state['evaluation_container_'] = False

if 'centroids_' not in st.session_state:
    st.session_state['centroids_'] = None

if 'labels_' not in st.session_state:
    st.session_state['labels_'] = None

if 'plot_' not in st.session_state:
    st.session_state['plot_'] = None

if 'k_' not in st.session_state:
    st.session_state['k_'] = 2

if 'dataset_size_' not in st.session_state:
    st.session_state['dataset_size_'] = 2

if 'scatter_colors_' not in st.session_state:
    st.session_state['scatter_colors_'] = []

datas_ = st.session_state['dataframe_']
labels_ = st.session_state['labels_']
centroids_ = st.session_state['centroids_']
K = st.session_state['k_']
scatter_colors = st.session_state['scatter_colors_']

# Definir la valeur de k
def define_k():
    st.session_state['show_csv_field_'] = not st.session_state['show_csv_field_']
    st.session_state['choose_k_'] = not st.session_state['choose_k_']
    st.session_state['dataset_size_'] = len(datas_)
    scatter_colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(st.session_state['dataset_size_'])]
    st.session_state['scatter_colors_'] = scatter_colors

def kmeans(data, k, n_iters=100):
    centroids = data[np.random.choice(len(data), k, replace=False)]
    for _ in range(n_iters):
        distances = np.linalg.norm(data[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, labels

def compute_std_devs(data, centroids, labels, k):
    std_devs = []
    for i in range(k):
        cluster_data = data[labels == i]
        std_dev = cluster_data.std(axis=0)
        std_devs.append(std_dev)
    return np.array(std_devs)

# Affichage des résultats sous forme de scatter plot 
def plot_clusters(data, centroids, labels, container):
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o')
    for i in range(len(centroids)):
        plt.scatter(centroids[:, 0][i], centroids[:, 1][i], c=scatter_colors[i], marker='x')
    plt.title('Clustering suivant les deux premières variables')

    columns_title = datas_.columns[:2]
    plt.xlabel(columns_title[0])
    plt.ylabel(columns_title[1])
    container.pyplot(plt)

def predict_group(point, centroids):
    distances = np.linalg.norm(centroids - point, axis=1)
    group = np.argmin(distances)
    return group

def inertia(data, centroids, labels):
    inertia = 0
    for i in range(len(centroids)):
        cluster_data = data[labels == i]
        inertia += np.sum((cluster_data - centroids[i]) ** 2)
    return inertia

# Fonction pour calculer le coefficient de Davies-Bouldin
def davies_bouldin_score(data, centroids, labels, k):
    # Calcul des distances intra-cluster
    cluster_distances = np.zeros(k)
    for i in range(k):
        cluster_data = data[labels == i]
        cluster_distances[i] = np.mean(np.linalg.norm(cluster_data - centroids[i], axis=1))
    
    # Calcul des distances inter-cluster
    db_index = 0
    for i in range(k):
        max_ratio = 0
        for j in range(k):
            if i != j:
                inter_cluster_dist = np.linalg.norm(centroids[i] - centroids[j])
                ratio = (cluster_distances[i] + cluster_distances[j]) / inter_cluster_dist
                max_ratio = max(max_ratio, ratio)
        db_index += max_ratio
    
    # Moyenne des ratios pour tous les clusters
    db_index /= k
    return db_index

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def silhouette_score(data, labels, centroids):
    n_clusters = len(np.unique(labels))
    n_samples = len(data)
    
    # Calculer la distance de chaque échantillon à son centroïde
    distances = np.zeros(n_samples)
    for i in range(n_samples):
        cluster_id = labels[i]
        centroid = centroids[cluster_id]
        distances[i] = euclidean_distance(data[i], centroid)
    
    # Calculer la distance moyenne de chaque échantillon aux autres clusters
    cluster_distances = np.zeros((n_samples, n_clusters))
    for i in range(n_samples):
        cluster_id = labels[i]
        for j in range(n_clusters):
            if j != cluster_id:
                cluster_centroid = centroids[j]
                cluster_distances[i, j] = euclidean_distance(data[i], cluster_centroid)
    
    # Calculer les scores de silhouette individuels
    silhouette_samples = np.zeros(n_samples)
    for i in range(n_samples):
        cluster_id = labels[i]
        a = distances[i]
        b = np.min(cluster_distances[i, cluster_distances[i] != 0])
        if a < b:
            silhouette_samples[i] = 1 - a / b
        elif b < a:
            silhouette_samples[i] = b / a - 1
        else:
            silhouette_samples[i] = 0
    
    # Calculer le score de silhouette global
    silhouette_score = np.mean(silhouette_samples)
    
    return silhouette_score


def show_prediction_widget():
    st.session_state['choose_k_'] = not st.session_state['choose_k_']
    st.session_state['point_predict_container_'] = not st.session_state['point_predict_container_']

def index_max(list_):
    value_max = list_[0]
    index_max = 0
    for i in range(1, len(list_)):
        if list_[i] > value_max:
            value_max = list_[i]
            index_max = i
    
    return index_max

# Déroulement du programme
if st.session_state['show_csv_field_']:
    uploaded_file = csv_loader.file_uploader("Charger un csv:", type=["csv"])
    if uploaded_file is not None:
        datas_ = pd.read_csv(uploaded_file)
        st.session_state['dataframe_'] = datas_
        
        csv_loader.button("Valider", on_click= define_k)
    

if st.session_state['choose_k_'] == True:
    dataframe_container_.dataframe(datas_)
    # b) Choisir K
    K = k_choice_.number_input('Valeur du K:', 2, max_value= 10)
    st.session_state['k_'] = K

    if  k_choice_.button("Lancer Kmeans"):
        data = datas_.to_numpy()
        centroids, labels = kmeans(data, K)
        std_devs = compute_std_devs(data, centroids, labels, K)
        st.session_state['centroids_'] = centroids
        st.session_state['labels_'] = labels

        # c) Afficher les résultats pour chaque centroïde et écart-type
        k_choice_.subheader("Centroïdes et écart-type par cluster :")

        clusters_info = []
        for i, (centroid, std_dev) in enumerate(zip(centroids, std_devs)):
            clusters_info.append({
                "Cluster": i+1,
                "Centroïde": centroid,
                "Écart-type": std_dev
            })
        clusters_df = pd.DataFrame(clusters_info)
        k_choice_.table(clusters_df)

        # d) Afficher le résultat sous forme visuelle
        k_choice_.subheader("Plot")
        plot_clusters(data, centroids, labels, k_choice_)

        k_choice_.button("Faire une prédiction pour un point", on_click= show_prediction_widget)


        metrics_.subheader("Métriques pour evaluation des qualités")
        metrics_.markdown("- Silhouette score:")
        metrics_.write("Cette métrique combine à la fois la compacité et la séparation. Elle calcule pour chaque point la différence entre la distance moyenne intra-cluster et la distance moyenne au cluster le plus proche, divisée par le maximum des deux. Les valeurs sont comprises entre -1 et 1, où 1 indique des clusters bien définis, 0 des clusters moyennement définis, et -1 des clusters mal définis.")
        metrics_.write(f"Ici, ce score est de: {silhouette_score(data= data, labels= labels, centroids= centroids)}\n")

        metrics_.markdown("- Inertie:")
        metrics_.write("Une inertie faible indique que les points sont proches des centroïdes de leur cluster respectif, ce qui signifie que les clusters sont compacts (bonne qualité intra-cluster). Une inertie élevée suggère que les points sont éloignés des centroïdes, ce qui indique des clusters peu compacts.")
        metrics_.write(f"Ici, ce score est de: {inertia(data= data, labels= labels, centroids= centroids)}\n")

        metrics_.markdown("- Coefficient de David-Bouldin:")
        metrics_.write("Cette métrique est donc utile pour comparer les résultats de différentes méthodes de clustering ou pour choisir le meilleur nombre de clusters K pour un algorithme comme K-means. On cherchera à minimiser la valeur de la métrique de Davies-Bouldin.")
        metrics_.write(f"Ici, ce score est de: {davies_bouldin_score(k=K, data= data, labels= labels, centroids= centroids)}\n")


        evaluation_.subheader("Evaluation des valeurs de K (Avec silhouette score)")

        metric_values = []
        for i in range(2, 11):
            eval_cetroids, eval_labels = kmeans(data= datas_.to_numpy(), k = i)
            metric_value = silhouette_score(
                data= datas_.to_numpy(),
                labels= eval_labels,
                centroids= eval_cetroids,
            )
            metric_values.append(metric_value)
            evaluation_.write(f"Pour K = {i}, on obtient: {metric_value}")

        evaluation_.write(f"Le nombre idéal de cluster pour ce dataset est: {index_max(metric_values) + 2} ayant le score le plus proche de 1")

if st.session_state['point_predict_container_'] == True:
    plot_clusters(data= datas_.to_numpy(), centroids= centroids_, labels= labels_, container= point_predict_container_)
    dataframe_container_.empty()
    position = 0
    point =  [0] * ( datas_.columns.size )
    for col in datas_.columns:
        point[position] = point_predict_container_.number_input(f"Entrez {col}:", key=col)
        position += 1
    
    group = predict_group(point= np.array(point), centroids= st.session_state['centroids_'])
    point_predict_container_.write(f"Ce point appartiendrait au groupe du centroide de couleur: ")
    square_html = f"""
    <div style='width: 20px; height: 20px; background-color: {scatter_colors[group]};'></div>"""
    st.markdown(square_html, unsafe_allow_html=True)
    
        