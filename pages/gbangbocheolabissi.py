import altair as alt
import streamlit as st
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import csv
import io
import sys
from typing import Optional

import numpy as np
from numpy.linalg import norm
import pandas as pd


def load_dataset_from_csv(file) -> list:
    uploaded_string = io.StringIO(file.getvalue().decode("utf-8"))
    csv_reader = csv.reader(uploaded_string)
    data = []
    for i, row in enumerate(csv_reader):
        if i == 0:
            try:
                _ = [int(col) for col in row]
            except ValueError:
                continue

        new_row = []
        for col in row:
            if col == '':
                new_row.append(0)
            else:
                new_row.append(int(col))
        data.append(new_row)

    return data


def convert_to_dataframe(dataset):
    return pd.DataFrame(
        dataset,
        index=[f"Line {i + 1}" for i in range(len(dataset))],
        columns=[f"Value {j + 1}" for j in range(len(dataset[0]))]
    )


def euclidean_distance(point: np.ndarray, data: np.ndarray) -> float:
    return np.sqrt(np.sum((point - data) ** 2))


def cosine_similarity(point: np.ndarray, data: np.ndarray) -> float:
    return np.dot(point, data) / (norm(point) * norm(data))


def jaccard_similarity(point: np.ndarray, data: np.ndarray) -> float:
    set1 = set(point)
    set2 = set(data)
    inter = set1.intersection(set2)
    union = set1.union(set2)
    return len(inter) / len(union)


def mean(data):
    return np.array(data).mean(axis=0)


def arrays_equal(arr1, arr2) -> bool:
    return np.array_equal(arr1, arr2)


def silhouette_score(cluster_index, data_point_index, clusters, distance_fn) -> float:
    total_distance_in_cluster = 0

    for item in clusters[cluster_index]:
        total_distance_in_cluster += distance_fn(clusters[cluster_index][data_point_index], item)
    average_distance_in_cluster = total_distance_in_cluster / max(1, len(clusters[cluster_index]) - 1)

    # Get min avg distance to a cluster
    average_distance_to_nearest_cluster = sys.maxsize
    for i in range(len(clusters)):
        if i != cluster_index:
            total_distance_with_cluster_i = 0
            for item in clusters[i]:
                total_distance_with_cluster_i += distance_fn(clusters[cluster_index][data_point_index], item)
            avg_distance_with_cluster_i = total_distance_with_cluster_i / len(clusters[i])
            average_distance_to_nearest_cluster = min(average_distance_to_nearest_cluster, avg_distance_with_cluster_i)

    max_distance = max(average_distance_to_nearest_cluster, average_distance_in_cluster)

    return (average_distance_to_nearest_cluster - average_distance_in_cluster) / max_distance


def intra_cluster_distance(centroid, cluster, distance_fn) -> float:
    distances = [distance_fn(element, centroid) for element in cluster]
    return np.mean(distances)


def inter_cluster_distance(centroids, index, distance_fn) -> float:
    item_count = len(centroids)
    total_distance_between_centroids = 0

    for i in range(item_count):
        if i != index:
            total_distance_between_centroids += distance_fn(centroids[index], centroids[i])

    return total_distance_between_centroids / (item_count - 1)


def to_data_point(data_point_as_str: str, features_number: int) -> Optional[np.ndarray]:
    extracted = data_point_as_str.split(',')
    if len(extracted) != features_number:
        return None

    return np.array([float(str_chunk) for str_chunk in extracted])


USE_WINE_DATASET = 'Use the wine dataset'
USE_IRIS_DATASET = 'Use the iris dataset'
PROVIDE_DATA = 'Provide data'

EUCLIDEAN_DISTANCE = 'Euclidean distance'
COSINE_SIMILARITY = 'Cosine similarity'
JACCARD_SIMILARITY = 'Jaccard similarity'

DISTANCES_FUNCTIONS = {
    EUCLIDEAN_DISTANCE: euclidean_distance,
    COSINE_SIMILARITY: cosine_similarity,
    JACCARD_SIMILARITY: jaccard_similarity
}


class KMeans:
    def __init__(self, dataset, clusters_count: int, max_iterations: int = 300, distance_fn=euclidean_distance):
        self.centroids = []
        self.distance_fn = distance_fn
        self.dataset = dataset
        self.clusters_count = clusters_count
        self.max_iterations = max_iterations

    def pick_centroids(self):
        """
        Using kmeans++ to pick the centroids we will be starting kmeans with
        We create the first centroid py picking it randomly assuming a uniform distribution
        Foreach data point, we compute the distance to the nearest centroid
        The new centroid is the point that is farther than all the other centroids
        (his shortest distance to other is the longest of all)
        """
        index_of_first_centroid = np.random.choice(range(len(self.dataset)))
        centroids = [self.dataset[index_of_first_centroid]]

        for cluster_k in range(self.clusters_count - 1):
            distances_to_centroids = []
            for datum in self.dataset:
                best_distance = sys.maxsize
                for centroid in centroids:
                    distance_to_current_centroid = self.distance_fn(datum, centroid)
                    best_distance = min(best_distance, distance_to_current_centroid)
                distances_to_centroids.append(best_distance)

            next_centroid = self.dataset[np.argmax(distances_to_centroids)]
            centroids.append(next_centroid)
        self.centroids = centroids

    def make_clusters(self):
        """
        K-means implementation to create the clusters
        While the clusters are changing, or we haven't exceeded the number of iterations we build clusters
        At each iteration, reset the clusters to avoid duplicates
        Foreach element, compute the distance with the centroids and assign it to the nearest one
        We generate the new centroids by taking the mean of their clusters and then update the clusters
        :return: cluster_keys : An array containing the centroids of each cluster,
                 cluster_values : An array containing the values of each cluster.
                 The indexes of both arrays are related
        """
        previous_centroids = []
        cluster_centroids = self.centroids[:]
        cluster_values = [[] for _ in range(self.clusters_count)]
        iterations = 0

        while not arrays_equal(previous_centroids, self.centroids) and iterations <= self.max_iterations:
            iterations += 1
            cluster_values = [[] for _ in range(self.clusters_count)]
            for datum in self.dataset:
                smallest_distance = sys.maxsize
                best_centroid_index = None

                for i in range(self.clusters_count):
                    distance = self.distance_fn(datum, cluster_centroids[i])
                    if distance < smallest_distance:
                        smallest_distance = distance
                        best_centroid_index = i

                cluster_values[best_centroid_index].append(datum)

            previous_centroids = self.centroids[:]
            for i in range(self.clusters_count):
                mean_point = mean(cluster_values[i])
                cluster_centroids[i] = mean_point
                self.centroids[i] = mean_point

        return cluster_centroids, cluster_values

    def cluster_standard_deviation(self, cluster_values):
        """
        Computing the standard deviation for a cluster
        For each element of the cluster, compute the squared distance to the mean and divide by the count
        :param cluster_values: The values contained inside a given cluster
        :return: The standard deviation in the given cluster
        """
        mean_value = mean(cluster_values)
        return sum(self.distance_fn(value, mean_value) ** 2 for value in cluster_values) / len(cluster_values)

    def predict_cluster(self, item):
        """
        Given an item, predict the cluster it will belong to
        :param item: The vector we want to find the appropriate cluster for
        :return: the index of the couple (centroid, cluster) and the related centroid
        """
        smallest_distance = sys.maxsize
        index = 0

        for i, centroid in enumerate(self.centroids):
            current_distance = self.distance_fn(item, centroid)
            if current_distance < smallest_distance:
                smallest_distance = current_distance
                index = i

        return index, self.centroids[index]

    @staticmethod
    def get_best_k_using_silhouette_score(dataset, distance_fn):
        """
        Estimate the best k that should (have been) be used for the clustering
        We test values of k between 2 and 10
        :param dataset: The dataset we want to compute the best k to use for clustering
        :param distance_fn: The function used to compute the distance between two items
        :return: The best k that maximizes the silhouette score we found and a dictionary containing the silhouette
        score obtained for each k
        """
        values_of_k_and_silhouette = {}
        best_k = 0
        best_silhouette = -10

        for k in range(2, 13):
            instance = KMeans(dataset, clusters_count=k)
            instance.pick_centroids()
            centroids, clusters = instance.make_clusters()
            count = 0
            total_silhouette_score = 0
            for cluster_index in range(k):
                for item_index in range(len(clusters[cluster_index])):
                    total_silhouette_score += silhouette_score(cluster_index, item_index, clusters, distance_fn)
                    count += 1

            avg_silhouette_score = total_silhouette_score / count
            values_of_k_and_silhouette[k] = avg_silhouette_score

            if best_silhouette < avg_silhouette_score:
                best_silhouette = avg_silhouette_score
                best_k = k

        return best_k, values_of_k_and_silhouette

    @staticmethod
    def get_best_k_using_intra_and_inter_cluster_distance(dataset, distance_fn):
        """
        Estimate the best k that should (have been) be used for the clustering
        :param dataset: The dataset we want to compute the best k to use for clustering
        :param distance_fn: The function used to compute the distance between two items
        :return: The best k that maximizes the inter-cluster distance while minimizing the intra-cluster distance
        and a list of tuples (k, inter-cluster distance, intra-cluster distance)
        """

        results = []
        for k in range(2, 13):
            instance = KMeans(dataset, clusters_count=k)
            instance.pick_centroids()
            centroids, clusters = instance.make_clusters()
            inter_cluster_distances = [inter_cluster_distance(centroids, i, distance_fn) for i in range(k)]
            avg_inter_cluster_distance = np.mean(inter_cluster_distances)
            intra_cluster_distances = [intra_cluster_distance(centroids[i], cluster, distance_fn) for i, cluster in
                                       enumerate(clusters)]
            avg_intra_cluster_distance = np.mean(intra_cluster_distances)
            results.append((k, avg_inter_cluster_distance, avg_intra_cluster_distance))

        best_k = max(enumerate(results), key=lambda x: x[1][1] / x[1][2])[0] + 2
        return best_k, results

    @staticmethod
    def get_results_using_sse(dataset, distance_fn):
        """
        Compute the sum of squared errors for different values of k
        :param dataset
        :param distance_fn
        :return: A dictionary containing the number of clusters and the associated sse
        """
        values_of_k_and_sse = {}
        best_k = 0

        for k in range(2, 13):
            instance = KMeans(dataset, clusters_count=k)
            instance.pick_centroids()
            centroids, clusters = instance.make_clusters()
            sse = 0
            for cluster_index in range(k):
                for item in clusters[cluster_index]:
                    sse += distance_fn(item, centroids[cluster_index]) ** 2

            values_of_k_and_sse[k] = sse

        return values_of_k_and_sse


def main():
    st.set_page_config(page_title="K-means")
    st.title('K-means playground')
    st.info(
        'You can either upload a csv file containing the data or enter manually the data. In both cases, you will be '
        'able to modify the data and start the process. You can also view the algorithm application on a sample '
        'dataset.\nThe data used will be rescaled', icon="ℹ️")
    st.divider()
    option = st.selectbox('I want to : ', [PROVIDE_DATA, USE_WINE_DATASET])

    if option is USE_WINE_DATASET:
        st.session_state.dataset = datasets.load_wine().data[:]
        items_count = len(st.session_state.dataset)
        features_count = len(st.session_state.dataset[0])
    else:
        uploaded_file = st.file_uploader("Upload a csv file", type="csv")

        if 'dataset' not in st.session_state:
            st.session_state.dataset = []

        if uploaded_file is not None:
            st.session_state.dataset = load_dataset_from_csv(uploaded_file)
            st.session_state.items_count = len(st.session_state.dataset)
            st.session_state.features_count = len(st.session_state.dataset[0])

        col1, col2 = st.columns(2)

        with col1:
            items_count = st.number_input('Or, enter the number of items', min_value=1, key="items_count",
                                          disabled=uploaded_file is not None)
        with col2:
            features_count = st.number_input('Enter the number of features', min_value=1, key="features_count",
                                             disabled=uploaded_file is not None)

        if uploaded_file is None:
            if st.session_state.dataset is None:
                st.session_state.dataset = [[0 for _ in range(features_count)] for _ in range(items_count)]
            else:
                if items_count < len(st.session_state.dataset):
                    st.session_state.dataset = st.session_state.dataset[:items_count]

                for i in range(len(st.session_state.dataset)):
                    if features_count < len(st.session_state.dataset[i]):
                        st.session_state.dataset[i] = st.session_state.dataset[i][:features_count]
                    elif features_count > len(st.session_state.dataset[i]):
                        st.session_state.dataset[i] += [0] * (features_count - len(st.session_state.dataset[i]))

                if items_count > len(st.session_state.dataset):
                    st.session_state.dataset += [[0] * features_count for _ in
                                                 range(items_count - len(st.session_state.dataset))]

        edited_df = st.data_editor(convert_to_dataframe(st.session_state.dataset), use_container_width=True)

        if edited_df is not None:
            st.session_state.dataset = edited_df.values.tolist()

    scaler = MinMaxScaler()
    np_dataset = scaler.fit_transform(np.array(st.session_state.dataset))

    if len(st.session_state.dataset) > 1:
        col1, col2 = st.columns(2)
        with col1:
            clusters_number = st.number_input('Enter the number of clusters to make', min_value=2)
        with col2:
            distance_function_str = st.selectbox(
                label='Select how will the distances be computed in the dataset',
                options=[EUCLIDEAN_DISTANCE, COSINE_SIMILARITY, JACCARD_SIMILARITY]
            )

        to_put_in_cluster = st.text_input(
            label=f'Enter a data point ({features_count} features) separated by commas in the following input and the '
                  f'cluster the point belongs to will be found after the clustering',
            key='to_put_in_cluster'
        )

        data_point = []
        if st.session_state.to_put_in_cluster is not None and st.session_state.to_put_in_cluster != '':
            data_point = to_data_point(st.session_state.to_put_in_cluster, features_count)
            if data_point is None:
                st.write("Make sure you entered a valid value with all the features")

        distance_function = DISTANCES_FUNCTIONS[distance_function_str]
        clicked = st.button("Generate the clusters", type="primary", use_container_width=True, key='clicked')

        if clicked:
            with st.spinner('Doing computations...'):
                kmeans = KMeans(np_dataset, clusters_count=clusters_number)
                kmeans.pick_centroids()
                centroids, groups = kmeans.make_clusters()
                standard_deviations = [kmeans.cluster_standard_deviation(cluster) for cluster in groups]
                inter_cluster_distances = [inter_cluster_distance(centroids, i, distance_function) for i in
                                           range(clusters_number)]
                intra_cluster_distances = [intra_cluster_distance(centroids[i], groups[i], distance_function) for i in
                                           range(clusters_number)]
                table_dataframe = pd.DataFrame(
                    data=[
                        [
                            [round(centroid, 2) for centroid in centroids[i]],
                            groups[i],
                            standard_deviations[i],
                            inter_cluster_distances[i],
                            intra_cluster_distances[i]
                        ] for i in range(clusters_number)
                    ],
                    columns=['Centroids', 'Cluster items', 'Standard deviation', 'Inter cluster distance',
                             'Intra cluster distance']
                )
            st.write("The following table shows details about the generated clusters (centroids, items, "
                     "standard deviation, inter-cluster and intra-cluster distance)")
            st.data_editor(table_dataframe, disabled=True)

            if features_count > 2:
                pca_transformer = PCA(n_components=2)
                centroids_transformed = pca_transformer.fit_transform(centroids)
                groups_transformed = [pca_transformer.transform(group) for group in groups]
            else:
                centroids_transformed = centroids
                groups_transformed = groups

            st.write("Here is a plot the of the clusters obtained and their centroid (squares)")
            plots = []

            centroids_df = pd.DataFrame({
                'Component 1': [a[0] for a in centroids_transformed],
                'Component 2': [a[1] for a in centroids_transformed]
            })

            plots.append(alt.Chart(centroids_df).mark_square(size=250)
                         .encode(x='Component 1', y='Component 2', color=alt.value('grey')))

            for index, group in enumerate(groups_transformed):
                group_df = pd.DataFrame({
                    'Component 1': [a[0] for a in group],
                    'Component 2': [a[1] for a in group],
                    'cluster': [index + 1 for _ in group]
                })

                plots.append(alt.Chart(group_df).mark_point().encode(
                    x='Component 1',
                    y='Component 2',
                    color=alt.Color('cluster', legend=alt.Legend(values=list(range(1, 1 + clusters_number)))))
                )
            st.altair_chart(alt.layer(*(plot for plot in plots)), use_container_width=True)

            if data_point is not None and len(data_point) > 2:
                cluster, centroid = kmeans.predict_cluster(scaler.transform([data_point])[0])
                st.write(f"The data point previously entered belongs to the cluster {cluster + 1} "
                         f"which has the centroid\n{centroid}")

            st.header('Finding the best value for k', divider='rainbow')
            st.subheader('First approach')
            st.write("We will use the intra and inter-cluster distance to find the best value of K since the optimal K "
                     "should minimize the intra-cluster distance, maximize the inter-cluster distance and therefore "
                     "maximize the inter/intra-cluster distance ratio")

            best_k_with_ratio, details_with_ratio = kmeans.get_best_k_using_intra_and_inter_cluster_distance(
                np_dataset,
                distance_function
            )

            st.line_chart(pd.DataFrame({
                'ratio': [a[1] / a[2] for a in details_with_ratio],
                'k': [a[0] for a in details_with_ratio]
            }), x="k", y="ratio")

            st.subheader('Second approach')
            st.write("As you can see, proceeding like that would lead to maximize the number of "
                     f"clusters ({best_k_with_ratio}).\nIt is possible to counter this downside by looking at the elbow "
                     f"of the curve of the sse")
            details_with_sse = kmeans.get_results_using_sse(
                np_dataset,
                distance_function
            )

            st.line_chart(pd.DataFrame({
                'sse': [a for a in details_with_sse.values()],
                'k': [a for a in details_with_sse.keys()]
            }), x="k", y="sse")

            st.subheader('Third approach')
            st.write("We will use the silhouette score which provides an analytical method to find "
                     "the best value of k.\n Here is the plot of the score obtained for different values of k")
            best_k_with_silhouette, details_with_silhouette = kmeans.get_best_k_using_silhouette_score(
                np_dataset,
                distance_function
            )

            st.line_chart(pd.DataFrame({
                'silhouette': [a for a in details_with_silhouette.values()],
                'k': [a for a in details_with_silhouette.keys()]
            }), x="k", y="silhouette")

            st.write(f"According to the silhouette score, the best k is {best_k_with_silhouette}")


if __name__ == '__main__':
    main()
