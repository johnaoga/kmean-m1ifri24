"""

"""
import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA


def load_data(file_path):
    """
    """
    return pd.read_csv(file_path)


def k_means(data, k, max_iterations=100):
    """
    """
    centroids = data.sample(n=k, replace=True).values
    for _ in range(max_iterations):
        clusters = np.argmin(np.linalg.norm(data.values[:, None] - centroids, axis=2), axis=1)
        new_centroids = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids
    return {
        'clusters': clusters,
        'centroids': centroids,
    }


def compute(type_, data, centroids, clusters):
    _ = 0
    centroid_global = np.mean(data, axis=0) # useful only for between case
    for i, centroid in enumerate(centroids):
        if type_ == 'with':
            _ += np.sum((data[clusters == i] - centroid) ** 2)
        elif type_ == 'between':
            _ += len(data[clusters == i]) * np.sum((centroid - centroid_global) ** 2)
    return _


def display_appreciation_based_on_metric(metric_value):
    if metric_value < 100:
        return ":green[Excellent]"
    elif metric_value < 500:
        return ":blue[Fine]"
    elif metric_value < 1000:
        return ":gray[Good]"
    elif metric_value < 2000:
        return ":orange[Average]"
    else:
        return ":red[Poor]"


def main():
    st.title("K-Means Clustering - Judicael AHYI")
    st.subheader("IFRI 2024, Master Génie Logiciel, Semestre 2")

    uploaded_file_path = st.file_uploader(
        "Upload your csv file",
        type=["csv"]
    )

    if uploaded_file_path is None:
        st.error("Please enter a valid csv file")
        st.stop()

    data = load_data(uploaded_file_path)
    st.write(data)

    # setup step
    st.header("Setup K-Means Clustering")

    k = st.number_input(
        "Enter value of K",
        step=1,
        min_value=1,
        max_value=len(data),
    )

    max_i = st.number_input(
        "Enter max number of iterations",
        step=1,
        value=100,
        min_value=1,
    )

    if st.button("Process Clustering"):
        k_result = k_means(data, k, max_i)
        clusters = k_result['clusters']
        centroids = k_result['centroids']
        st.success("Processing is completed")
        st.session_state.clusters = clusters
        st.session_state.centroids = centroids

    clusters = st.session_state.get('clusters')
    centroids = st.session_state.get('centroids')
    if clusters is not None and centroids is not None:
        st.header("Display results")
        pca = PCA(n_components=2)
        data_reduced = pca.fit_transform(data)
        data_reduced_df = pd.DataFrame(data_reduced, columns=['PC1', 'PC2'])

        if st.toggle('RAW'):
            st.subheader("Results (RAW)")
            st.write("Centroids:")
            st.write(pd.DataFrame(centroids, columns=data.columns))
            st.write("Clusters:")
            st.write(pd.DataFrame({"Data Point": np.arange(len(data)), "Cluster": clusters}))
            st.write("Standard deviation of each cluster")
            for i in range(k):
                st.write(f"Cluster {i + 1}: {np.std(data[clusters == i], axis=0)}")

        st.subheader("Results (Visual)")
        clusters = st.session_state.get('clusters')
        plot_data = pd.concat([data_reduced_df, pd.Series(clusters, name='Cluster')], axis=1)
        fig = px.scatter(plot_data, x='PC1', y='PC2', color='Cluster', title='K-means Clustering with PCA')
        st.plotly_chart(fig)

        st.header("Predict Cluster")
        centroids = st.session_state.get('centroids')
        if centroids is not None:
            point_values = []
            for i in range(data.shape[1]):
                value = st.number_input(f"Enter the value of feature {i + 1}", step=0.01)
                point_values.append(value)

            if st.button("Predict Cluster"):
                distances = np.linalg.norm(centroids - point_values, axis=1)
                cluster_prediction = np.argmin(distances)
                st.write(f"The point is predicted to belong to cluster {cluster_prediction + 1}.")

        st.header("Intra and Inter-cluster Metrics")
        clusters = st.session_state.get('clusters')
        if centroids is not None and clusters is not None:
            w_ = compute('with', data, centroids, clusters)
            b_ = compute('between', data, centroids, clusters)
            st.write(f"Sum of squares")
            st.write(f"-- within clusters: {w_.iloc[0]} - {display_appreciation_based_on_metric(w_.iloc[0])}")
            st.write(f"-- between clusters: {b_} - {display_appreciation_based_on_metric(b_)}")
            for i, centroid in enumerate(centroids):
                cluster_data = data[clusters == i]
                st.subheader(f"Cluster {i}")
                st.write("Number of elements:", len(cluster_data))
                st.write("Standard deviation:", np.std(cluster_data, axis=0))
                st.write("Mean:", np.mean(cluster_data, axis=0))


if __name__ == "__main__":
    main()
