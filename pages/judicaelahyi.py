"""

"""
import numpy as np
import streamlit as st
import pandas as pd


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
    st.subheader("Setup K-Means Clustering")

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
        st.write(clusters, centroids)


if __name__ == "__main__":
    main()
