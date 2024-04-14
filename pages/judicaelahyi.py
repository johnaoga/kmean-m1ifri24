"""

"""
import streamlit as st
import pandas as pd


def load_data(file_path):
    """
    """
    return pd.read_csv(file_path)


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

    i = st.number_input(
        "Enter max number of iterations",
        step=1,
        value=100,
        min_value=1,
    )


if __name__ == "__main__":
    main()
