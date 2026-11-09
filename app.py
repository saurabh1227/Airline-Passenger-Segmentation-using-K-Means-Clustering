"""
Airline Passenger Segmentation - Streamlit app (K-Means on service ratings).

Run:  streamlit run app.py
Needs the files created by the notebook: the model/ folder and sample_data.csv
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from utils import RATING_COLS, TARGET_COL, assign_clusters

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

st.set_page_config(page_title="Airline Passenger Segmentation", page_icon="✈️", layout="wide")


@st.cache_resource
def load_artifacts():
    """Load the saved model files once and reuse them."""
    kmeans = joblib.load(MODEL_DIR / "kmeans.joblib")
    scaler = joblib.load(MODEL_DIR / "scaler.joblib")
    pca = joblib.load(MODEL_DIR / "pca.joblib")
    with open(MODEL_DIR / "meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    profile = pd.read_csv(MODEL_DIR / "cluster_profile.csv")
    k_table = pd.read_csv(MODEL_DIR / "k_selection.csv")
    return kmeans, scaler, pca, meta, profile, k_table


kmeans, scaler, pca, meta, profile, k_table = load_artifacts()
COLORS = plt.get_cmap("tab10")


def pca_scatter(X_scaled, labels, max_points=5000):
    """2D PCA picture of the clusters (sampled when the data is large)."""
    idx = np.arange(len(labels))
    if len(idx) > max_points:
        idx = np.random.RandomState(42).choice(idx, max_points, replace=False)
    coords = pca.transform(X_scaled[idx])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for c in sorted(set(labels)):
        mask = labels[idx] == c
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6,
                   color=COLORS(int(c) % 10), label=f"Cluster {c}")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.legend(markerscale=2)
    fig.tight_layout()
    return fig


def show_results(df, labels, X_scaled):
    results = df.copy()
    results["cluster"] = labels

    c1, c2 = st.columns(2)
    c1.metric("Passengers", f"{len(results):,}")
    c2.metric("Clusters found", results["cluster"].nunique())

    left, right = st.columns(2)
    with left:
        st.subheader("Cluster sizes")
        st.bar_chart(results["cluster"].value_counts().sort_index())
    with right:
        st.subheader("Clusters in 2D (PCA)")
        fig = pca_scatter(X_scaled, labels)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Average rating per cluster")
    summary = results.groupby("cluster")[RATING_COLS].mean().round(2)
    summary.insert(0, "passengers", results["cluster"].value_counts().sort_index())
    if TARGET_COL in results.columns:
        summary.insert(1, "satisfied %", (results[TARGET_COL].eq("satisfied")
                                          .groupby(results["cluster"]).mean() * 100).round(1))
    st.dataframe(summary)

    st.subheader("What defines each cluster")
    st.dataframe(pd.DataFrame({"Cluster": list(meta["traits"].keys()),
                               "Main traits": list(meta["traits"].values())}))

    st.download_button("Download data with cluster column",
                       results.to_csv(index=False).encode("utf-8"),
                       file_name="passengers_with_clusters.csv", mime="text/csv")


# ---------- sidebar ----------
with st.sidebar:
    st.header("About")
    st.write(
        "Passengers are grouped with **K-Means** using their 14 service ratings "
        "(seat comfort, wifi, cleanliness, ...). Ratings are standardized "
        "(z-score) before clustering."
    )
    st.write(f"Number of clusters: **{meta['k']}**")
    st.caption("Satisfaction is not used for clustering. It is only used to describe the clusters.")

st.title("Airline Passenger Segmentation")
st.caption("Group airline passengers by how they rate the service, using K-Means clustering.")

tab1, tab2, tab3 = st.tabs(["Segment a CSV", "Single passenger", "Model details"])

# ---------- tab 1: CSV ----------
with tab1:
    source = st.radio("Data source", ["Use sample data", "Upload CSV"], horizontal=True)
    data = None
    if source == "Use sample data":
        data = pd.read_csv(BASE_DIR / "sample_data.csv")
    else:
        uploaded = st.file_uploader("Upload passenger CSV (needs the 14 rating columns)", type=["csv"])
        if uploaded is not None:
            data = pd.read_csv(uploaded)
        else:
            st.info("Upload a CSV with the 14 rating columns, or switch to the sample data.")
    if data is not None:
        st.write(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
        try:
            labels, X_scaled = assign_clusters(data, scaler, kmeans)
        except ValueError as err:
            st.error(str(err))
        else:
            show_results(data, labels, X_scaled)

# ---------- tab 2: one passenger ----------
with tab2:
    st.write("Rate each service from 0 to 5 (0 = not applicable) and see which cluster the passenger belongs to.")
    with st.form("passenger_form"):
        cols = st.columns(3)
        ratings = {}
        for i, name in enumerate(RATING_COLS):
            ratings[name] = cols[i % 3].slider(name, 0, 5, 3)
        submitted = st.form_submit_button("Find cluster")
    if submitted:
        row = pd.DataFrame([ratings])
        label, _ = assign_clusters(row, scaler, kmeans)
        label = int(label[0])
        info = profile[profile["cluster"] == label].iloc[0]
        st.success(f"This passenger belongs to **Cluster {label}**")
        st.write(meta["traits"][str(label)])
        st.write(f"In this cluster, **{info['satisfied_pct']:.1f}%** of passengers are satisfied "
                 f"({int(info['size']):,} passengers, {info['share_pct']:.1f}% of all).")
        compare = pd.DataFrame({"This passenger": pd.Series(ratings),
                                f"Cluster {label} average": info[RATING_COLS].astype(float)})
        st.bar_chart(compare)

# ---------- tab 3: model details ----------
with tab3:
    st.subheader("Choosing the number of clusters")
    st.dataframe(k_table.round(4))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(k_table["k"], k_table["inertia"], marker="o")
    axes[0].axvline(meta["k"], color="red", linestyle="--")
    axes[0].set_title("Elbow plot (inertia)")
    axes[0].set_xlabel("k")
    sil = k_table.dropna(subset=["silhouette"])
    axes[1].plot(sil["k"], sil["silhouette"], marker="o")
    axes[1].axvline(meta["k"], color="red", linestyle="--")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("k")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(f"Red line = chosen k ({meta['k']}).")

    st.subheader("Cluster profile (full training data)")
    st.dataframe(profile)

    st.subheader("Cluster centers (standardized ratings)")
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=RATING_COLS)
    centers.index.name = "cluster"
    fig, ax = plt.subplots(figsize=(10, 3 + 0.5 * len(centers)))
    sns.heatmap(centers, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Positive = rated above the average passenger, negative = below.")

    with st.expander("Why are the ratings scaled?"):
        st.write(
            "K-Means uses distances, so every feature must be on a comparable scale. "
            "This project uses **StandardScaler (z-score)**: z = (x - mean) / std, "
            "which gives every rating mean 0 and standard deviation 1."
        )
