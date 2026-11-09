"""Shared helpers for the Airline Passenger Segmentation (K-Means) project.

Used by both the notebook (training) and app.py (Streamlit), so the feature
selection is identical in both places.
"""
import numpy as np
import pandas as pd

TARGET_COL = "satisfaction"  # NOT used for clustering, only to describe the clusters

# The 14 service-rating columns (0 to 5) used as X for K-Means
RATING_COLS = [
    "Seat comfort",
    "Departure/Arrival time convenient",
    "Food and drink",
    "Gate location",
    "Inflight wifi service",
    "Inflight entertainment",
    "Online support",
    "Ease of Online booking",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
    "Online boarding",
]

# Optional columns: used only to describe the clusters when they are present
PROFILE_COLS = [
    "Age",
    "Flight Distance",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]


def prepare_features(df):
    """Return the 14 rating columns as the feature matrix X for K-Means."""
    missing = [c for c in RATING_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required rating columns: {missing}")
    X = df[RATING_COLS].apply(pd.to_numeric, errors="coerce")
    bad = [c for c in RATING_COLS if X[c].isna().any()]
    if bad:
        raise ValueError(f"Missing or non-numeric values in columns: {bad}")
    return X.astype(float)


def assign_clusters(df, scaler, kmeans):
    """Return (cluster labels, scaled feature matrix) for raw passenger rows."""
    X_scaled = scaler.transform(prepare_features(df))
    return kmeans.predict(X_scaled), X_scaled


def find_elbow(ks, inertias):
    """Elbow of the inertia curve: the point farthest from the straight line
    joining the first and the last point (same idea as KneeLocator)."""
    ks = np.asarray(list(ks), dtype=float)
    y = np.asarray(inertias, dtype=float)
    x_n = (ks - ks.min()) / (ks.max() - ks.min())
    y_n = (y - y.min()) / (y.max() - y.min())
    # line from (0, y_n[0]) to (1, y_n[-1]); distance of each point to that line
    dist = np.abs((y_n[-1] - y_n[0]) * x_n - (1.0) * (y_n - y_n[0])) / np.hypot(y_n[-1] - y_n[0], 1.0)
    return int(ks[int(np.argmax(dist))])


def cluster_traits(centers, top=3, cutoff=0.25):
    """Describe each cluster by the ratings that differ most from the average.

    `centers` is a DataFrame (clusters x RATING_COLS) of standardized centroids.
    """
    traits = {}
    for c in centers.index:
        z = centers.loc[c].sort_values(ascending=False)
        high = [f for f in z.index[:top] if z[f] > cutoff]
        low = [f for f in z.index[::-1][:top] if z[f] < -cutoff]
        parts = []
        if high:
            parts.append("Rated higher: " + ", ".join(high))
        if low:
            parts.append("Rated lower: " + ", ".join(low))
        traits[str(c)] = " | ".join(parts) if parts else "Close to the average passenger"
    return traits
