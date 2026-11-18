"""PCA, K-means, and a DBSCAN check on the same principal-component space."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors

RANDOM_STATE = 42


def fit_pca(scaled: pd.DataFrame, variance: float = 0.85) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=variance, svd_solver="full", random_state=RANDOM_STATE)
    scores = pca.fit_transform(scaled.values)
    return scores, pca


def fit_pca_2d(scaled: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pca = PCA(n_components=2, svd_solver="full", random_state=RANDOM_STATE)
    scores = pca.fit_transform(scaled.values)
    return scores, pca.explained_variance_ratio_


def _kmeans(scores: np.ndarray, k: int) -> KMeans:
    model = KMeans(n_clusters=k, n_init=50, random_state=RANDOM_STATE)
    model.fit(scores)
    return model


def select_k(
    scores: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    min_size: int = 15,
) -> tuple[int, np.ndarray, pd.DataFrame]:
    """Highest silhouette on the PCA scores, requiring every cluster to have ``min_size`` patients."""
    rows = []
    fitted: dict[int, np.ndarray] = {}
    for k in range(k_min, k_max + 1):
        model = _kmeans(scores, k)
        labels = model.labels_.astype(int)
        sizes = np.bincount(labels, minlength=k)
        sil = float(silhouette_score(scores, labels))
        fitted[k] = labels
        rows.append(
            {
                "k": k,
                "silhouette": sil,
                "inertia": float(model.inertia_),
                "min_size": int(sizes.min()),
                "max_size": int(sizes.max()),
                "eligible": bool(sizes.min() >= min_size),
            }
        )
    table = pd.DataFrame(rows)
    eligible = table.loc[table["eligible"]].sort_values(
        ["silhouette", "k"], ascending=[False, True]
    )
    if eligible.empty:
        chosen = int(table.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    else:
        chosen = int(eligible.iloc[0]["k"])
    return chosen, fitted[chosen], table


def _knee_eps(k_distance: np.ndarray) -> tuple[float, int]:
    """Epsilon from the maximum second derivative of the sorted k-distance curve."""
    y = np.asarray(k_distance, dtype=float)
    span = y.max() - y.min()
    y_n = (y - y.min()) / (span + 1e-12)
    second = np.gradient(np.gradient(y_n))
    lo = max(1, int(0.50 * len(y)))
    hi = max(lo + 1, len(y) - 2)
    idx = lo + int(np.argmax(second[lo:hi]))
    eps = float(y[idx])
    if eps <= 0:
        eps = float(np.median(y[y > 0])) if np.any(y > 0) else 1.0
    return eps, int(idx)


def run_dbscan(scores: np.ndarray, kmeans_labels: np.ndarray) -> dict:
    n_pcs = int(scores.shape[1])
    min_samples = int(2 * n_pcs)
    if min_samples >= len(scores):
        min_samples = max(2, len(scores) // 10)
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors.fit(scores)
    distances, _ = neighbors.kneighbors(scores)
    k_distance = np.sort(distances[:, -1])
    eps, knee_index = _knee_eps(k_distance)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(scores)
    n_noise = int((labels == -1).sum())
    n_clusters = int(len(set(labels) - {-1}))
    ari = float(adjusted_rand_score(kmeans_labels, labels))
    return {
        "eps": eps,
        "knee_index": knee_index,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": n_noise / len(labels),
        "ari_vs_kmeans": ari,
        "labels": labels,
    }


def _median_survival(time: pd.Series, event: pd.Series) -> float:
    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter()
    kmf.fit(time, event)
    median = float(kmf.median_survival_time_)
    return median


def profile_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Observed (not imputed) summaries of each K-means group."""
    work = df.copy()
    work["cluster"] = labels
    rows = []
    for cluster, part in work.groupby("cluster"):
        time = part["survival_time"]
        event = part["survival_status"].astype(int)
        rows.append(
            {
                "cluster": int(cluster),
                "n": int(len(part)),
                "events": int(event.sum()),
                "event_rate": float(event.mean()),
                "median_survival_days": _median_survival(time, event),
                "Recipientage_mean": float(part["Recipientage"].mean()),
                "Donorage_mean": float(part["Donorage"].mean()),
                "CD34_median": float(part["CD34kgx10d6"].median()),
                "log_CD34_mean": float(part["log_CD34"].mean()),
                "pct_high_risk": float(part["Riskgroup"].mean()),
                "pct_malignant": float(part["Diseasegroup"].mean()),
                "pct_pbsc": float(part["Stemcellsource"].mean()),
                "pct_txpostrelapse": float(part["Txpostrelapse"].mean()),
                "pct_gender_mismatch": float(part["Gendermatch"].mean()),
                "pct_abo_match": float(part["ABOmatch"].mean(skipna=True)),
                "HLAmatch_mean": float(part["HLAmatch"].mean()),
                "CMVstatus_mean": float(part["CMVstatus"].mean(skipna=True)),
                "pct_ALL": float((part["Disease"] == "ALL").mean()),
                "pct_AML": float((part["Disease"] == "AML").mean()),
                "pct_chronic": float((part["Disease"] == "chronic").mean()),
                "pct_nonmalignant": float((part["Disease"] == "nonmalignant").mean()),
                "pct_lymphoma": float((part["Disease"] == "lymphoma").mean()),
            }
        )
    profile = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    profile["name"] = name_clusters(profile)
    return profile


def name_clusters(profile: pd.DataFrame) -> list[str]:
    """Short names: dominant diagnosis, then the trait that most separates the cluster."""
    disease_cols = [
        ("pct_ALL", "ALL"),
        ("pct_AML", "AML"),
        ("pct_chronic", "chronic"),
        ("pct_nonmalignant", "nonmalignant"),
        ("pct_lymphoma", "lymphoma"),
    ]
    specs = [
        ("pct_pbsc", "mostly PBSC", "mostly marrow"),
        ("log_CD34_mean", "higher CD34", "lower CD34"),
        ("pct_high_risk", "high-risk", "low-risk"),
        ("pct_txpostrelapse", "often post-relapse", "rarely post-relapse"),
        ("Recipientage_mean", "older recipients", "younger recipients"),
        ("HLAmatch_mean", "poorer HLA match", "better HLA match"),
    ]
    scores: dict[str, np.ndarray] = {}
    for column, _high, _low in specs:
        center = float(profile[column].mean())
        scale = float(profile[column].std(ddof=0))
        if scale == 0 or np.isnan(scale):
            scores[column] = np.zeros(len(profile))
        else:
            scores[column] = ((profile[column] - center) / scale).to_numpy()
    names: list[str] = []
    for loc, row in enumerate(profile.itertuples(index=False)):
        share, disease = max((float(getattr(row, col)), label) for col, label in disease_cols)
        parts: list[str] = [disease] if share >= 0.70 else []
        ranked = sorted(specs, key=lambda spec: abs(scores[spec[0]][loc]), reverse=True)
        for column, high, low in ranked:
            trait = high if scores[column][loc] >= 0 else low
            if trait not in parts:
                parts.append(trait)
            if len(parts) >= 2:
                break
        names.append(", ".join(parts))
    return names
