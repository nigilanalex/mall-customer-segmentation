"""Fit four algorithms in the same feature space and compare valid partitions."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from .config import SEED, FEATURES, CLASSIC_FEATURES
from .preprocessing import make_preprocessor


def evaluate_clustering(x, labels):
    """Noise (-1) is excluded from all three metrics, with coverage reported.

    Silhouette uses all assigned rows up to 2,000; larger sets use one seeded
    sample. DBI and CH use all assigned rows. Invalid partitions return NaN.
    """
    labels = np.asarray(labels)
    assigned = labels != -1
    count = len(np.unique(labels[assigned]))
    result = {"Clusters": count, "Coverage": float(assigned.mean()),
              "Noise customers": int((~assigned).sum()), "Silhouette": np.nan,
              "Davies-Bouldin": np.nan, "Calinski-Harabasz": np.nan, "Status": "Invalid partition"}
    if not 2 <= count < assigned.sum():
        return result
    points, groups = x[assigned], labels[assigned]
    try:
        result.update({"Silhouette": float(silhouette_score(points, groups, sample_size=min(2000, len(points)), random_state=SEED)),
                       "Davies-Bouldin": float(davies_bouldin_score(points, groups)),
                       "Calinski-Harabasz": float(calinski_harabasz_score(points, groups)),
                       "Status": "Valid"})
    except ValueError as exc:
        result["Status"] = f"Metrics unavailable: {exc}"
    return result


def rank_models(table):
    """Equal metric ranks, plus a coverage penalty; lower rank score is better.

    Require 80% coverage to prevent winning by excluding most difficult cases.
    This is an internal comparison, not held-out performance or business lift.
    """
    table = table.copy()
    metrics = ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"]
    table["Eligible"] = table[metrics].notna().all(axis=1) & (table.Coverage >= .8)
    table["Rank score"] = np.nan
    valid = table.loc[table.Eligible, metrics]
    if valid.empty:
        raise ValueError("No model produced a valid partition with at least 80% coverage. Try another k or better data.")
    ranks = pd.concat([valid.Silhouette.rank(ascending=False),
                       valid["Davies-Bouldin"].rank(),
                       valid["Calinski-Harabasz"].rank(ascending=False)], axis=1).mean(axis=1)
    table.loc[ranks.index, "Rank score"] = ranks + (1 - table.loc[ranks.index, "Coverage"]) * len(valid)
    ordered = table.loc[table.Eligible].sort_values(["Rank score", "Silhouette", "Model"], ascending=[True, False, True])
    return table, ordered.iloc[0].Model


def elbow_analysis(x):
    maximum = min(10, len(x) - 1, len(np.unique(x, axis=0)))
    rows = []
    for k in range(1, maximum + 1):
        fit = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(x)
        rows.append({"k": k, "Inertia": float(fit.inertia_)})
    table = pd.DataFrame(rows)
    normalized_x = (table.k - 1) / (maximum - 1)
    normalized_y = (table.Inertia - table.Inertia.iloc[-1]) / (table.Inertia.iloc[0] - table.Inertia.iloc[-1])
    suggested = max(2, int(table.loc[((1 - normalized_x) - normalized_y).idxmax(), "k"]))
    return table, suggested


def compare_models(data, mode="advanced", k=None, eps=0.75, min_samples=10):
    features = FEATURES if mode == "advanced" else CLASSIC_FEATURES
    preprocessor = make_preprocessor(features)
    x = preprocessor.fit_transform(data[features])
    if len(np.unique(x, axis=0)) < 3:
        raise ValueError("Need three distinct feature vectors after imputation.")
    elbow, suggested = elbow_analysis(x)
    k = suggested if k is None else int(k)
    if not 2 <= k <= int(elbow.k.max()):
        raise ValueError(f"k must be between 2 and {int(elbow.k.max())}.")
    if not np.isfinite(eps) or eps <= 0 or not 2 <= min_samples <= len(data):
        raise ValueError("DBSCAN requires eps > 0 and min_samples between 2 and the customer count.")
    candidates = {
        "K-Means": KMeans(n_clusters=k, n_init=10, random_state=SEED),
        "Agglomerative": AgglomerativeClustering(n_clusters=k, linkage="ward"),
        "DBSCAN": DBSCAN(eps=eps, min_samples=min_samples),
        "Gaussian Mixture": GaussianMixture(n_components=k, n_init=3, reg_covar=1e-5, random_state=SEED),
    }
    models, reports = {}, []
    # Persist imputed original-unit profiles, separate from transformed model centers.
    filled = data.copy()
    filled[features] = preprocessor["imputer"].transform(data[features])
    for name, estimator in candidates.items():
        labels = estimator.fit_predict(x)
        metrics = evaluate_clustering(x, labels)
        if name == "Gaussian Mixture" and not estimator.converged_:
            metrics.update({"Silhouette": np.nan, "Status": "GMM did not converge"})
        reports.append({"Model": name, **metrics})
        groups = sorted(int(v) for v in np.unique(labels) if v != -1)
        centroids = np.array([x[labels == group].mean(axis=0) for group in groups])
        profiles = filled.assign(Cluster=labels).groupby("Cluster")[features].mean()
        profiles["Customers"] = pd.Series(labels).value_counts().reindex(profiles.index).values
        models[name] = {"estimator": estimator, "labels": labels, "groups": groups,
                        "centroids": centroids, "profiles": profiles}
    comparison, best = rank_models(pd.DataFrame(reports))
    return {"schema_version": 1, "data": filled, "features": features, "mode": mode,
            "preprocessor": preprocessor, "models": models, "comparison": comparison,
            "best_model": best, "elbow": elbow, "suggested_k": suggested, "k": k,
            "eps": eps, "min_samples": min_samples,
            "feature_min": filled[features].min(), "feature_max": filled[features].max()}
