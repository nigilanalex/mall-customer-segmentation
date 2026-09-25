"""Reproducible customer segmentation. Run: python main.py --help."""
from pathlib import Path
import argparse
import json
import pickle
import sys

import matplotlib
matplotlib.use("Agg")  # Save charts without requiring a desktop window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
DATASET = BASE / "dataset" / "Mall_Customers.csv"
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
COLUMNS = ["CustomerID", "Gender", "Age", *FEATURES]
SEED = 42


def generate_dataset(path=DATASET, records=500):
    """Create clearly synthetic demo data, without overwriting an existing file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    rng = np.random.default_rng(SEED)
    prototypes = np.array([[25, 22], [25, 80], [60, 50], [95, 20], [95, 82]])
    groups = np.arange(records) % len(prototypes)
    rng.shuffle(groups)
    points = prototypes[groups] + rng.normal(0, [7, 7], size=(records, 2))
    data = pd.DataFrame({
        "CustomerID": np.arange(1, records + 1),
        "Gender": rng.choice(["Female", "Male"], records),
        "Age": rng.integers(18, 71, records),
        FEATURES[0]: np.round(np.clip(points[:, 0], 10, 150), 1),
        FEATURES[1]: np.rint(np.clip(points[:, 1], 1, 100)).astype(int),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return data


def load_data(source):
    """Accept a file path or a Streamlit uploaded CSV."""
    try:
        return pd.read_csv(source)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Cannot read CSV: {exc}") from exc


def preprocess(data):
    """Validate the schema and median-impute missing numeric measurements."""
    data = data.copy()
    data.columns = data.columns.str.strip()
    if data.columns.duplicated().any():
        raise ValueError("Column names must be unique.")
    missing = sorted(set(COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if len(data) < 3:
        raise ValueError("Provide at least three customers.")
    if data.CustomerID.isna().any() or data.CustomerID.duplicated().any():
        raise ValueError("CustomerID must be present and unique for every row.")
    notes = []
    for column in ["Age", *FEATURES]:
        original = data[column]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any():
            raise ValueError(f"{column} contains nonnumeric values.")
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError(f"{column} contains infinite values.")
        if numeric.notna().sum() == 0:
            raise ValueError(f"{column} has no observed numeric values.")
        if column == FEATURES[1] and not numeric.dropna().between(1, 100).all():
            raise ValueError("Spending scores must be between 1 and 100.")
        if column != FEATURES[1] and (numeric.dropna() < 0).any():
            raise ValueError(f"{column} cannot be negative.")
        if numeric.isna().any():
            notes.append(f"{column}: filled {numeric.isna().sum()} missing values with median {numeric.median():.2f}.")
        data[column] = numeric.fillna(numeric.median())
    data["Gender"] = data["Gender"].fillna("Unknown")
    if len(data[FEATURES].drop_duplicates()) < 3:
        raise ValueError("Provide at least three distinct income/spending pairs.")
    return data, notes


def elbow_analysis(data):
    """Estimate a knee by maximum distance below the normalized endpoint line.

    This is a heuristic suggestion, not proof of an optimal cluster count.
    """
    scaled = StandardScaler().fit_transform(data[FEATURES])
    maximum = min(10, len(data) - 1, len(data[FEATURES].drop_duplicates()))
    ks = np.arange(1, maximum + 1)
    inertias = [KMeans(n_clusters=int(k), n_init=10, random_state=SEED).fit(scaled).inertia_ for k in ks]
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (np.array(inertias) - inertias[-1]) / (inertias[0] - inertias[-1])
    suggested = int(ks[np.argmax((1 - x) - y)])
    suggested = max(2, suggested)
    return pd.DataFrame({"k": ks, "Inertia": inertias}), suggested


def train_model(data, k):
    """Scale the two features and fit K-Means; return centers in original units."""
    maximum = min(len(data) - 1, len(data[FEATURES].drop_duplicates()))
    if not 2 <= k <= maximum:
        raise ValueError(f"Clusters must be between 2 and {maximum}.")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=SEED)),
    ])
    labeled = data.copy()
    labeled["Cluster"] = pipeline.fit_predict(data[FEATURES])
    centers = pd.DataFrame(pipeline["scaler"].inverse_transform(pipeline["kmeans"].cluster_centers_), columns=FEATURES)
    centers.index.name = "Cluster"
    scaled = pipeline["scaler"].transform(data[FEATURES])
    # Bound silhouette computation for larger uploads.
    score = silhouette_score(scaled, labeled.Cluster, sample_size=min(5000, len(data)), random_state=SEED)
    return pipeline, labeled, centers, float(score)


def save_charts(data, elbow, centers, k, output):
    """Save EDA, elbow, and labeled scatter plots as presentation-ready PNGs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, column in zip(axes, ["Age", *FEATURES]):
        sns.histplot(data[column], bins=20, ax=ax, color="#2563eb")
    fig.suptitle("Customer data: feature distributions")
    fig.tight_layout()
    fig.savefig(output / "customer_eda.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(elbow.k, elbow.Inertia, "o-", color="#2563eb")
    ax.axvline(k, color="#ea580c", linestyle="--", label=f"Selected k = {k}")
    ax.set(xlabel="Number of clusters (k)", ylabel="Within-cluster sum of squares (scaled units)", title="Elbow method")
    ax.set_xticks(elbow.k)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "elbow_curve.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(data=data, x=FEATURES[0], y=FEATURES[1], hue="Cluster", palette="tab10", alpha=.75, ax=ax)
    ax.scatter(centers[FEATURES[0]], centers[FEATURES[1]], marker="X", s=220, c="black", edgecolors="white", label="Centers")
    ax.set_title(f"Mall customer segmentation | k = {k}")
    ax.legend(title="Cluster")
    fig.tight_layout()
    fig.savefig(output / "customer_clusters.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATASET, help="Input CSV path")
    parser.add_argument("--clusters", type=int, help="Override the elbow suggestion")
    parser.add_argument("--output", type=Path, default=BASE / "outputs")
    args = parser.parse_args()
    try:
        if args.data == DATASET and not DATASET.exists():
            generate_dataset()
            print("Generated 500 SYNTHETIC demonstration customers.")
        raw = load_data(args.data)
        print("\nDATASET INFORMATION")
        raw.info()
        print("\nMissing values before preprocessing:\n", raw.isna().sum())
        data, notes = preprocess(raw)
        print("\nPreprocessing:", notes or "No missing numeric measurements.")
        print("\nDescriptive statistics:\n", data.describe().round(2))
        elbow, suggested = elbow_analysis(data)
        k = args.clusters if args.clusters is not None else suggested
        model, labeled, centers, score = train_model(data, k)
        print(f"\nElbow suggestion: {suggested}; number of clusters: {k}")
        print(f"Silhouette score: {score:.3f} (not classification accuracy)")
        print("\nCUSTOMER CLUSTER ASSIGNMENTS\n", labeled[["CustomerID", "Cluster"]].to_string(index=False))
        print("\nCLUSTER CENTERS (original units)\n", centers.round(2))
        print("\nCustomers per cluster:\n", labeled.Cluster.value_counts().sort_index())
        save_charts(labeled, elbow, centers, k, args.output)
        labeled.to_csv(args.output / "segmented_customers.csv", index=False)
        centers.to_csv(args.output / "cluster_centers.csv")
        elbow.to_csv(args.output / "elbow_scores.csv", index=False)
        metadata = {"rows": len(data), "features": FEATURES, "suggested_k": suggested, "selected_k": k, "silhouette_score": score, "random_state": SEED, "input": str(args.data.resolve())}
        (args.output / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        model_dir = BASE / "models"
        model_dir.mkdir(exist_ok=True)
        with (model_dir / "kmeans_pipeline.pkl").open("wb") as file:
            pickle.dump(model, file)
        print(f"\nSaved charts, tables, and metrics to {args.output.resolve()}")
        return 0
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
