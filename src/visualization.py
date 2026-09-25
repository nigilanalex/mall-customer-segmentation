"""Static presentation charts and a portable interactive 3D figure."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from .config import INCOME, SPENDING, FREQUENCY, MONETARY
from .recommendation import customer_results


def save_charts(bundle, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="Set2")
    data = customer_results(bundle)
    centers = bundle["models"][bundle["best_model"]]["profiles"]
    elbow = bundle["elbow"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(elbow.k, elbow.Inertia, "o-", color="#0d9488")
    ax.axvline(bundle["k"], linestyle="--", color="#f59e0b", label=f"Selected k = {bundle['k']}")
    ax.set(title="K-Means elbow analysis", xlabel="Number of clusters", ylabel="Inertia in transformed feature space")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "elbow_curve.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x=INCOME, y=SPENDING, hue=data.Cluster.astype(str), alpha=.6, s=25, ax=ax)
    ax.scatter(centers[INCOME], centers[SPENDING], marker="X", s=180, c="black", label="Profile means")
    ax.set(title=f"{bundle['best_model']} | income/spending projection of {len(bundle['features'])}-feature clusters", xlabel="Annual income (k$)")
    ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output / "customer_clusters.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, feature in zip(axes.flat, bundle["features"]):
        sns.histplot(data[feature], bins=25, ax=ax, color="#0d9488")
    for ax in list(axes.flat)[len(bundle["features"]):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "customer_eda.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, direction in zip(axes, ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz"], ["higher", "lower", "higher"]):
        values = bundle["comparison"]
        ax.bar(values.Model, values[metric], color=["#0d9488", "#6366f1", "#f59e0b", "#ec4899"])
        ax.set_title(f"{metric} ({direction} is better)")
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle("Internal model comparison; DBSCAN metrics exclude noise")
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    data.Cluster.value_counts().sort_index().plot.bar(ax=ax, color="#0d9488")
    ax.set(title="Customers per cluster (-1 means noise)", ylabel="Customers")
    fig.tight_layout()
    fig.savefig(output / "cluster_distribution.png", dpi=160)
    plt.close(fig)
    if all(feature in data for feature in ["Recency", FREQUENCY, MONETARY]):
        chart = px.scatter_3d(data.assign(Cluster=data.Cluster.astype(str)), x="Recency", y=FREQUENCY, z=MONETARY, color="Cluster", hover_data=["CustomerID", "Category"], title="RFM customer segments")
        chart.write_html(output / "customer_3d.html", include_plotlyjs=True)
