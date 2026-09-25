"""New-customer assignment and faithful, appropriately scoped explanations."""
import numpy as np
import pandas as pd
from .config import INCOME, SPENDING, FREQUENCY, MONETARY


def predict_customer(bundle, customer, model_name=None):
    name = model_name or bundle["best_model"]
    model = bundle["models"][name]
    estimator = model["estimator"]
    features = bundle["features"]
    frame = pd.DataFrame([customer])
    if not set(features).issubset(frame):
        raise ValueError(f"Prediction requires: {features}")
    for feature in features:
        try:
            value = float(frame.loc[0, feature])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{feature} must be numeric.") from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{feature} must be a finite, nonnegative number.")
        if feature == SPENDING and not 1 <= value <= 100:
            raise ValueError("Spending Score must be between 1 and 100.")
        if feature == "Age" and value > 120:
            raise ValueError("Age must be at most 120.")
        if feature in ("Recency", FREQUENCY) and value != int(value):
            raise ValueError(f"{feature} must be a whole number.")
    if FREQUENCY in features and ((customer[FREQUENCY] == 0) != (customer[MONETARY] == 0)):
        raise ValueError("Zero purchases require zero monetary value, and vice versa.")
    x = bundle["preprocessor"].transform(frame[features])[0]
    probability = None
    if name in ("K-Means", "Gaussian Mixture"):
        label = int(estimator.predict(x.reshape(1, -1))[0])
        method = "Native model prediction"
        if name == "Gaussian Mixture":
            probability = float(estimator.predict_proba(x.reshape(1, -1))[0, label])
            method = "Highest Gaussian Mixture posterior probability"
    elif name == "Agglomerative":
        nearest = int(np.argmin(np.sum((model["centroids"] - x) ** 2, axis=1)))
        label = model["groups"][nearest]
        method = "Nearest training centroid proxy; hierarchical clustering has no native predict method"
    else:
        core = estimator.components_
        if len(core):
            distances = np.linalg.norm(core - x, axis=1)
            nearest = int(np.argmin(distances))
            label = int(estimator.labels_[estimator.core_sample_indices_[nearest]]) if distances[nearest] <= estimator.eps else -1
        else:
            label = -1
        method = "Nearest DBSCAN core sample within eps; otherwise unassigned (extension rule)"
    warnings = [f"{feature} is outside the training range."
                for feature in features if frame.loc[0, feature] < bundle["feature_min"][feature] or frame.loc[0, feature] > bundle["feature_max"][feature]]
    if label == -1:
        return {"model": name, "cluster": -1, "category": "Unassigned / unusual behavior",
                "recommendation": "Review this customer before selecting a campaign.",
                "method": method, "probability": None, "warnings": warnings,
                "explanation": "No fitted DBSCAN core sample is within the configured radius.", "details": pd.DataFrame()}
    profile = model["profiles"].loc[label]
    # Similarity is descriptive, not a causal explanation or SHAP attribution.
    center = model["centroids"][model["groups"].index(label)]
    reference = center
    scope = "Descriptive similarity to the assigned cluster's transformed mean; not a causal explanation."
    if name == "K-Means":
        reference = estimator.cluster_centers_[label]
    elif name == "DBSCAN":
        reference = estimator.components_[nearest]
        scope = "Feature distances to the core sample used by the DBSCAN extension rule."
    details = pd.DataFrame({"Feature": features, "Customer value": frame[features].iloc[0].values,
                            "Cluster mean": profile[features].values,
                            "Standardized gap": np.abs(x - reference)})
    if name in ("K-Means", "Agglomerative") and len(model["groups"]) > 1:
        centers = estimator.cluster_centers_ if name == "K-Means" else model["centroids"]
        index = label if name == "K-Means" else model["groups"].index(label)
        distances = np.sum((centers - x) ** 2, axis=1)
        distances[index] = np.inf
        alternate = int(np.argmin(distances))
        details["Support vs next center"] = (x - centers[alternate]) ** 2 - (x - centers[index]) ** 2
        scope = "Positive support means this feature favors the assigned center over the next nearest center, in squared standardized distance."
    nearest_features = details.nsmallest(3, "Standardized gap").Feature.tolist()
    explanation = f"Closest feature matches: {', '.join(nearest_features)}. {scope}"
    if name == "Gaussian Mixture":
        explanation += " The actual GMM decision uses covariance and component weights, not nearest-mean distance."
    return {"model": name, "cluster": label, "category": profile.Category,
            "recommendation": profile.Recommendation, "method": method,
            "probability": probability, "warnings": warnings,
            "explanation": explanation, "details": details}
