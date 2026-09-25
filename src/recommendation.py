"""Transparent marketing rules applied to learned cluster profiles.

    These are decision-support suggestions, not learned campaign effectiveness.
"""
import pandas as pd
from .config import INCOME, SPENDING, FREQUENCY, MONETARY


def recommendation_engine(profile, thresholds, cluster=0):
    if int(cluster) == -1:
        return "Unassigned / unusual behavior", "Review the customer profile before selecting a campaign."
    high_income = profile[INCOME] >= thresholds[INCOME]
    high_spending = profile[SPENDING] >= thresholds[SPENDING]
    if high_income and high_spending:
        category, action = "VIP segment", "Premium membership, luxury offers and early access."
    elif high_income:
        category, action = "Growth opportunity", "Special discounts and personalized product campaigns."
    elif high_spending:
        category, action = "Value-focused loyalists", "Budget-friendly offers, bundles and loyalty rewards."
    else:
        category, action = "Engagement segment", "Engagement campaigns and introductory value offers."
    if pd.notna(profile.get("Recency")) and profile["Recency"] > 90:
        category = "At-risk / " + category
        action += " Prioritize a win-back message because average recency exceeds 90 days."
    if pd.notna(profile.get(FREQUENCY)) and profile[FREQUENCY] >= thresholds.get(FREQUENCY, float("inf")):
        action += " Offer a repeat-purchase loyalty benefit."
    return category, action


def attach_recommendations(bundle):
    data = bundle["data"]
    # Freeze cohort thresholds with the model; predictions never recompute them.
    thresholds = {name: float(data[name].median()) for name in [INCOME, SPENDING, FREQUENCY, MONETARY] if name in data}
    bundle["thresholds"] = thresholds
    for model in bundle["models"].values():
        profiles = model["profiles"].copy()
        actions = [recommendation_engine(row, thresholds, cluster) for cluster, row in profiles.iterrows()]
        profiles["Category"] = [item[0] for item in actions]
        profiles["Recommendation"] = [item[1] for item in actions]
        model["profiles"] = profiles
    return bundle


def customer_results(bundle, model_name=None):
    model_name = model_name or bundle["best_model"]
    model = bundle["models"][model_name]
    data = bundle["data"].copy()
    data["Cluster"] = model["labels"]
    return data.join(model["profiles"][["Category", "Recommendation"]], on="Cluster")
