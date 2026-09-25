"""Shared paths, units, and reproducible settings."""
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SEED = 42
AS_OF = "2026-09-25"
CUSTOMERS = BASE / "dataset" / "Mall_Customers.csv"
TRANSACTIONS = BASE / "dataset" / "Transactions.csv"
DATABASE = BASE / "database" / "customers.db"
BUNDLE = BASE / "models" / "analytics_bundle.pkl"
INCOME = "Annual Income"
SPENDING = "Spending Score"
FREQUENCY = "Purchase Frequency"
MONETARY = "Total Purchase Amount"
CLASSIC_FEATURES = [INCOME, SPENDING]
FEATURES = [INCOME, SPENDING, "Age", "Recency", FREQUENCY, MONETARY]
RFM = ["Recency", FREQUENCY, MONETARY]
ALGORITHMS = ["K-Means", "Agglomerative", "DBSCAN", "Gaussian Mixture"]
