"""CSV validation and transaction-derived RFM features."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from .config import AS_OF, CLASSIC_FEATURES, FEATURES, FREQUENCY, MONETARY, INCOME, SPENDING, RFM


def load_csv(source):
    try:
        return pd.read_csv(source)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError(f"Cannot read CSV: {exc}") from exc


def normalize_columns(data):
    data = data.copy()
    data.columns = data.columns.str.strip()
    data = data.rename(columns={"Annual Income (k$)": INCOME,
                                "Spending Score (1-100)": SPENDING,
                                "Frequency": FREQUENCY, "Monetary": MONETARY})
    if data.columns.duplicated().any():
        raise ValueError("Duplicate column names after normalizing aliases.")
    return data


def numeric(series, name, low=0, high=None, integer=False):
    values = pd.to_numeric(series, errors="coerce")
    if (series.notna() & values.isna()).any() or np.isinf(values.astype(float)).any():
        raise ValueError(f"{name} must contain finite numbers or missing values.")
    observed = values.dropna()
    if (observed < low).any() or (high is not None and (observed > high).any()):
        raise ValueError(f"{name} must be >= {low}" + (f" and <= {high}." if high is not None else "."))
    if integer and not np.allclose(observed, np.round(observed)):
        raise ValueError(f"{name} must contain whole numbers.")
    return values.astype(float)


def compute_rfm(customers, transactions, as_of=AS_OF):
    """Aggregate purchases in [as_of - 365 days, as_of], inclusive.

    Customers with no purchases in the window get frequency/monetary 0,
    missing last-purchase date and recency 366 (an explicit inactivity sentinel).
    """
    data = normalize_columns(customers)
    tx = transactions.copy()
    required = {"TransactionID", "CustomerID", "Purchase Date", "Amount"}
    if not required.issubset(tx):
        raise ValueError(f"Transactions require columns: {sorted(required)}")
    if "CustomerID" not in data or data.CustomerID.isna().any() or data.CustomerID.duplicated().any():
        raise ValueError("CustomerID must be present and unique.")
    if tx.TransactionID.isna().any() or tx.TransactionID.duplicated().any():
        raise ValueError("TransactionID must be present and unique.")
    if tx.CustomerID.isna().any() or not tx.CustomerID.isin(data.CustomerID).all():
        raise ValueError("Every transaction must refer to a known customer.")
    tx["Amount"] = numeric(tx.Amount, "Transaction Amount", low=.01)
    if tx.Amount.isna().any():
        raise ValueError("Transaction amounts cannot be missing.")
    try:
        dates = pd.to_datetime(tx["Purchase Date"], errors="raise", format="ISO8601").dt.normalize()
        reference = pd.Timestamp(as_of).normalize()
        if dates.isna().any() or (dates > reference).any():
            raise ValueError("Purchase dates cannot be missing or later than the analysis date.")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid transaction dates: {exc}") from exc
    tx["Purchase Date"] = dates
    tx = tx.loc[dates >= reference - pd.Timedelta(days=365)]
    aggregate = tx.groupby("CustomerID").agg(**{
        FREQUENCY: ("TransactionID", "count"), MONETARY: ("Amount", "sum"),
        "Last Purchase Date": ("Purchase Date", "max"),
    })
    data = data.drop(columns=[*RFM, "Last Purchase Date"], errors="ignore").join(aggregate, on="CustomerID")
    data[FREQUENCY] = data[FREQUENCY].fillna(0).astype(int)
    data[MONETARY] = data[MONETARY].fillna(0).round(2)
    data["Recency"] = (reference - data["Last Purchase Date"]).dt.days.fillna(366).astype(int)
    data["Last Purchase Date"] = data["Last Purchase Date"].dt.strftime("%Y-%m-%d")
    return data


def prepare_customers(raw, transactions=None, as_of=AS_OF, mode="advanced"):
    data = normalize_columns(raw)
    if mode not in ("advanced", "classic"):
        raise ValueError("Feature mode must be advanced or classic.")
    features = FEATURES if mode == "advanced" else CLASSIC_FEATURES
    required = ["CustomerID", "Gender", "Age", *CLASSIC_FEATURES]
    if not set(required).issubset(data):
        raise ValueError(f"Missing columns: {sorted(set(required) - set(data))}")
    if not 20 <= len(data) <= 10000:
        raise ValueError("Use 20 to 10,000 customers for the interactive comparison. The original smaller-data demo is in legacy/.")
    if data.CustomerID.isna().any() or data.CustomerID.duplicated().any():
        raise ValueError("CustomerID must be present and unique.")
    notes = []
    if transactions is not None:
        data = compute_rfm(data, transactions, as_of)
        notes.append("RFM recomputed from the purchase ledger in the 365-day analysis window.")
    elif mode == "advanced":
        if not set(RFM).issubset(data):
            raise ValueError("Advanced mode needs Recency, Purchase Frequency and Total Purchase Amount, or a transaction CSV. Use classic mode for the original five-column CSV.")
        notes.append("Using supplied RFM summary values; transaction reconciliation was not available.")
    bounds = {"Age": (0, 120), INCOME: (0, None), SPENDING: (1, 100),
              "Recency": (0, None), FREQUENCY: (0, None), MONETARY: (0, None),
              "Customer Satisfaction Score": (1, 5)}
    for column, (low, high) in bounds.items():
        if column not in data:
            continue
        data[column] = numeric(data[column], column, low, high, column in ("Recency", FREQUENCY))
        if data[column].isna().all() and column in features:
            raise ValueError(f"{column} cannot be entirely missing.")
        if data[column].isna().any():
            notes.append(f"{column}: {int(data[column].isna().sum())} missing measurements; model inputs use fitted medians.")
    if mode == "advanced":
        known = data[FREQUENCY].notna() & data[MONETARY].notna()
        if (((data[FREQUENCY] == 0) != (data[MONETARY] == 0)) & known).any():
            raise ValueError("Frequency and monetary value must both be zero for a customer without purchases.")
    if "Last Purchase Date" in data:
        try:
            last = pd.to_datetime(data["Last Purchase Date"], errors="raise", format="ISO8601")
            age = (pd.Timestamp(as_of).normalize() - last).dt.days
            if (age.dropna() < 0).any():
                raise ValueError("Last Purchase Date is later than the analysis date.")
            if "Recency" in data and ((age - data.Recency).abs().dropna() > 0).any():
                raise ValueError("Recency disagrees with Last Purchase Date and the analysis date.")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Invalid Last Purchase Date: {exc}") from exc
    data["Gender"] = data.Gender.fillna("Unknown").astype(str)
    if len(data[features].drop_duplicates()) < 3:
        raise ValueError("Need at least three distinct customer feature vectors.")
    return data.reset_index(drop=True), notes


def make_preprocessor(features):
    # Keep original feature order so explanations map to the correct columns.
    columns = ColumnTransformer([
        (f"f{i}", FunctionTransformer(np.log1p, feature_names_out="one-to-one") if name in RFM else "passthrough", [i])
        for i, name in enumerate(features)
    ])
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("log_rfm", columns), ("scaler", StandardScaler())])
