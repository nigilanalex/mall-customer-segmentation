"""Generate demo customers and an internally consistent purchase ledger."""
import numpy as np
import pandas as pd
from .config import AS_OF, SEED, INCOME, SPENDING, FREQUENCY, MONETARY


def generate_data(records=2000, as_of=AS_OF, seed=SEED):
    """Return synthetic data; behavior overlaps and does not encode true labels.

    Income is in thousands of dollars; transaction amounts are in dollars.
    Frequency and monetary value cover a fixed trailing 365-day window.
    """
    if records < 20:
        raise ValueError("Generate at least 20 customers.")
    rng = np.random.default_rng(seed)
    # Income, spending, typical frequency, typical recency, basket size, age.
    prototypes = np.array([
        [110, 84, 32, 12, 210, 42], [100, 28, 6, 95, 105, 48],
        [32, 78, 23, 23, 45, 28], [30, 22, 4, 150, 30, 36],
        [62, 53, 14, 40, 90, 39],
    ])
    groups = rng.choice(len(prototypes), records, p=[.18, .18, .22, .18, .24])
    sample = prototypes[groups]
    customers = pd.DataFrame({
        "CustomerID": np.arange(1, records + 1),
        "Gender": rng.choice(["Female", "Male", "Other"], records, p=[.49, .49, .02]),
        "Age": np.clip(np.rint(rng.normal(sample[:, 5], 10)), 18, 80).astype(int),
        INCOME: np.round(np.clip(rng.normal(sample[:, 0], 14), 12, 200), 2),
        SPENDING: np.clip(np.rint(rng.normal(sample[:, 1], 12)), 1, 100).astype(int),
        "Customer Satisfaction Score": np.round(np.clip(rng.normal(3.6, .85, records), 1, 5), 1),
    })
    reference = pd.Timestamp(as_of).normalize()
    ledger = []
    for i, row in customers.iterrows():
        frequency = max(1, int(rng.poisson(sample[i, 2])))
        recency = int(np.clip(rng.gamma(2, sample[i, 3] / 2), 1, 330))
        # Include the latest purchase explicitly, making recency reproducible.
        offsets = np.r_[recency, rng.integers(recency, 365, frequency - 1)]
        amounts = np.round(rng.lognormal(np.log(sample[i, 4]), .45, frequency), 2)
        for offset, amount in zip(offsets, amounts):
            ledger.append((len(ledger) + 1, int(row.CustomerID),
                           (reference - pd.Timedelta(days=int(offset))).strftime("%Y-%m-%d"), float(amount)))
    transactions = pd.DataFrame(ledger, columns=["TransactionID", "CustomerID", "Purchase Date", "Amount"])
    from .preprocessing import compute_rfm
    customers = compute_rfm(customers, transactions, as_of)
    columns = ["CustomerID", "Gender", "Age", INCOME, SPENDING, FREQUENCY,
               MONETARY, "Last Purchase Date", "Recency", "Customer Satisfaction Score"]
    return customers[columns], transactions
