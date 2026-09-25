"""SQLite persistence with immutable dataset/run IDs and parameterized inserts."""
import json
import sqlite3
from uuid import uuid4
import pandas as pd
from .config import DATABASE

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (dataset_id TEXT PRIMARY KEY, as_of TEXT NOT NULL, source TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS customers (dataset_id TEXT NOT NULL, customer_id TEXT NOT NULL, details TEXT NOT NULL,
 PRIMARY KEY(dataset_id, customer_id), FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id));
CREATE TABLE IF NOT EXISTS transactions (dataset_id TEXT NOT NULL, transaction_id TEXT NOT NULL, customer_id TEXT NOT NULL,
 purchase_date TEXT NOT NULL, amount REAL NOT NULL CHECK(amount>0), PRIMARY KEY(dataset_id, transaction_id),
 FOREIGN KEY(dataset_id, customer_id) REFERENCES customers(dataset_id, customer_id));
CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
 best_model TEXT NOT NULL, settings TEXT NOT NULL, FOREIGN KEY(dataset_id) REFERENCES datasets(dataset_id));
CREATE TABLE IF NOT EXISTS cluster_results (run_id TEXT NOT NULL, model TEXT NOT NULL, customer_id TEXT NOT NULL, cluster INTEGER NOT NULL,
 PRIMARY KEY(run_id, model, customer_id), FOREIGN KEY(run_id) REFERENCES runs(run_id));
CREATE TABLE IF NOT EXISTS recommendations (run_id TEXT NOT NULL, model TEXT NOT NULL, cluster INTEGER NOT NULL,
 category TEXT NOT NULL, recommendation TEXT NOT NULL, profile TEXT NOT NULL,
 PRIMARY KEY(run_id, model, cluster), FOREIGN KEY(run_id) REFERENCES runs(run_id));
"""


def connect(path=DATABASE):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(SCHEMA)
    return connection


def store_dataset(customers, transactions, as_of, source, path=DATABASE):
    dataset_id = uuid4().hex
    records = json.loads(customers.to_json(orient="records", date_format="iso"))
    connection = connect(path)
    try:
        with connection:
            connection.execute("INSERT INTO datasets VALUES(?,?,?)", (dataset_id, str(as_of), source))
            connection.executemany("INSERT INTO customers VALUES(?,?,?)", [(dataset_id, str(row["CustomerID"]), json.dumps(row)) for row in records])
            if transactions is not None:
                connection.executemany("INSERT INTO transactions VALUES(?,?,?,?,?)", [
                    (dataset_id, str(row["TransactionID"]), str(row["CustomerID"]), str(row["Purchase Date"]), float(row["Amount"]))
                    for row in transactions.to_dict("records")])
    finally:
        connection.close()
    return dataset_id


def load_customers(dataset_id, path=DATABASE):
    connection = connect(path)
    try:
        rows = connection.execute("SELECT details FROM customers WHERE dataset_id=? ORDER BY rowid", (dataset_id,)).fetchall()
        if not rows:
            raise ValueError("Dataset not found in SQLite.")
        return pd.DataFrame([json.loads(row[0]) for row in rows])
    finally:
        connection.close()


def store_results(bundle, dataset_id, path=DATABASE):
    run_id = uuid4().hex
    settings = {key: bundle[key] for key in ("mode", "k", "suggested_k", "eps", "min_samples", "features", "as_of")}
    connection = connect(path)
    try:
        with connection:
            connection.execute("INSERT INTO runs(run_id,dataset_id,best_model,settings) VALUES(?,?,?,?)", (run_id, dataset_id, bundle["best_model"], json.dumps(settings)))
            for name, model in bundle["models"].items():
                connection.executemany("INSERT INTO cluster_results VALUES(?,?,?,?)", [(run_id, name, str(customer), int(label)) for customer, label in zip(bundle["data"].CustomerID, model["labels"])])
                connection.executemany("INSERT INTO recommendations VALUES(?,?,?,?,?,?)", [(run_id, name, int(cluster), row.Category, row.Recommendation, row.to_json()) for cluster, row in model["profiles"].iterrows()])
    finally:
        connection.close()
    return run_id
