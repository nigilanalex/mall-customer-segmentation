"""Train and export the complete analytics project: python main.py."""
import argparse
import json
import sqlite3
import sys
from pathlib import Path
from src.config import BASE, CUSTOMERS, TRANSACTIONS, DATABASE, BUNDLE, AS_OF, ALGORITHMS
from src.data_generation import generate_data
from src.preprocessing import load_csv, prepare_customers
from src.database import store_dataset, load_customers, store_results
from src.workflow import analyze, save_bundle
from src.recommendation import customer_results
from src.visualization import save_charts
from src.prediction import predict_customer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CUSTOMERS)
    parser.add_argument("--transactions", type=Path, help="Purchase ledger; bundled ledger only used with bundled customers")
    parser.add_argument("--as-of", default=AS_OF, help="RFM reference date, YYYY-MM-DD")
    parser.add_argument("--mode", choices=["advanced", "classic"], default="advanced")
    parser.add_argument("--clusters", type=int, help="Override elbow-suggested k")
    parser.add_argument("--eps", type=float, default=0.75)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--output", type=Path, default=BASE / "outputs")
    parser.add_argument("--database", type=Path, default=DATABASE)
    parser.add_argument("--model-dir", type=Path, default=BASE / "models")
    parser.add_argument("--generate", action="store_true", help="Regenerate bundled synthetic data with 2,000 customers")
    args = parser.parse_args()
    try:
        if args.generate or not CUSTOMERS.exists():
            customers, transactions = generate_data(as_of=args.as_of)
            CUSTOMERS.parent.mkdir(exist_ok=True)
            customers.to_csv(CUSTOMERS, index=False)
            transactions.to_csv(TRANSACTIONS, index=False)
            (CUSTOMERS.parent / "provenance.json").write_text(json.dumps({"synthetic": True, "customers": 2000, "seed": 42, "as_of": args.as_of, "window_days": 365, "income_unit": "thousands of USD", "amount_unit": "USD"}, indent=2))
        raw = load_csv(args.data)
        tx_path = args.transactions or (TRANSACTIONS if args.data.resolve() == CUSTOMERS.resolve() and TRANSACTIONS.exists() else None)
        transactions = load_csv(tx_path) if tx_path else None
        print("\nDATASET INFORMATION")
        raw.info()
        print("\nMissing values:\n", raw.isna().sum())
        clean, notes = prepare_customers(raw, transactions, args.as_of, args.mode)
        dataset_id = store_dataset(clean, transactions, args.as_of, args.data.name, args.database)
        database_data = load_customers(dataset_id, args.database)
        bundle = analyze(database_data, as_of=args.as_of, mode=args.mode, k=args.clusters, eps=args.eps, min_samples=args.min_samples)
        bundle.update({"notes": notes, "raw_missing": raw.isna().sum(), "source": args.data.name, "dataset_id": dataset_id,
                       "transaction_count": 0 if transactions is None else len(transactions)})
        bundle["run_id"] = store_results(bundle, dataset_id, args.database)
        args.output.mkdir(parents=True, exist_ok=True)
        args.model_dir.mkdir(parents=True, exist_ok=True)
        save_bundle(bundle, args.model_dir / BUNDLE.name)
        for name in ALGORITHMS:
            artifact = {key: value for key, value in bundle.items() if key != "models"}
            artifact["models"] = {name: bundle["models"][name]}
            artifact["best_model"] = name
            save_bundle(artifact, args.model_dir / (name.lower().replace(" ", "_").replace("-", "_") + ".pkl"))
        results = customer_results(bundle)
        results.to_csv(args.output / "segmented_customers.csv", index=False)
        bundle["models"][bundle["best_model"]]["profiles"].to_csv(args.output / "cluster_centers.csv")
        bundle["comparison"].to_csv(args.output / "model_comparison.csv", index=False)
        bundle["elbow"].to_csv(args.output / "elbow_scores.csv", index=False)
        metrics = {key: bundle[key] for key in ["best_model", "mode", "features", "k", "suggested_k", "as_of", "eps", "min_samples"]}
        metrics.update({"customers": len(raw), "transactions": 0 if transactions is None else len(transactions)})
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        example = predict_customer(bundle, bundle["data"].iloc[0].to_dict())
        example["details"] = example["details"].to_dict("records")
        (args.output / "sample_prediction.json").write_text(json.dumps(example, indent=2), encoding="utf-8")
        save_charts(bundle, args.output)
        print("\n", bundle["comparison"].round(4).to_string(index=False))
        print(f"\nRecommended model: {bundle['best_model']}; selected k: {bundle['k']}; elbow suggestion: {bundle['suggested_k']}")
        print("\nCLUSTER PROFILES\n", bundle["models"][bundle["best_model"]]["profiles"].round(2).to_string())
        print("\nCUSTOMER ASSIGNMENTS\n", results[["CustomerID", "Cluster", "Category"]].to_string(index=False))
        print(f"\nSaved outputs to {args.output}; SQLite run: {bundle['run_id']}")
        return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
