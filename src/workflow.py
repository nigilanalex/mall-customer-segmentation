"""Shared analysis workflow and artifact persistence."""
import pickle
from pathlib import Path
from .config import AS_OF
from .preprocessing import prepare_customers
from .clustering import compare_models
from .recommendation import attach_recommendations


def analyze(customers, transactions=None, as_of=AS_OF, mode="advanced", k=None, eps=0.75, min_samples=10):
    prepared, notes = prepare_customers(customers, transactions, as_of, mode)
    bundle = compare_models(prepared, mode, k, eps, min_samples)
    bundle.update({"as_of": str(as_of), "notes": notes, "raw_missing": customers.isna().sum()})
    return attach_recommendations(bundle)


def save_bundle(bundle, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(bundle, stream)
    temporary.replace(path)


def load_bundle(path):
    """Load only trusted local artifacts, never uploaded pickle files."""
    with Path(path).open("rb") as stream:
        bundle = pickle.load(stream)
    if bundle.get("schema_version") != 1:
        raise ValueError("Unsupported artifact version; rerun main.py.")
    return bundle
