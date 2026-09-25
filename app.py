"""Six-page customer analytics dashboard: python -m streamlit run app.py."""
import sqlite3
import streamlit as st
from src.config import BUNDLE, CUSTOMERS, TRANSACTIONS, AS_OF, ALGORITHMS
from src.preprocessing import load_csv, prepare_customers
from src.workflow import analyze, load_bundle
from src.database import store_dataset, load_customers, store_results
from src.auth import require_login, revoke_session

st.set_page_config(page_title="Customer intelligence | AI segmentation", page_icon=":material/insights:", layout="wide")
user = require_login()


@st.cache_data(max_entries=2, show_spinner="Loading customer analytics...")
def default_analysis(model_timestamp):
    if BUNDLE.exists():
        return load_bundle(BUNDLE)
    return analyze(load_csv(CUSTOMERS), load_csv(TRANSACTIONS) if TRANSACTIONS.exists() else None)


try:
    if "analytics" not in st.session_state:
        st.session_state.analytics = default_analysis(BUNDLE.stat().st_mtime_ns if BUNDLE.exists() else 0)
        saved_source = st.session_state.analytics.get("source", "customer dataset")
        st.session_state.source_label = f"Saved analysis: {saved_source}"
    with st.sidebar:
        st.title("Customer intelligence")
        st.caption("AI SEGMENTATION & MARKETING")
        st.markdown(f"**{user['username']}** · {user['role']}")
        if st.button("Log out", icon=":material/logout:", width="stretch"):
            revoke_session(st.session_state.get("auth_token"))
            st.session_state.clear()
            st.rerun()
        with st.expander("Data and model settings"):
            with st.form("analysis_settings"):
                customer_upload = st.file_uploader("Customer CSV", type="csv")
                transaction_upload = st.file_uploader("Transaction CSV (optional)", type="csv")
                mode = st.selectbox("Feature mode", ["advanced", "classic"])
                date = st.date_input("Analysis date", value=__import__("datetime").date.fromisoformat(AS_OF))
                k = st.number_input("Clusters (0 = elbow suggestion)", 0, 10, 0)
                eps = st.number_input("DBSCAN radius", .05, 10.0, .75, .05)
                minimum = st.number_input("DBSCAN minimum samples", 2, 100, 10)
                st.caption("Running analysis stores these customer records and results in local SQLite. Uploaded CSVs do not replace the bundled files.")
                run = st.form_submit_button("Run analysis", type="primary")
            if run:
                with st.spinner("Preparing data and comparing four models..."):
                    raw = load_csv(customer_upload if customer_upload is not None else CUSTOMERS)
                    tx = load_csv(transaction_upload) if transaction_upload is not None else (load_csv(TRANSACTIONS) if customer_upload is None and TRANSACTIONS.exists() else None)
                    prepared, notes = prepare_customers(raw, tx, str(date), mode)
                    dataset_id = store_dataset(prepared, tx, date, customer_upload.name if customer_upload is not None else CUSTOMERS.name)
                    bundle = analyze(load_customers(dataset_id), as_of=str(date), mode=mode, k=k or None, eps=eps, min_samples=minimum)
                    bundle.update({"notes": notes, "raw_missing": raw.isna().sum(), "dataset_id": dataset_id,
                                   "transaction_count": len(tx) if tx is not None else None})
                    bundle["run_id"] = store_results(bundle, dataset_id)
                    st.session_state.analytics = bundle
                    st.session_state.source_label = customer_upload.name if customer_upload is not None else "Bundled synthetic data"
                    st.session_state.active_model = bundle["best_model"]
                    st.session_state.pop("prediction", None)
        bundle = st.session_state.analytics
        st.session_state.setdefault("active_model", bundle["best_model"])
        st.selectbox("Active model", ALGORITHMS, key="active_model")
        st.caption(f"Source: {st.session_state.source_label}")
        st.caption(f"{len(bundle['data']):,} customers | {bundle['mode']} features")
    page = st.navigation([
        st.Page("app_pages/home.py", title="Home dashboard", icon=":material/home:", default=True),
        st.Page("app_pages/dataset_analysis.py", title="Customer analysis", icon=":material/table_chart:"),
        st.Page("app_pages/customer_segmentation.py", title="Clustering results", icon=":material/scatter_plot:"),
        st.Page("app_pages/model_comparison.py", title="Model comparison", icon=":material/leaderboard:"),
        st.Page("app_pages/customer_prediction.py", title="Customer prediction", icon=":material/person_search:"),
        st.Page("app_pages/marketing_recommendations.py", title="Marketing recommendations", icon=":material/campaign:"),
    ])
    page.run()
except (ValueError, OSError, sqlite3.Error) as exc:
    st.error(str(exc))
    st.info("Check your CSV columns, numeric ranges and analysis date. Run python main.py to rebuild bundled model artifacts if needed.")
