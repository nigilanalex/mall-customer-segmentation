import streamlit as st
import plotly.express as px
from src.dashboard import context, chart, distribution, COLORS
from src.config import MONETARY, FREQUENCY

bundle, name, data, display, profiles = context()
st.caption("CUSTOMER INTELLIGENCE / OVERVIEW")
st.title("Home dashboard")
st.markdown("Explore purchasing behavior, compare customer segments, and turn profiles into actionable marketing suggestions.")
st.info(f"{st.session_state.source_label} • Analysis date: {bundle['as_of']} • Active model: {name}")
st.caption("The bundled dataset is synthetic. Uploaded customer data is analyzed as supplied.")
with st.container(horizontal=True):
    st.metric("Total customers", f"{len(data):,}", border=True)
    total_transactions = bundle.get("transaction_count")
    st.metric("Total transactions", f"{total_transactions:,}" if total_transactions is not None else "Not supplied", border=True)
    st.metric("Best model", bundle["best_model"], border=True)
    st.metric("Number of clusters", len([i for i in profiles.index if i != -1]), border=True)
    active_score = bundle["comparison"].set_index("Model").loc[name, "Silhouette"]
    st.metric("Silhouette score", f"{active_score:.3f}" if __import__("pandas").notna(active_score) else "Unavailable", border=True)
left, right = st.columns([1.25, 1])
with left, st.container(border=True):
    st.subheader("Your customer landscape")
    chart(distribution(data))
with right, st.container(border=True):
    st.subheader("From data to decisions")
    st.markdown("1. **Explore** customer and purchase patterns.\n2. **Compare** four clustering approaches.\n3. **Understand** profiles and assignment explanations.\n4. **Act** on segment-specific marketing suggestions.")
    st.caption("Recommendations use explicit rules. No campaigns are sent automatically.")
st.subheader("Segment snapshots")
for cluster, profile in profiles.head(3).iterrows():
    with st.container(border=True):
        st.markdown(f"**Segment {cluster} · {profile.Category}**")
        st.caption(f"{int(profile.Customers):,} customers")
        st.markdown(profile.Recommendation)
