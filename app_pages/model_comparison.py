import streamlit as st
import plotly.express as px
from src.dashboard import context, chart, COLORS

bundle, name, data, display, profiles = context()
st.caption("CUSTOMER INTELLIGENCE / MODEL BENCHMARK")
st.title("Model comparison")
st.success(f"Recommended for this run: {bundle['best_model']}")
st.markdown("All four models use the same fitted preprocessing. The recommendation combines three internal clustering metrics; it does not measure campaign effectiveness.")
table = bundle["comparison"]
st.dataframe(table.round(4), hide_index=True)
metric = st.selectbox("Compare a metric", ["Silhouette", "Davies-Bouldin", "Calinski-Harabasz", "Coverage"], key="comparison_metric")
chart(px.bar(table, x="Model", y=metric, color="Model", color_discrete_sequence=COLORS, title=f"{metric} by model"))
st.markdown("**Higher is better:** Silhouette and Calinski–Harabasz. **Lower is better:** Davies–Bouldin and the combined rank score.")
with st.expander("How the recommendation is selected", expanded=True):
    st.markdown("Each valid model receives a rank for each metric. Their average is combined with a penalty for unassigned customers. The lowest score wins; ties use silhouette, then model name. A model needs at least two clusters and 80% coverage to qualify.")
    st.markdown("DBSCAN noise (`-1`) is excluded from metric calculations. Coverage and noise counts remain visible because rejecting difficult customers can inflate scores. Silhouette uses up to 2,000 assigned customers; the other metrics use all assigned rows.")
    st.caption("This compares the current settings, not every possible hyperparameter. K-Means, Ward and GMM share the selected k; DBSCAN uses the sidebar radius and minimum samples.")
st.download_button("Download model comparison", table.to_csv(index=False), "model_comparison.csv", "text/csv")
