import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.dashboard import context, chart, distribution, COLORS
from src.config import INCOME, SPENDING, RFM

bundle, name, data, display, profiles = context()
st.caption("CUSTOMER INTELLIGENCE / SEGMENT EXPLORER")
st.title("Clustering results")
st.markdown(f"Explore **{name}** segments. Charts show projections; the model uses {len(bundle['features'])} features.")
dimension = st.segmented_control("Visualization", ["2D income & spending", "3D customer view"], default="2D income & spending")
if dimension == "3D customer view":
    axes = RFM if all(column in data for column in RFM) else [INCOME, SPENDING, "Age"]
    figure = px.scatter_3d(display, x=axes[0], y=axes[1], z=axes[2], color="Cluster", hover_data=["CustomerID", "Category"], color_discrete_sequence=COLORS, height=600)
    figure.update_traces(marker_size=3)
else:
    figure = px.scatter(display, x=INCOME, y=SPENDING, color="Cluster", hover_data=["CustomerID", "Age", "Category"], color_discrete_sequence=COLORS, opacity=.7, height=510)
    figure.add_trace(go.Scatter(x=profiles[INCOME], y=profiles[SPENDING], mode="markers", marker=dict(symbol="x", size=15, color="#172b3a"), name="Profile means"))
chart(figure)
left, right = st.columns(2)
with left:
    curve = px.line(bundle["elbow"], x="k", y="Inertia", markers=True, title="K-Means elbow curve")
    curve.add_vline(x=bundle["k"], line_dash="dash", line_color="#f59e0b")
    chart(curve)
    st.caption(f"Heuristic suggestion: {bundle['suggested_k']}. Selected k: {bundle['k']}. DBSCAN determines its own number of clusters.")
with right:
    chart(distribution(data))
st.subheader("Cluster profiles in original units")
st.dataframe(profiles.round(2))
st.download_button("Download cluster profiles", profiles.to_csv(), "cluster_profiles.csv", "text/csv")
st.download_button("Download customer assignments", data.to_csv(index=False), "segmented_customers.csv", "text/csv")
