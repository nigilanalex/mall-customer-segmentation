"""Interactive dashboard: python -m streamlit run app.py."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from main import DATASET, FEATURES, elbow_analysis, load_data, preprocess, train_model

st.set_page_config(page_title="Mall Customer Segmentation", page_icon="ðŸ›ï¸", layout="wide")
st.title("Mall Customer Segmentation")
st.write("Explore income and spending patterns with K-Means clustering.")
st.caption("The bundled 500-customer dataset is synthetic and deliberately contains five groups. It is not real mall research.")
upload = st.sidebar.file_uploader("Upload customer CSV", type=["csv"])
st.sidebar.caption("Required: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100).")


@st.cache_data(show_spinner=False)
def analyze(raw):
    clean, notes = preprocess(raw)
    elbow, suggested = elbow_analysis(clean)
    return clean, notes, elbow, suggested


try:
    raw = load_data(upload if upload is not None else DATASET)
    with st.spinner("Validating data and finding the elbow..."):
        data, notes, elbow, suggested = analyze(raw)
    with st.expander("Dataset information and missing values"):
        st.dataframe(pd.DataFrame({"Type": raw.dtypes.astype(str), "Missing": raw.isna().sum()}))
        st.dataframe(raw.head(10))
        st.dataframe(data.describe())
    for note in notes:
        st.info(note)
    st.sidebar.write(f"Elbow suggestion: **{suggested} clusters**")
    k = st.sidebar.slider("Number of clusters", 2, int(elbow.k.max()), suggested)
    st.sidebar.caption("The elbow is a heuristic. Compare nearby values and inspect the centers.")
    with st.spinner("Fitting clusters..."):
        _, labeled, centers, score = train_model(data, k)
    a, b, c = st.columns(3)
    a.metric("Customers", len(data))
    b.metric("Clusters", k)
    c.metric("Silhouette score", f"{score:.3f}")
    st.caption("Silhouette measures separation (âˆ’1 to 1); it is not prediction accuracy. Cluster numbers have no rank.")
    left, right = st.columns([1, 2])
    with left:
        curve = px.line(elbow, x="k", y="Inertia", markers=True, title="Elbow method")
        curve.add_vline(x=k, line_dash="dash", line_color="orange")
        st.plotly_chart(curve, width="stretch")
    with right:
        display = labeled.assign(Cluster=labeled.Cluster.astype(str))
        chart = px.scatter(display, x=FEATURES[0], y=FEATURES[1], color="Cluster", hover_data=["CustomerID", "Age", "Gender"], title="Customer segments", category_orders={"Cluster": [str(i) for i in range(k)]})
        chart.add_trace(go.Scatter(x=centers[FEATURES[0]], y=centers[FEATURES[1]], mode="markers", marker=dict(symbol="x", size=18, color="black"), name="Centers"))
        st.plotly_chart(chart, width="stretch")
    st.subheader("Cluster centers and sizes")
    summary = centers.join(labeled.groupby("Cluster").size().rename("Customers"))
    st.dataframe(summary.round(2), width="stretch")
    st.subheader("Explore the data")
    feature = st.selectbox("Distribution", ["Age", *FEATURES])
    st.plotly_chart(px.histogram(display, x=feature, color="Cluster", barmode="overlay", opacity=.65), width="stretch")
    st.subheader("Customer assignments")
    st.dataframe(labeled, width="stretch")
    st.download_button("Download segmented customers", labeled.to_csv(index=False).encode("utf-8"), "segmented_customers.csv", "text/csv")
    st.download_button("Download cluster centers", summary.to_csv().encode("utf-8"), "cluster_centers.csv", "text/csv")
except (ValueError, OSError) as exc:
    st.error(str(exc))
    st.info("Check the CSV columns and numeric values, then upload the corrected file.")
    st.stop()

