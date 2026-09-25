import streamlit as st
import plotly.express as px
import pandas as pd
from src.dashboard import context, chart, COLORS
from src.config import FEATURES, RFM

bundle, name, data, display, profiles = context()
st.caption("CUSTOMER INTELLIGENCE / DATA QUALITY")
st.title("Customer analysis")
st.markdown("Inspect the inputs before interpreting the segments. Income is in thousands of dollars; monetary value is in dollars.")
with st.container(horizontal=True):
    st.metric("Rows", f"{len(data):,}", border=True)
    st.metric("Model features", len(bundle["features"]), border=True)
    st.metric("Missing input cells", int(bundle["raw_missing"].sum()), border=True)
for note in bundle["notes"]:
    st.caption(note)
left, right = st.columns(2)
with left:
    feature = st.selectbox("Explore a feature", bundle["features"], key="eda_feature")
    chart(px.histogram(display, x=feature, color="Cluster", barmode="overlay", opacity=.65, color_discrete_sequence=COLORS, title=f"Distribution of {feature}"))
with right:
    chart(px.imshow(data[bundle["features"]].corr(), zmin=-1, zmax=1, color_continuous_scale="Tealrose", title="Feature correlations", aspect="auto"))
if all(column in data for column in RFM):
    st.subheader("RFM analysis")
    st.markdown("**Recency:** days since the last purchase · **Frequency:** purchases in the window · **Monetary:** total purchase amount. Customers with no purchases receive recency 366.")
    st.dataframe(data[["CustomerID", *RFM]].head(100), hide_index=True)
st.subheader("Dataset preview and statistics")
st.dataframe(data, hide_index=True)
with st.expander("Descriptive statistics and original missing values"):
    st.dataframe(data[bundle["features"]].describe())
    st.dataframe(bundle["raw_missing"].rename("Missing values").to_frame())
st.download_button("Download analyzed customer data", data.to_csv(index=False), "analyzed_customers.csv", "text/csv")
