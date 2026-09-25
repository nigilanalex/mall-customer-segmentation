"""Small shared view helpers; ML logic lives in the other src modules."""
import plotly.express as px
import streamlit as st
from .recommendation import customer_results

COLORS = ["#0d9488", "#6366f1", "#f59e0b", "#ec4899", "#0284c7", "#84cc16", "#a855f7", "#64748b"]


def context():
    bundle = st.session_state.analytics
    name = st.session_state.active_model
    data = customer_results(bundle, name)
    display = data.assign(Cluster=data.Cluster.astype(str))
    return bundle, name, data, display, bundle["models"][name]["profiles"]


def chart(figure):
    figure.update_layout(font=dict(family="Arial", size=13), margin=dict(l=16, r=16, t=45, b=20))
    st.plotly_chart(figure, width="stretch")


def distribution(data):
    counts = data.groupby("Cluster").size().rename("Customers").reset_index()
    counts["Cluster"] = counts.Cluster.astype(str)
    return px.bar(counts, x="Cluster", y="Customers", color="Cluster", color_discrete_sequence=COLORS, title="Customers per segment")
