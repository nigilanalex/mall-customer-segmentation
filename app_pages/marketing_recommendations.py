import streamlit as st
from src.dashboard import context
from src.config import INCOME, SPENDING

bundle, name, data, display, profiles = context()
st.caption("CUSTOMER INTELLIGENCE / MARKETING PLAYBOOK")
st.title("Marketing recommendations")
st.markdown("Turn segment profiles into a campaign starting point. These are transparent rules, not predictions of campaign success.")
st.caption(f"Cohort thresholds: income ≥ {bundle['thresholds'][INCOME]:.2f} k$ is high; spending score ≥ {bundle['thresholds'][SPENDING]:.2f} is high. Recency > 90 days adds a win-back suggestion.")
selected = st.selectbox("Customer segment", list(profiles.index), format_func=lambda value: f"Segment {value} · {profiles.loc[value, 'Category']}")
profile = profiles.loc[selected]
with st.container(border=True):
    st.subheader(profile.Category)
    with st.container(horizontal=True):
        st.metric("Customers", int(profile.Customers))
        st.metric("Mean income", f"${profile[INCOME]:.1f}k")
        st.metric("Mean spending score", f"{profile[SPENDING]:.1f}")
    st.markdown(f"**Campaign suggestion:** {profile.Recommendation}")
members = data[data.Cluster == selected]
st.subheader("Customer profile")
customer_id = st.selectbox("Customer ID", members.CustomerID.tolist())
customer = members[members.CustomerID == customer_id].iloc[0]
with st.container(border=True):
    st.markdown(f"**Customer {customer_id} · {customer.Category}**")
    st.dataframe(customer[bundle["features"]].rename("Value").to_frame())
    st.markdown(customer.Recommendation)
st.download_button("Download this campaign audience", members.to_csv(index=False), "campaign_audience.csv", "text/csv")
st.download_button("Download all recommendations", profiles[["Customers", "Category", "Recommendation"]].to_csv(), "marketing_recommendations.csv", "text/csv")
