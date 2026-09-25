import streamlit as st
from src.dashboard import context
from src.config import INCOME, SPENDING, FREQUENCY, MONETARY
from src.prediction import predict_customer

bundle, name, data, display, profiles = context()
st.caption("CUSTOMER INTELLIGENCE / LIVE ASSIGNMENT")
st.title("Customer prediction")
st.markdown(f"Assign a new customer using **{name}** and the fitted preprocessing from this analysis.")
with st.form("customer_prediction_form"):
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age", 0, 120, 35)
        gender = st.selectbox("Gender", ["Female", "Male", "Other", "Prefer not to say"])
        income = st.number_input("Annual income (k$)", 0.0, value=95.0, step=5.0)
        spending = st.number_input("Spending score (1–100)", 1, 100, 85)
    with right:
        recency = st.number_input("Recency (days)", 0, value=12)
        frequency = st.number_input("Purchase frequency", 0, value=28)
        monetary = st.number_input("Monetary value ($)", 0.0, value=5500.0, step=100.0)
        st.caption("Use the same 365-day window and analysis date as the training data. Gender is descriptive and excluded from clustering.")
    submitted = st.form_submit_button("Predict customer segment", type="primary")
if submitted:
    try:
        result = predict_customer(bundle, {"Age": age, "Gender": gender, INCOME: income, SPENDING: spending,
                                          "Recency": recency, FREQUENCY: frequency, MONETARY: monetary}, name)
        st.session_state.prediction = result
    except ValueError as exc:
        st.error(str(exc))
        st.session_state.pop("prediction", None)
result = st.session_state.get("prediction")
if result and result["model"] == name:
    with st.container(border=True):
        st.subheader(result["category"])
        st.markdown(f"**Cluster {result['cluster']}** · {result['model']}")
        st.markdown(f"**Recommended action:** {result['recommendation']}")
        st.caption(result["method"])
        if result["probability"] is not None:
            st.metric("GMM component probability", f"{result['probability']:.1%}")
            st.caption("Model probability, not calibrated confidence or a correctness guarantee.")
    for warning in result["warnings"]:
        st.warning(warning)
    st.subheader("Why this assignment?")
    st.markdown(result["explanation"])
    if not result["details"].empty:
        st.dataframe(result["details"].round(3), hide_index=True)
