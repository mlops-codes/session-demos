import streamlit as st
import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Loan Approval Prediction")

income = st.number_input("Income", min_value=0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
loan_amount = st.number_input("Loan Amount", min_value=1000)
loan_term = st.selectbox("Loan Term (months)", [180, 240, 360])

if st.button("Predict"):
    input_data = np.array([[income, credit_score, loan_amount, loan_term]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
