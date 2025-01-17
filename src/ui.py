import json

import requests
import streamlit as st


def preprocess_data(data):
    url = "http://localhost:8000/preprocess"
    response = requests.post(url, json=data)
    return response.json()


def predict(data):
    url = "http://localhost:8000/predict"
    response = requests.post(url, json=data)
    return response.json()


submitted = False


if submitted:
    st.rerun()
else:
    st.title("Loan Prediction")

    with st.form("Loan Prediction"):
        age = st.text_input("Age")
        income = st.text_input("Income")
        home_ownership = st.selectbox(
            "Home Ownership",
            ("MORTGAGE", "OTHER", "OWN", "RENT")
        )
        employment_length = st.text_input("Employment Length")
        loan_intent = st.selectbox(
            "Loan Intent",
            ("DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT", "MEDICAL", "PERSONAL", "VENTURE")
        )
        loan_grade = st.selectbox(
            "Loan Grade",
            ("A", "B", "C", "D", "E", "F", "G")
        )
        loan_amount = st.text_input("Loan Amount")
        loan_interest_rate = st.text_input("Interest Rate")
        loan_percent_income = st.text_input("Loan Percent Income")
        cb_person_default_on_file = st.selectbox(
            "CB Person Default On File",
            ("Y", "N")
        )
        credit_history_length = st.text_input("CB Person Credit History Length")

        submitted = st.form_submit_button("Predict")
        if submitted:
            data = {
                "age": age,
                "income": income,
                "home_ownership": home_ownership,
                "employment_length": employment_length,
                "loan_intent": loan_intent,
                "loan_grade": loan_grade,
                "loan_amount": loan_amount,
                "loan_interest_rate": loan_interest_rate,
                "loan_percent_income": loan_percent_income,
                "cb_person_default_on_file": cb_person_default_on_file,
                "credit_history_length": credit_history_length,
            }

            preprocessed_data = preprocess_data(data)
            prediction = predict(preprocessed_data)
            st.write(prediction)
