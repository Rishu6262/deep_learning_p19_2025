import streamlit as st
import numpy as np
import pandas as pd
import pickle

# load model & objects
model = pickle.load(open("model.pkl", "rb"))

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    columns = pickle.load(f)

st.set_page_config(page_title="Churn Prediction", layout="centered")

st.title("🔮 Customer Churn Prediction")
st.write("Predict whether a customer will exit or not")

# user inputs
credit_score = st.number_input("Credit Score", 300, 900, 650)
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure (years)", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
products = st.number_input("Number of Products", 1, 4, 1)
has_crcard = st.selectbox("Has Credit Card?", [0, 1])
is_active = st.selectbox("Is Active Member?", [0, 1])
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

gender = st.selectbox("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

# prepare input
input_dict = {
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": products,
    "HasCrCard": has_crcard,
    "IsActiveMember": is_active,
    "EstimatedSalary": salary,
    "Geography_Germany": 1 if geography == "Germany" else 0,
    "Geography_Spain": 1 if geography == "Spain" else 0,
    "Gender_Male": 1 if gender == "Male" else 0,
}

input_df = pd.DataFrame([input_dict])

# align columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# scale
input_scaled = scaler.transform(input_df)

# prediction
if st.button("Predict Churn"):
    pred = model.predict(input_scaled)[0]
    result = "❌ Customer will EXIT" if pred == 1 else "✅ Customer will NOT exit"

    st.subheader("Prediction Result")
    st.write(result)
