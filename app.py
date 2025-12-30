import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# load model & objects
model = load_model('model.keras',compile=False)

with open('scaler.pkl','rb') as f:
    scaler = pickle.load(f)

with open('columns.pkl','rb') as f:
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
    "Geography_France": 0,
    "Geography_Germany": 0,
    "Geography_Spain": 0,
    "Gender_Male": 0
}

# encoding
if geography == "Germany":
    input_dict["Geography_Germany"] = 1
elif geography == "Spain":
    input_dict["Geography_Spain"] = 1

if gender == "Male":
    input_dict["Gender_Male"] = 1

# dataframe
input_df = pd.DataFrame([input_dict])

# align columns
input_df = input_df.reindex(columns=columns, fill_value=0)

# scale
input_scaled = scaler.transform(input_df)

# prediction
if st.button("Predict Churn"):
    prob = model.predict(input_scaled)[0][0]
    result = "❌ Customer will EXIT" if prob > 0.5 else "✅ Customer will NOT exit"

    st.subheader("Prediction Result")
    st.write(result)
    st.write(f"Churn Probability: **{prob:.2f}**")


