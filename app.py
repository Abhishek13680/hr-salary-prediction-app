import streamlit as st
import pandas as pd
import joblib

# Load model (must be in repo root, named exactly final_mlr_model.joblib)
model = joblib.load("final_mlr_model.joblib")

# Load dataset for dropdowns (must be in repo root, named exactly cleaned_file.xlsx)
df = pd.read_excel("cleaned_file.xlsx")

st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")
st.title("💼 HR Salary Prediction App")
st.write("Enter details to estimate expected salary using the trained MLR model.")

# Collect dropdown options from data
roles = sorted(df["Role"].dropna().unique())
countries = sorted(df["Country"].dropna().unique())

# Inputs
age = st.slider("Age", 18, 70, 30)
happiness = st.slider("Total Happiness Score", 1.0, 10.0, 7.0)
role = st.selectbox("Role", roles)
country = st.selectbox("Country", countries)

# Predict
if st.button("Predict Salary"):
    input_data = pd.DataFrame([{
        "Age": age,
        "Total Happines": happiness,
        "Role": role,
        "Country": country
    }])
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Predicted Salary: {prediction:.2f} K USD")
