import pandas as pd
import pickle
import streamlit as st
from pathlib import Path
from src.data import preprocess_data

# Load the trained model
with open('artifacts/churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the scaler
with open("artifacts/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define the input prompts as Streamlit input widgets
gender = st.radio("Gender", ["Female", "Male"])
senior_citizen = st.radio("Is the customer a senior citizen?", ["No", "Yes"])
partner = st.radio("Does the customer have a partner?", ["No", "Yes"])
dependents = st.radio("Does the customer have dependents?", ["No", "Yes"])
tenure = st.number_input("Enter the tenure (number of months with the company)", value=0)
phone_service = st.radio("Does the customer have phone service?", ["No", "Yes"])
internet_service = st.selectbox(
    "Enter the type of internet service",
    ["DSL", "Fiber optic", "No"]
)
online_security = st.radio("Does the customer have online security?", ["No", "Yes"])
contract = st.selectbox("Enter the type of contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.selectbox(
    "Enter the payment method",
    ["Bank transfer", "Credit card", "Electronic check", "Mailed check"]
)
monthly_charges = st.number_input("Enter the monthly charges", value=0.0)
total_charges = st.number_input("Enter the total charges", value=0.0)

# Create a DataFrame with the input data
data = pd.DataFrame({
    "gender": [0 if gender == "Female" else 1],
    "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
    "Partner": [1 if partner == "Yes" else 0],
    "Dependents": [1 if dependents == "Yes" else 0],
    "tenure": [tenure],
    "PhoneService": [1 if phone_service == "Yes" else 0],
    "InternetService": [0 if internet_service == "DSL" else 1 if internet_service == "Fiber optic" else 2],
    "OnlineSecurity": [1 if online_security == "Yes" else 0],
    "Contract": [0 if contract == "Month-to-month" else 1 if contract == "One year" else 2],
    "PaymentMethod": [0 if payment_method == "Bank transfer" else 1 if payment_method == "Credit card" else 2 if payment_method == "Electronic check" else 3],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

if st.button("Predict"):
    # Preprocess the data
    preprocessed_data = preprocess_data(data)

    # # Scale the data
    # scaled_data = scaler.fit_transform(preprocessed_data)

    # Make predictions
    prediction = model.predict(preprocessed_data)

    if prediction[0] == 1:
        st.write("Churn")
    else:
        st.write("No churn")
