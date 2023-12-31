import pandas as pd
import pickle
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
from config.config import logger
from src.data import preprocess_data

def load_model():
    with open('artifacts/churn_model.pkl', 'rb') as file:
        model = pickle.load(file)
        logger.info('Loaded the model')
    return model
def predict_churn(data: pd.DataFrame):
    model = load_model()
    preprocessed_data = preprocess_data(data)
    predictions = model.predict(preprocessed_data)
    churn_predictions = pd.DataFrame({'Churn': predictions})
    return churn_predictions

# Prompt the user to enter their information
gender = input("Enter your gender (0 for female, 1 for male): ")
senior_citizen = input("Are you a senior citizen? (0 for No, 1 for Yes): ")
partner = input("Do you have a partner? (0 for No, 1 for Yes): ")
dependents = input("Do you have dependents? (0 for No, 1 for Yes): ")
tenure = input("Enter the tenure (number of months): ")
phone_service = input("Do you have phone service? (0 for No, 1 for Yes): ")
internet_service = input("Do you have internet service? (0 for No, 1 for Yes): ")
online_security = input("Do you have online security? (0 for No, 1 for Yes): ")
contract = input("Enter the contract type (0 for Month-to-month, 1 for One year, 2 for Two year): ")
payment_method = input("Enter the payment method (0 for Electronic check, 1 for Mailed check, 2 for Bank transfer (automatic), 3 for Credit card (automatic)): ")
monthly_charges = input("Enter the monthly charges: ")
total_charges = input("Enter the total charges: ")

# Create a DataFrame from the user input
user_data = pd.DataFrame({
    'gender': [float(gender)],
    'SeniorCitizen': [float(senior_citizen)],
    'Partner': [float(partner)],
    'Dependents': [float(dependents)],
    'tenure': [float(tenure)],
    'PhoneService': [float(phone_service)],
    'InternetService': [float(internet_service)],
    'OnlineSecurity': [float(online_security)],
    'Contract': [float(contract)],
    'PaymentMethod': [float(payment_method)],
    'MonthlyCharges': [float(monthly_charges)],
    'TotalCharges': [float(total_charges)]
})

# Make churn prediction for the user data
churn_result = predict_churn(user_data)
print(churn_result)
