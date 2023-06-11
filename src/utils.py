import numpy as np
from config import logger 
import pandas as pd


def get_columns():
    df = pd.read_csv("/Users/tarakram/Documents/Customer-Churn/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df.columns


def get_num_columns():
    return ['SeniorCitizen', 'tenure', 'MonthlyCharges']

logger.info('Got the Numerical columns !')

def get_cat_columns():
    return ['customerID', 'gender', 'Partner', 'Dependents', 'PhoneService', 
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
            'PaperlessBilling', 'PaymentMethod', 'TotalCharges', 'Churn']
logger.info('Got the Categorical columns !')