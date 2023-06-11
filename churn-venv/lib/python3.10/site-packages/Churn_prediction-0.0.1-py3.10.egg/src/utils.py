import numpy as np
import pandas as pd
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
from config.config import logger


def get_columns():
    df = pd.read_csv("/Users/tarakram/Documents/Churn-Prediction/data/raw/customer_churn_raw_data.csv")
    return df.columns

def get_num_columns():
    return ['SeniorCitizen', 'tenure', 'MonthlyCharges']

logger.info('Got the Numerical columns !')

def get_cat_columns():
    return ['gender',
 'Partner',
 'Dependents',
 'PhoneService',
 'InternetService',
 'OnlineSecurity',
 'Contract',
 'PaymentMethod',
 'TotalCharges',
 'Churn']
    
logger.info('Got the Categorical columns !')