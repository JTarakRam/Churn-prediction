from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import os
import warnings
import sys
# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (project directory)
project_dir = os.path.dirname(current_dir)
# Add the project directory to the Python path
sys.path.append(project_dir)
# Now you can import the config module
from config.config import ARTIFACTS_DIR
from config.config import logger
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
warnings.filterwarnings("ignore")

def load_data(file_path):
    '''Load the dataset from a file or DataFrame.'''
    if isinstance(file_path, str):
        # If the input is a string (file path), load the data from CSV
        data = pd.read_csv(file_path)
        logger.info('Loaded the dataset from CSV!')
    elif isinstance(file_path, pd.DataFrame):
        # If the input is a DataFrame, use it directly
        data = file_path
        logger.info('Loaded the dataset from DataFrame!')
    else:
        raise ValueError("Invalid input type. Please provide either a file path (str) or a DataFrame.")
    
    return data

def clean_data(data):
    cleaned_data = data.drop_duplicates()

    # Handle outliers and fill missing values
    num_vars = cleaned_data.select_dtypes(include=np.number).columns
    cat_vars = cleaned_data.select_dtypes(include='object').columns
    for var in num_vars:
        if data[var].isnull().sum() > 0:
            has_outliers = False

            # Check for outliers
            Q1 = data[var].quantile(0.25)
            Q3 = data[var].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR

            if data[var].max() > upper_bound or data[var].min() < lower_bound:
                has_outliers = True

            if has_outliers:
                # Has outliers, fill missing values with median
                data[var].fillna(data[var].median(), inplace=True)
            else:
                # No outliers, fill missing values with mean
                data[var].fillna(data[var].mean(), inplace=True)

    for var in cat_vars:
        # Fill missing values with mode
        data[var].fillna(data[var].mode().iloc[0], inplace=True)

    logger.info('Cleaning completed!')

    return data

def replace_with_no(cleaned_data):
    '''
    Replace 'No internet service' and 'No phone service' with 'No' for the specified columns in the dataset.
    '''
    columns = ['OnlineSecurity']
    for column in columns:
        if column in cleaned_data.columns:
            cleaned_data[column] = cleaned_data[column].replace('No internet service', 'No')

    logger.info('Replaced with No')
    return cleaned_data

def label_encoding(cleaned_data):
    label_encoder = LabelEncoder()
    encode_vals = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                   'InternetService', 'OnlineSecurity', 'Contract', 'PaymentMethod', 'Churn']
    for column in encode_vals:
        if column in cleaned_data.columns:
            cleaned_data[column] = label_encoder.fit_transform(cleaned_data[column])
    logger.info('Label Encoding completed!')
    return cleaned_data

def to_numeric(cleaned_data):
    cleaned_data['TotalCharges'] = pd.to_numeric(cleaned_data['TotalCharges'], errors='coerce').fillna(0).astype(float)
    logger.info('Converted into int!')
    return cleaned_data

def scaling(cleaned_data):
    scaler = MinMaxScaler()
    numeric_columns = cleaned_data.select_dtypes(include=np.number).columns
    cleaned_data = cleaned_data.copy()
    cleaned_data[numeric_columns] = scaler.fit_transform(cleaned_data[numeric_columns])
    logger.info('Scaling Completed!')
    return cleaned_data

def preprocess_data(file_path):
    '''
    Preprocess the data by applying all the necessary steps.
    '''
    data = load_data(file_path)
    cleaned_data = clean_data(data)
    cleaned_data = replace_with_no(cleaned_data)
    cleaned_data = label_encoding(cleaned_data)
    cleaned_data = to_numeric(cleaned_data)
    cleaned_data = scaling(cleaned_data)
    logger.info('Data preprocessed!')

    return cleaned_data


if __name__ == "__main__":
    df = pd.read_csv("/Users/tarakram/Documents/Churn-Prediction/data/raw/customer_churn_raw_data.csv")
    label_encoder = LabelEncoder()
    scaler = MinMaxScaler()
    processed_data = preprocess_data(df)

    processed_data.to_csv(ARTIFACTS_DIR / 'processed_data.csv', index=False)
    with open(ARTIFACTS_DIR / 'label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    with open(ARTIFACTS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)