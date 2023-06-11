import pandas as pd
import numpy as np
import pickle
import shutil
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from evidently import ColumnMapping
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset
import warnings
warnings.simplefilter('ignore')
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
from config.config import REPORTS_DIR
from config.config import DATA_DIR

def data_report():
    ref_data = pd.read_csv("/Users/tarakram/Documents/Churn-Prediction/data/processed/pre-processed_data.csv")
    cur_data = pd.read_csv("/Users/tarakram/Documents/Churn-Prediction/data/processed/test_data.csv")
    classification_performance_report = Report(metrics=[
    DataDriftPreset(), DataQualityPreset()
    ])
    classification_performance_report.run(reference_data=ref_data, current_data=cur_data, column_mapping = None)
    classification_performance_report.save_html(Path(REPORTS_DIR, 'data_report.html'))

    # Additional data analysis or reports can be added here using pandas_profiling library
    
    
def model_report():
    ref_data = pd.read_csv("/Users/tarakram/Documents/Churn-Prediction/data/processed/pre-processed_data.csv")
    cur_data = pd.read_csv("/Users/tarakram/Documents/Churn-Prediction/data/processed/test_data.csv")

    with open("artifacts/churn_model.pkl", "rb") as f:
        model = pickle.load(f)

    ref_X = ref_data.drop(["Churn"], axis=1)
    ref_y = ref_data["Churn"]

    cur_X = cur_data.drop(["Churn"], axis=1)
    cur_y = cur_data["Churn"]
    
    ref_X_train, ref_X_test, ref_y_train, ref_y_test = train_test_split(ref_X, ref_y, test_size=0.2, random_state=42)

    ref_pred = model.predict(ref_X_test)
    ref_pred = pd.DataFrame(ref_pred, columns=["Prediction"])
    cur_pred = model.predict(cur_X)
    cur_pred = pd.DataFrame(cur_pred, columns=["Prediction"])

    ref_X_test.reset_index(inplace=True, drop=True)
    ref_y_test.reset_index(inplace=True, drop=True)
    ref_merged = pd.concat([ref_X_test, ref_y_test], axis=1)
    ref_merged = pd.concat([ref_merged, ref_pred], axis=1)
    print(ref_merged)

    cur_X.reset_index(inplace=True, drop=True)
    cur_y.reset_index(inplace=True, drop=True)
    cur_merged = pd.concat([cur_X, cur_y], axis=1)
    cur_merged = pd.concat([cur_merged, cur_pred], axis=1)
    print(cur_merged)

    cm = ColumnMapping()
    cm.target = "Churn"
    cm.prediction = "Prediction"
    cm.target_names = ['Non-Churned', 'Churned']

    classification_performance_report = Report(metrics=[
    ClassificationPreset()
    ])

    classification_performance_report.run(reference_data=ref_merged, current_data=cur_merged, column_mapping = cm)
    classification_performance_report.save_html(Path(REPORTS_DIR, 'model_report.html'))

def generate_reports():
    data_report()
    model_report()
    # src = Path(REPORTS_DIR)
    # dst = 'frontend/reports/'
    # shutil.rmtree(dst)
    # shutil.copytree(src,dst)

generate_reports()