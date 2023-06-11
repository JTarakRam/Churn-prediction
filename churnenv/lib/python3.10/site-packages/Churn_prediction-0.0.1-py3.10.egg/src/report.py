import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from evidently import ColumnMapping
from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, ClassificationPreset
import warnings

warnings.simplefilter('ignore')

def data_report(ref_data_path, cur_data_path):
    ref_data = pd.read_csv(ref_data_path)
    cur_data = pd.read_csv(cur_data_path)

    ref_X = ref_data.drop(["Churn"], axis=1)
    ref_y = ref_data["Churn"]

    cur_X = cur_data.drop(["Churn"], axis=1)
    cur_y = cur_data["Churn"]

    ref_X_train, ref_X_test, ref_y_train, ref_y_test = train_test_split(ref_X, ref_y, test_size=0.2, random_state=42)

    # Additional data analysis or reports can be added here using pandas_profiling library

def model_report(ref_data_path, cur_data_path):
    ref_data = pd.read_csv(ref_data_path)
    cur_data = pd.read_csv(cur_data_path)

    ref_X = ref_data.drop(["Churn"], axis=1)
    ref_y = ref_data["Churn"]

    cur_X = cur_data.drop(["Churn"], axis=1)
    cur_y = cur_data["Churn"]

    ref_X_train, ref_X_test, ref_y_train, ref_y_test = train_test_split(ref_X, ref_y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier()
    rf.fit(ref_X.values, ref_y.values)

    ref_pred = rf.predict(ref_X_test)
    ref_pred = pd.DataFrame(ref_pred, columns=["Prediction"])
    cur_pred = rf.predict(cur_X)
    cur_pred = pd.DataFrame(cur_pred, columns=["Prediction"])

    ref_X_test.reset_index(inplace=True, drop=True)
    ref_y_test.reset_index(inplace=True, drop=True)
    ref_merged = pd.concat([ref_X_test, ref_y_test], axis=1)
    ref_merged = pd.concat([ref_merged, ref_pred], axis=1)

    cur_X.reset_index(inplace=True, drop=True)
    cur_y.reset_index(inplace=True, drop=True)
    cur_merged = pd.concat([cur_X, cur_y], axis=1)
    cur_merged = pd.concat([cur_merged, cur_pred], axis=1)

    cm = ColumnMapping()
    cm.target = "Churn"
    cm.prediction = "Prediction"
    cm.target_names = ['Non-Churned', 'Churned']

    classification_performance_report = Report(metrics=[ClassificationPreset()])
    classification_performance_report.run(reference_data=ref_merged, current_data=cur_merged, column_mapping=cm)
    classification_performance_report.save_html("classification_performance_report.html")

ref_data_path = pd.read_csv("/Users/tarakram/Documents/Customer_Churn_Classification/data/processed/pre_processed_data.csv")
cur_data_path = pd.read_csv("/Users/tarakram/Documents/Customer_Churn_Classification/data/processed/test_data.csv")

data_report(ref_data_path, cur_data_path)
model_report(ref_data_path, cur_data_path)
