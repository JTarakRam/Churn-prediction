import warnings
from pathlib import Path
import typer
import pickle
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)

from config.config import logger
from config.config import ARTIFACTS_DIR, DATA_DIR
from src.data import (
    load_data,
    clean_data,
    replace_with_no,
    label_encoding,
    to_numeric,
    scaling,
)
from train import churn_prediction
from src.eda import (
    plot_churn_distribution,
    plot_demographics_churn_rate,
    plot_reasons_for_churn,
    plot_churn_by_internet_service,
)

app = typer.Typer()

def get_data():
    df = pd.read_csv("/Users/tarakram/Documents/Churn-Prediction/data/processed/pre-processed_data.csv")
    return df

def eda(df):
    q1 = plot_churn_distribution(df)
    q2 = plot_demographics_churn_rate(df)
    q3 = plot_reasons_for_churn(df)
    q4 = plot_churn_by_internet_service(df)

    json_obj = {
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4" : q4
    }

    with open(Path(ARTIFACTS_DIR, "eda.json"), "w+") as f:
        json.dump(json_obj, f)
    
@app.command()
def preprocess():
    df = pd.read_csv("/Users/tarakram/Documents/Churn-Prediction/data/processed/pre-processed_data.csv")
    df = load_data(df)
    df = clean_data(df)
    df = replace_with_no(df)
    df = label_encoding(df)
    df = scaling(df)
    df.to_csv("/Users/tarakram/Documents/Churn-Prediction/data/processed/preprocessed_data.csv", index=False)
    return df

@app.command()
def split_data():
    df = pd.read_csv("/Users/tarakram/Documents/Churn-Prediction/data/processed/preprocessed_data.csv")
    target = 'Churn'
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target])
    train_data.to_csv(Path(DATA_DIR, "processed/train_data.csv"), index=False)
    test_data.to_csv(Path(DATA_DIR, "processed/test_data.csv"), index=False)

@app.command()
def train():
    df = pd.read_csv("/Users/tarakram/Documents/Churn-Prediction/data/processed/train_data.csv")
    scores_df, best_model, best_model_name, report = churn_prediction(df)
    print("Scores")
    print(scores_df)
    print("Best model")
    print(best_model)
    print("Classification report")
    print(report)
    with open(Path(ARTIFACTS_DIR, "churn_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    scores_df = scores_df.to_json()
    report = report.to_json()
    model_metrics = [scores_df, report, best_model_name]
    with open(Path(ARTIFACTS_DIR, "model_metrics.json"), "w+") as f:
        json.dump(model_metrics, f)

get_data()
eda(get_data())
df = preprocess()
split_data()
print(df)
train()

app(prog_name="main.py")
