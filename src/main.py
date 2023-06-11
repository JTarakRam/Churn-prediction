import warnings
from pathlib import Path
import typer
import pickle
import json
import os
import sys
import pandas as pd
from data import clean_data, label_encoding, preprocess_data
from train import churn_prediction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "/Users/tarakram/Documents/Customer-Churn/data")
PROCESSED_DIR = os.path.join(DATA_DIR, "/Users/tarakram/Documents/Customer-Churn/data/processed")
sys.path.append('/Users/tarakram/Documents/Customer-Churn/config')
NOTEBOOK_DIR = Path(BASE_DIR, "/Users/tarakram/Documents/Customer-Churn/notebooks")
ARTIFACTS_DIR = Path(BASE_DIR, "/Users/tarakram/Documents/Customer-Churn/artifacts")
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)  # Create the processed directory if it doesn't exist

warnings.filterwarnings("ignore")
app = typer.Typer()

def get_data():
    df = pd.read_csv("/Users/tarakram/Documents/Customer-Churn/data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df
@app.command()
def preprocessing():
    df = get_data()
    df = clean_data(df)
    df = label_encoding(df)
    df = preprocess_data(df)
    df.to_csv(Path(PROCESSED_DIR, "preprocessed.csv"), index=False)
    return df
@app.command()
def train():
    df = pd.read_csv(Path(PROCESSED_DIR, "preprocessed.csv"))
    scores_df, best_model, best_model_name, best_model_report = churn_prediction(df)
    print("Scores")
    print(scores_df)
    print("Best model")
    print(best_model)
    print("Classification report")
    print(best_model_report)
    # with open(Path(ARTIFACTS_DIR, "churn_model.pkl"), "wb") as f:
    #     pickle.dump(best_model, f)

    scores_df = scores_df.to_json()
    best_model_report = best_model_report.to_json()
    model_metrics = [scores_df, best_model_report, best_model_name]
    with open(Path(ARTIFACTS_DIR, "model_metrics.json"), "w+") as f:
        json.dump(model_metrics, f)

if __name__ == "__main__":
    df = preprocessing()
    print