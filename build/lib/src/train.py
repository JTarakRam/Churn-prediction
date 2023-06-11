import mlflow
import mlflow.sklearn
import json
import pandas as pd
from pathlib import Path
import pickle
from imblearn.over_sampling import SMOTE
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
from config.config import logger
from config.config import ARTIFACTS_DIR

# Define models for churn prediction
lr = LogisticRegression(max_iter=5)
bnb = BernoulliNB()
mnb = MultinomialNB()
gnb = GaussianNB()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

models = [lr, bnb, mnb, gnb, dt, rf]
model_names = ["Logistic Regression", "Bernoulli Naive Bayes", "Multinomial Naive Bayes",
               "Gaussian Naive Bayes", "Decision Tree", "Random Forest"]

def churn_prediction(df: pd.DataFrame):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    oversample = SMOTE(sampling_strategy='auto')
    X_sample, y_sample = oversample.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

    scores = []
    for i, model in enumerate(models):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        prec = precision_score(y_test, y_pred) * 100
        rec = recall_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred) * 100
        scores.append([acc, prec, rec, f1])

        with mlflow.start_run():
            mlflow.sklearn.log_model(model, model_names[i])
            mlflow.log_metric("Accuracy", acc)
            mlflow.log_metric("Precision", prec)
            mlflow.log_metric("Recall", rec)
            mlflow.log_metric("F1", f1)

            logger.info(f"Model {model_names[i]} logged to MLflow.")
            logger.info(f"Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")

            report = classification_report(y_test, y_pred, output_dict=True)
            report = pd.DataFrame(report).transpose()
            logger.info(f"Classification Report for Model {model_names[i]}:\n{report}")
            

    scores_df = pd.DataFrame(columns=["Model"], data=model_names)
    scores_df = pd.concat([scores_df, pd.DataFrame(scores, columns=["Accuracy", "Precision", "Recall", "F1"])], axis=1)

    best_model_idx = scores_df["F1"].idxmax()
    best_model = models[best_model_idx]
    best_model_name = model_names[best_model_idx]
    y_pred = best_model.predict(X_test)

    best_model_report = classification_report(y_test, y_pred, output_dict=True)
    best_model_report = pd.DataFrame(best_model_report).transpose()

    with open(Path(ARTIFACTS_DIR, 'churn_model.pkl'), 'wb') as file:
        pickle.dump(best_model, file)
    scores_df.to_csv(Path(ARTIFACTS_DIR, 'scores.csv'), index=False)
    best_model_report.to_csv(Path(ARTIFACTS_DIR, 'classification_report.csv'), index=False)

    logger.info(f"Best model: {best_model_name}")
    logger.info(f"Churn model and evaluation scores saved to {ARTIFACTS_DIR}.")
    print(best_model_name)

    # Save metrics in JSON format
    metrics = {
        "scores": scores_df.to_dict(),
        "report": best_model_report.to_dict(),
        "best_model_name": best_model_name
    }

    with open(Path(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    return scores_df, best_model, best_model_name, best_model_report

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('/Users/tarakram/Documents/Churn-Prediction/artifacts/processed_data.csv')

    # Call the churn_prediction function
    scores_df, best_model, best_model_name, best_model_report = churn_prediction(df)
    
    # Print classification reports for all models
    for i, model_name in enumerate(model_names):
        logger.info(f"Classification Report for Model {model_name}:\n{scores_df.iloc[i]}")
