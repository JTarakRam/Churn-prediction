import warnings
import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly
import json
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_dir)
sys.path.append(project_dir)
from config.config import ARTIFACTS_DIR
warnings.filterwarnings("ignore")


def plot_churn_distribution(data):
    churn_distribution = data['Churn'].value_counts()
    fig = px.bar(x=churn_distribution.index, y=churn_distribution.values)
    fig.update_layout(
        xaxis_title='Churn',
        yaxis_title='Count',
        title='Distribution of Churn'
    )

    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json


def plot_demographics_churn_rate(data):
    demographics_churn_rate = data.groupby(['gender', 'SeniorCitizen', 'Partner', 'Dependents'])['Churn'].mean().reset_index()
    fig = px.bar(
        demographics_churn_rate,
        x='Churn',
        y='Churn',
        color='gender',
        facet_row='SeniorCitizen',
        facet_col='Partner',
        facet_col_wrap=2
    )
    fig.update_layout(
        yaxis_title='Churn Rate',
        title='Churn Rate by Demographics'
    )

    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json


def plot_reasons_for_churn(data):
    reasons_for_churn = data[['InternetService', 'OnlineSecurity', 'Contract', 'PaymentMethod', 'Churn']].groupby(['InternetService', 'OnlineSecurity', 'Contract', 'PaymentMethod'])['Churn'].sum().nlargest(5).reset_index()
    fig = px.bar(
        reasons_for_churn,
        x='Churn',
        y='Churn',
        color='InternetService',
        facet_row='OnlineSecurity',
        facet_col='Contract',
        facet_col_wrap=2
    )
    fig.update_layout(
        yaxis_title='Count',
        title='Top Reasons for Churn'
    )

    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json


def plot_churn_by_internet_service(data):
    churn_by_internet_service = data.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack()
    fig = px.bar(
        churn_by_internet_service,
        barmode='stack',
        labels={'index': 'Internet Service'},
        title='Churn Rate by Internet Service'
    )

    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

def save_eda_obj(data):
    with open(Path(ARTIFACTS_DIR, 'eda.json'), 'w') as f:
        json.dump(data, f)

data = pd.read_csv('/Users/tarakram/Documents/Churn-Prediction/data/processed/pre-processed_data.csv')
# Generate the EDA object
eda_data = {
    "churn_distribution": plot_churn_distribution(data),
    "demographics_churn_rate": plot_demographics_churn_rate(data),
    "reasons_for_churn": plot_reasons_for_churn(data),
    "churn_by_internet_service": plot_churn_by_internet_service(data),
}
# Save the EDA object as JSON
save_eda_obj(eda_data)