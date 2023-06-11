import warnings
import os
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly
import json

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Set up logging
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOGS_DIR, "info.log"),
    format="%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


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


def plot_tenure_churn_correlation(data):
    data_copy = data.copy()  # Create a copy of the DataFrame
    data_copy['Churn'] = data_copy['Churn'].map({'Yes': 1, 'No': 0})
    tenure_churn_correlation = data_copy[['tenure', 'Churn']].corr().loc['tenure', 'Churn']

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Churn = No', 'Churn = Yes'))

    fig.add_trace(go.Box(x=data_copy[data_copy['Churn'] == 0]['Churn'], y=data_copy[data_copy['Churn'] == 0]['tenure'], name='No Churn'), row=1, col=1)
    fig.add_trace(go.Box(x=data_copy[data_copy['Churn'] == 1]['Churn'], y=data_copy[data_copy['Churn'] == 1]['tenure'], name='Churn'), row=1, col=2)

    fig.update_layout(
        yaxis_title='Tenure',
        title='Correlation between Tenure and Churn'
    )

    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return plot_json

def save_eda_obj(data):
    with open(Path(ARTIFACTS_DIR, 'eda.json'), 'w') as f:
        json.dump(data, f)

data = pd.read_csv('/Users/tarakram/Documents/Customer-Churn/data/processed/pre_processed_data.csv')
# Generate the EDA object
eda_data = {
    "churn_distribution": plot_churn_distribution(data),
    "demographics_churn_rate": plot_demographics_churn_rate(data),
    "reasons_for_churn": plot_reasons_for_churn(data),
    "churn_by_internet_service": plot_churn_by_internet_service(data),
    "tenure_churn_correlation": plot_tenure_churn_correlation(data)
}

# Save the EDA object as JSON
save_eda_obj(eda_data)