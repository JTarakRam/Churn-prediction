import streamlit as st
from annotated_text import annotated_text
import pandas as pd
import json

st.set_page_config(page_title="Performance Measures", layout="wide")

st.title('Performance Measures')

model_file_path = "/Users/tarakram/Documents/Churn-Prediction/artifacts/metrics.json"

with open(model_file_path, 'r') as file:
    model_data = json.load(file)

col1, col2 = st.columns(2)

with col1:
    model_scores = model_data["scores"]
    model_scores = pd.DataFrame(model_scores)
    st.subheader("MODEL: Churn or No churn")
    st.write('Model Scores')
    st.dataframe(model_scores)

    best_model_name = model_data["best_model_name"]
    annotated_text((best_model_name, "Best Model Name"))

    model_report = model_data["report"]
    model_report = pd.DataFrame(model_report)
    st.write('Best Model Report')
    st.dataframe(model_report)
