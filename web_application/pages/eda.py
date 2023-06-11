import streamlit as st
import json
import plotly.io as pio

st.title('Exploratory Data Analysis')

# Load the EDA object from JSON
with open('/Users/tarakram/Documents/Churn-Prediction/artifacts/eda.json', 'r') as f:
    eda_data = json.load(f)

# Function to render Plotly charts
def render_plotly_chart(plot_json):
    fig = pio.from_json(plot_json)
    st.plotly_chart(fig)

# Question 1: What is the distribution of churn in the dataset?
st.header('Question 1')
st.write("What is the distribution of churn in the dataset?")
render_plotly_chart(eda_data['churn_distribution'])
st.markdown("""
            >**The dataset contains most of the non-churn labels, which is double than the churn labels.**
            """)
st.divider()

# Question 2: How does the churn rate vary based on different customer demographics?
st.header('Question 2')
st.write("How does the churn rate vary based on different customer demographics, such as gender, senior citizen status, partner, and dependents?")
render_plotly_chart(eda_data['demographics_churn_rate'])
st.divider()

# Question 3: What are the most common reasons for churn based on available features?
st.header('Question 3')
st.write("What are the most common reasons for churn based on the available features like internet service, online security, contract type, and payment method?")
render_plotly_chart(eda_data['reasons_for_churn'])
st.divider()

# Question 4: How does the churn rate vary based on the type of internet service?
st.header('Question 4')
st.write("How does the churn rate vary based on the type of internet service?")
render_plotly_chart(eda_data['churn_by_internet_service'])
