import streamlit as st
import pandas as pd

st.set_page_config(page_title="Churn Prediction", layout="wide")

st.title('Churn Prediction')

st.markdown('''
## 1. Overview

This design doc outlines the development of a web application for churn prediction using customer data. The application will utilize a machine learning model
 that predicts whether a customer is likely to churn based on various features such as gender, senior citizen status, partner, dependents, tenure, services subscribed, contract type, billing method, and charges.

## 2. Motivation

Churn prediction can help businesses identify customers who are at risk of leaving and take proactive measures to retain them.
 Developing a web application for churn prediction can provide businesses with real-time insights into customer churn, enabling targeted retention strategies and reducing revenue loss.

## 3. Success Metrics

The success of the project will be measured based on the following metrics:

- Accuracy, precision, recall, and F1 score of the machine learning model.
- Responsiveness and ease of use of the web application.
- Improvement in customer retention rate.

## 4. Requirements & Constraints

### 4.1 Functional Requirements

The web application should provide the following functionality:

- Users can enter customer details and receive a prediction of whether the customer is likely to churn.
- Users can view and analyze the predictions for different customers.
- Users can explore the impact of different features on the churn prediction.

### 4.2 Non-functional Requirements

The web application should meet the following non-functional requirements:

- The model should have high accuracy, precision, recall, and F1 score.
- The web application should be responsive and user-friendly.
- The web application should be secure and protect customer data.

### 4.3 Constraints

- The application should be built using Streamlit and deployed using cloud infrastructure like Heroku or AWS.
- The cost of deployment should be minimal.

### 4.4 Out-of-scope

- Integrating with external applications or data sources.
- Providing personalized retention strategies.

## 5. Methodology

### 5.1. Problem Statement

The problem is to develop a machine learning model that predicts customer churn based on various features.

### 5.2. Data

The dataset consists of customer data with features such as gender, senior citizen status, partner, dependents,
 tenure, services subscribed, contract type, billing method, and charges. The target variable is a binary label indicating whether the customer churned or not.

### 5.3. Techniques

We will utilize a binary classification model to predict customer churn. The following machine learning techniques will be used:

- Data preprocessing and cleaning
- Feature engineering and selection
- Model selection and training
- Hyperparameter tuning
- Model evaluation and testing

## 6. Architecture

The web application architecture will consist of the following components:

- A frontend web application built using Streamlit
- A machine learning model for churn prediction
- Cloud infrastructure to host the application

The frontend will allow users to enter customer details and display churn predictions.
The machine learning model will be trained and deployed using cloud infrastructure.
The application will be hosted on platforms like Heroku or AWS.

## 7. Conclusion

This design doc outlines the development of a web application for churn prediction using customer data.
The application will utilize a machine learning model that predicts whether a customer is likely to churn based on various features.
The web application will be built using Streamlit and deployed using cloud infrastructure. By accurately predicting customer churn, businesses can implement targeted retention strategies and reduce revenue loss.
''')