## Customer Churn Prediction


[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/zenml)](https://pypi.org/project/zenml/)

**Problem statement**: The problem at hand involves developing a predictive model for a telecom company to identify customers who are likely to churn (cancel their subscription). 

By accurately predicting churn, the company can proactively take measures to retain these customers and minimize the overall churn rate, thereby improving revenue and profitability.

The project requires analyzing a telecom data set that contains historical information about customers, including demographics, usage patterns, service subscriptions, and churn status. 

The goal is to build a machine learning model that can effectively predict whether a customer is likely to churn based on these features. The model should be capable of processing new data and providing churn predictions with high accuracy.

## Table of Contents

- [Motivation](#motivation)
- [Features](#features)
- [Success Metrics](#success-metrics)
- [Requirements & Constraints](#requirements--constraints)
- [Methodology](#methodology)
- [Architecture](#architecture)
- [Usage](#usage)
- [Conclusion](#conclusion)

## Motivation

Churn prediction is essential for businesses to identify customers who are likely to leave and implement targeted retention strategies. This web application provides real-time insights into customer churn, enabling businesses to take proactive measures and improve customer retention rates.

## Features

- Input customer details and receive a churn prediction indicating whether the customer is likely to churn or not.
- View and analyze churn predictions for different customers.
- Explore the impact of different features on the churn prediction.

## Success Metrics

The success of the project will be measured based on the following metrics:

- Accuracy, precision, recall, and F1 score of the machine learning model.
- Responsiveness and ease of use of the web application.
- Improvement in customer retention rate.

## Requirements & Constraints

### Functional Requirements

- Users can enter customer details and receive a churn prediction.
- Users can view and analyze churn predictions for different customers.
- Users can explore the impact of different features on the churn prediction.

### Non-functional Requirements

- The model should have high accuracy, precision, recall, and F1 score.
- The web application should be responsive and user-friendly.
- The web application should be secure and protect customer data.

### Constraints

- The application is built using Streamlit and deployed using cloud infrastructure.
- The cost of deployment should be kept minimal.

### Out-of-scope

- Integrating with external applications or data sources.
- Providing personalized retention strategies.

## Methodology

### Problem Statement

The problem is to develop a machine learning model that predicts customer churn based on various features.

### Data 

The dataset consists of customer data with features such as demographics (gender, SeniorCitizen), service usage (Partner, Dependents, tenure, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies), contract details (PaperlessBilling, MonthlyCharges, TotalCharges, Contract_Month-to-month, Contract_One year, Contract_Two year), payment methods (PaymentMethod_Bank transfer (automatic), PaymentMethod_Credit card (automatic), PaymentMethod_Electronic check, PaymentMethod_Mailed check), and internet service options (InternetService_DSL, InternetService_Fiber optic, InternetService_No).

The target variable is a binary label indicating whether the customer churned or not.

### Techniques

We utilize a binary classification model to predict customer churn. The following machine learning techniques are used:

- Data preprocessing and cleaning
- Feature engineering and selection
- Model selection and training
- Hyperparameter tuning
- Model evaluation and testing

## Architecture

The web application architecture consists of the following components:

- Frontend: A web application built using Streamlit.
- Machine Learning Model: A model for churn prediction.
- Cloud Infrastructure: Hosting the application on platforms like Heroku or AWS.

The frontend allows users to input customer details and displays churn predictions. The machine learning model is trained and deployed using cloud infrastructure.

## Usage

To use the churn prediction web application, follow these steps:

- Install the necessary dependencies by running pip install -r requirements.txt.
- Prepare your dataset by ensuring it has the required features mentioned in the Data section.
- Train your churn prediction machine learning model using the dataset and the techniques mentioned in the Methodology section.
- save the trained model in a format compatible with the web application.
- Run the Streamlit application by executing streamlit run frontend/pages/streamlit.py.
- Access the application in your web browser using the provided URL.
- Enter the customer details in the web application and receive the churn prediction.
- Explore the churn predictions for different customers and analyze the impact of different features.

## Conclusion

This project has developed a web application for churn prediction using customer data. By accurately predicting customer churn, businesses can implement targeted retention strategies and improve customer retention rates. The application provides users with real-time insights into customer churn, enabling proactive measures to reduce churn and enhance customer satisfaction.

Please note that the above steps assume that you have prepared your dataset, trained your machine learning model, and saved it in a compatible format. You may need to modify the code and adapt it to your specific requirements and dataset structure.

For more detailed instructions and code examples, refer to the documentation provided in the project repository.