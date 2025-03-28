from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import sys
import os

# Ensure app_utils can be imported
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
# Add the "src" directory to Python's module search path
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)  # Prioritize the "src" directory
from app_utils import Evaluation as evalu, Transform as trans, Tuning as tune, Plots as plt


# Define dir for the dataset to be extracted
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "datasets", "forecasting", "final")

st.title('Predictive Analysis for Supply Chain Management')

st.info('Machine Learning and Data Visualisation for actionable insights!')

with st.expander('Data'):
    # Show the processed dataset which will be used
    st.write('**Processed Dataset**')
    product_sales = pd.read_csv(os.path.join(
        data_dir, 'product_sales.csv'), sep=',', header=0)
    # Show entire dataset with scrolling for large data
    st.dataframe(product_sales)

    # Show the feature variables
    st.write('**Independant variables / Features (X)**')
    X = product_sales.drop('OrderQuantity', axis=1)
    st.dataframe(X)  # Show features in a scrollable table

    # Show the target variable
    st.write('**Dependant variable / Target (y)**')
    y = product_sales.OrderQuantity
    st.write(y)

with st.expander('Data Visualisations'):
    st.write('**Customer Order Quantity**')
    st.write(
        'Customer order quantity distribution. Customers shown are ABL, FRE, MOM and UND.')

    # Customers rows which will be plotted
    selected_prefixes = ['ALB', 'FRE', 'MOM', 'UND']

    # Extract prefix and filter customer code, everything before the first '-'
    product_sales['ProductGroup'] = product_sales['ProductNumber'].str.split(
        '-').str[0]
    filtered_data = product_sales[product_sales['ProductGroup'].isin(
        selected_prefixes)]

    # Define and filter the date range
    start_date = '2024-01-01'
    end_date = '2024-03-30'
    filtered_data = filtered_data[(filtered_data['OrderDate'] >= start_date) & (
        filtered_data['OrderDate'] <= end_date)]

    st.scatter_chart(data=filtered_data, x='OrderDate',
                     y='OrderQuantity', color='ProductGroup')

with st.expander('Demand Forecasting'):
    # Save user entered customer code
    customer_code = st.text_input("Enter a valid Customer Code, i.e ALB...")

    # Check if user has entered a customer code
    if customer_code:
        filtered_data = product_sales[product_sales['ProductGroup']
                                      == customer_code]

        if filtered_data.empty:
            st.write("No data found for the given customer code.")
        else:
            st.write(f"Data for customer code: {customer_code}")

        # Perform zscore removal for abnormally high OrderQuantities with related products
        # THIS IS CONTROVERSIAL IN THIS SCENARIO BECAUSE ITS REMOVING ACTUAL CUSTOMER ORDERS
        filtered_data = trans.compute_zscore(filtered_data)

        st.dataframe(filtered_data)  # Ensure it's scrollable if data is large

        # Show the feature variables
        st.write('**Independant variables / Features (X)**')
        X = filtered_data.drop('OrderQuantity', axis=1)
        st.dataframe(X)  # Show features in a scrollable table

        # Show the target variable
        st.write('**Dependant variable / Target (y)**')
        y = filtered_data.OrderQuantity
        st.write(y)

        st.info('Model predictions could take a few minutes...')

        # Code for models below
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42)

        X_train_preprocessed, X_val_preprocessed = trans.transform_data(
            X_train, X_val, X)

        # Stacking ensemble
        tune.stacked_ensemble_tuned.fit(X_train_preprocessed, y_train)

        # Get Stacked Ensemble predictions (validation data)
        val_pred_stacked = tune.stacked_ensemble_tuned.predict(
            X_val_preprocessed)
        train_pred_stacked = tune.stacked_ensemble_tuned.predict(
            X_train_preprocessed)

        overall_plt = plt.overall(
            filtered_data, y_val, X_val, val_pred_stacked, customer_code)
        st.pyplot(overall_plt)

        # Calculate and display base and advanced metrics of model preds
        metrics = evalu.metrics(y_val, val_pred_stacked)
        st.markdown(metrics, unsafe_allow_html=False)

        advanced_metrics = evalu.advanced_metrics(y_val, val_pred_stacked)
        st.markdown(advanced_metrics, unsafe_allow_html=False)

    else:
        st.write("Please enter a customer code to filter the data.")
