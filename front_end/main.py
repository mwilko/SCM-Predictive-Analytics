import streamlit as st
import pandas as pd
import sys
import os

# Add the absolute path to the 'src' directory
sys.path.append('D:/SCM---ML-Visualisation/src')
from app_utils import Transform as trans, Tuning as tune, Plots as plt
from sklearn.model_selection import train_test_split

# Define dir for the dataset to be extracted
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "datasets", "forecasting", "final")

st.title('Predictive Analytics for Supply Chain Management')

st.info('This app uses machine learning models to predict a certain criteria')

with st.expander('Data'):
    # Show the processed dataset which will be used
    st.write('**Processed Dataset**')
    product_sales = pd.read_csv(os.path.join(data_dir, 'product_sales.csv'), sep=',', header=0)
    st.dataframe(product_sales)  # Show entire dataset with scrolling for large data
    
    # Show the feature variables
    st.write('**X**')
    X = product_sales.drop('OrderQuantity', axis=1)
    st.dataframe(X)  # Show features in a scrollable table
    
    # Show the target variable
    st.write('**y**')
    y = product_sales.OrderQuantity
    st.write(y)

with st.expander('Data Visualisations'):
    st.write('**Customer Order Quantity**')
    st.write('Customer order quantity distribution. Customers shown are ABL, FRE, MOM and UND.')

    # Customers rows which will be plotted
    selected_prefixes = ['ALB', 'FRE', 'MOM', 'UND']
    
    # Extract prefix and filter customer code, everything before the first '-'
    product_sales['ProductGroup'] = product_sales['ProductNumber'].str.split('-').str[0]
    filtered_data = product_sales[product_sales['ProductGroup'].isin(selected_prefixes)]

    # Define and filter the date range
    start_date = '2024-01-01'
    end_date = '2024-03-30'
    filtered_data = filtered_data[(filtered_data['OrderDate'] >= start_date) & (filtered_data['OrderDate'] <= end_date)]

    st.scatter_chart(data=filtered_data, x='OrderDate', y='OrderQuantity', color='ProductGroup')

with st.expander('Demand Forecasting'):
    # Save user entered customer code
    customer_code = st.text_input("Enter a valid Customer Code, i.e ALB...")

    # Check if user has entered a customer code
    if customer_code:
        filtered_data = product_sales[product_sales['ProductGroup'] == customer_code]

        if filtered_data.empty:
            st.write("No data found for the given customer code.")
        else:
            st.write(f"Data for customer code: {customer_code}")
            st.dataframe(filtered_data)  # Ensure it's scrollable if data is large
    else:
        st.write("Please enter a customer code to filter the data.")

    # Code for models below
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_preprocessed, X_val_preprocessed = trans.transform_data(X_train, X_val, X)

    # # Fit and evaluate the model on both train and validation sets
    # tune.rf_tuned.fit(X_train_preprocessed, y_train)

    # # Get RF predictions (validation data)
    # y_pred_rf = tune.rf_tuned.predict(X_val_preprocessed)

    # # Fit and evaluate the model on both train and validation sets
    # tune.mlp_tuned.fit(X_train_preprocessed, y_train)

    # # Get MLP predictions (validation data)
    # y_pred_mlp = tune.mlp_tuned.predict(X_val_preprocessed)

    # # Fit and evaluate the model on both train and validation sets
    # tune.xbg_tuned.fit(X_train_preprocessed, y_train)

    # # Get XGB predictions (validation data)
    # y_pred_xgb = tune.xbg_tuned.predict(X_val_preprocessed)

    # Stacking ensemble
    tune.stacked_ensemble_tuned.fit(X_train_preprocessed, y_train)

    # Get Stacked Ensemble predictions (validation data)
    val_pred_stacked = tune.stacked_ensemble_tuned.predict(X_val_preprocessed)
    train_pred_stacked = tune.stacked_ensemble_tuned.predict(X_train_preprocessed)

    plt.overall(filtered_data, y_val, X_val, val_pred_stacked, customer_code)







