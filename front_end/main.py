from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import sys
import os
from sklearn.model_selection import TimeSeriesSplit

# Ensure app_utils can be imported
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# fmt: off
from app_utils import Evaluation as evalu, Transform as trans, Tuning as tune, Plots as plt
# fmt: on

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
    filtered_data = filtered_data[(filtered_data['OrderDate'] >= start_date) &
                                  (filtered_data['OrderDate'] <= end_date)]

    st.scatter_chart(data=filtered_data, x='OrderDate',
                     y='OrderQuantity', color='ProductGroup')

with st.expander('Demand Forecasting'):
    model_choices = [
        'Random Forest',
        'Multi-Layer Perceptron (MLP/Neural Network)',
        'XGBoost',
        'CatBoost',
        'N-BEATS (Time Series)',
        'Model with best predictions'
    ]

    chosen_model = st.selectbox('Select learning model', model_choices)
    # Save user entered customer code
    customer_code = st.text_input('Enter a valid Customer Code, i.e ALB...')

    # Check if user has entered a customer code
    if customer_code:
        filtered_data = product_sales[product_sales['ProductGroup']
                                      == customer_code]

        if filtered_data.empty:
            st.write('No data found for the given customer code.')
        else:
            st.write(f'Data for customer code: {customer_code}')

            # Perform zscore removal for abnormally high OrderQuantities with related products
            # THIS IS CONTROVERSIAL IN THIS SCENARIO BECAUSE ITS REMOVING ACTUAL CUSTOMER ORDERS
            filtered_data = trans.compute_zscore(filtered_data)
            # Ensure it's scrollable if data is large
            st.dataframe(filtered_data)

            # Show the feature variables
            st.write('**Independant variables / Features (X)**')
            X = filtered_data.drop('OrderQuantity', axis=1)
            st.dataframe(X)

            # Show the target variable
            st.write('**Dependant variable / Target (y)**')
            y = filtered_data.OrderQuantity
            st.write(y)

            # '''
            # ML model Train and test code --->
            # '''
            st.info('Model predictions could take a few minutes...')

            # Define models with tuned params
            models = {
                'Random Forest': tune.rf_tuned,
                'MLP': tune.mlp_tuned,
                'XGBoost': tune.xbg_tuned,
                'CatBoost': tune.catb_tuned,
                # 'N-BEATS': tune.nbeats_tuned
            }

            results = []
            tscv = TimeSeriesSplit(n_splits=5)

            # Run all the models and display the model with the best performance for the cust
            if chosen_model == 'Model with best predictions':
                for name, model in models.items():
                    with st.spinner(f'Running {name}...'):
                        try:
                            results.append(evalu.run_model(
                                name, model, X, y, filtered_data, customer_code))
                        except Exception as e:
                            st.error(f"Error with {name}: {str(e)}")

                # Display model with best performance
                st.subheader("Model Comparison")
                comparison_df = pd.DataFrame(results).T
                st.dataframe(comparison_df.style.highlight_min(
                    axis=0, color='#fffd75'))
                best_model = comparison_df['RMSE'].idxmin()
                st.success(
                    f"Best performing model: {best_model} (Lowest RMSE)")

            else:  # Run the single model the user selected
                with st.spinner(f'Running {chosen_model}...'):
                    try:
                        evalu.run_model(
                            chosen_model, models[chosen_model], X, y, filtered_data, customer_code)
                    except Exception as e:
                        st.error(f"Error with {chosen_model}: {str(e)}")
    else:  # User entered customer isn't avaliable
        st.write("Please enter a customer code to filter the data.")
