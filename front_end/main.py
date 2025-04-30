from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
import sys
import os
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure app_utils can be imported
current_dir = os.path.dirname(__file__)
src_dir = os.path.abspath(os.path.join(current_dir, "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# fmt: off
from app_utils import Evaluation as evalu, Transform as trans, Tuning as tune, Plots as plots
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
    
    # Show the distribution of popular customer order quantities
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
    
    # Monthly aggregate trend + moving average
    st.write('**Monthly Sales Trend & 3-Month Moving Average**')
    product_sales['OrderDate'] = pd.to_datetime(product_sales['OrderDate'])
    monthly = (
        product_sales
        .set_index('OrderDate')
        .resample('M')['OrderQuantity']
        .sum()
        .to_frame()
    )
    monthly['MovingAverage3m'] = monthly['OrderQuantity'].rolling(3).mean()
    st.line_chart(monthly)
    
    # Weekday vs. month heatmap to expose seasonality
    st.write('**Avg Order Quantity: Weekday × Month**')
    pivot = product_sales.pivot_table(
        index='order_weekday',
        columns='order_month',
        values='OrderQuantity',
        aggfunc='mean'
    )
    fig, ax = plt.subplots(figsize=(6,3))
    cax = ax.matshow(pivot, aspect='auto', cmap='Oranges')
    # for (i, j), val in np.ndenumerate(pivot.values): # Uncomment to show values in each cell
    #     ax.text(j, i, f"{val:.0f}", ha='center', va='center', fontsize=8)
    ax.set_xticks(range(12)); ax.set_xticklabels(range(1,13))
    ax.set_yticks(range(7)); ax.set_yticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
    ax.set_xlabel('Month'); ax.set_ylabel('Weekday')
    fig.colorbar(cax, label='Avg Qty')
    st.pyplot(fig)

    # Feature - Target Correlations
    st.write('**Feature Correlation to Order Quantity**')
    numerical_features = [
        'OrderQuantity', 'prev_month_sales', 'prev_week_sales', 'prev_2_month_sales',
        'prev_3_month_sales', 'var_3m', 'var_6m', 'var_12m', 'var_18m',
        'log_var_3m', 'log_var_6m', 'log_var_12m', 'log_var_18m',
        'yoy_growth', 'moving_avg_3m', 'moving_avg_6m', 'moving_avg_12m', 'moving_avg_18m',
        'sales_2022', 'sales_2023', 'sales_2024'
    ]
    corr_matrix = product_sales[numerical_features].corr()

    # disable background gridlines
    sns.set_style("white", {"axes.grid": False})

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    sns.heatmap(
        corr_matrix[['OrderQuantity']] * 100, # Convert to percentage
        cmap='Oranges',
        annot=True,
        fmt=".2f",
        linewidths=0, # No lines between cells
        cbar_kws={'label': 'Correlation (%)'},
        ax=ax2
    )
    ax2.set_ylabel("")  
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    sns.despine(ax=ax2, left=True, bottom=True) # Remove gridlines and ticks
    st.pyplot(fig2)

with st.expander('Demand Forecasting'):
    # Create tabs for model selection, data view, and results
    select_tab, data_tab, results_tab = st.tabs(
        ["Select Model", "Data View", "Results"])

    with select_tab:
        model_choices = [
            'Best Predictive Accuracy',
            'Random Forest',
            'Multi-Layer Perceptron (MLP/Neural Network)',
            'XGBoost',
            'CatBoost',
            'Stacking Ensemble (All-in-one)'
        ]

        chosen_model = st.selectbox('Select learning model', model_choices)
        # Save user entered customer code
        customer_code = st.text_input(
            'Enter a valid Customer Code, i.e ALB...')

    with data_tab:
        # Only show data if user has entered a customer code
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

                # Display the filtered data
                st.subheader("Filtered Data")
                st.dataframe(filtered_data)

                # Show the feature variables
                st.subheader('**Independent variables / Features (X)**')
                X = filtered_data.drop('OrderQuantity', axis=1)
                st.dataframe(X)

                # Show the target variable
                st.subheader('**Dependent variable / Target (y)**')
                y = filtered_data.OrderQuantity
                st.write(y)
        else:
            st.write("Please enter a customer code to view data.")

    with results_tab:
        # Only show results if user has entered a customer code
        if customer_code:
            filtered_data = product_sales[product_sales['ProductGroup']
                                          == customer_code]

            if filtered_data.empty:
                st.write('No data found for the given customer code.')
            else:
                # Ensure data is processed - we need to make sure we use the same processed data from the data tab
                filtered_data = trans.compute_zscore(filtered_data)
                X = filtered_data.drop('OrderQuantity', axis=1)
                y = filtered_data.OrderQuantity

                # '''
                # ML model Train and test code --->
                # '''
                st.info('Model predictions could take a few minutes...')

                # Define models with tuned params
                models = {
                    'Random Forest': tune.rf_tuned,
                    'Multi-Layer Perceptron (MLP/Neural Network)': tune.mlp_tuned,
                    'XGBoost': tune.xbg_tuned,
                    'CatBoost': tune.catb_tuned,
                    'Stacked Ensemble (All-in-one)': tune.stacked_ensemble_tuned,
                }

                # Run all the models and display the model with the best performance for the customer
                if chosen_model == 'Best Predictive Accuracy':
                    model_results = {}  # Initialize dictionary to hold results from all models

                    for name, model in models.items():
                        if name == 'Stacked Ensemble (All-in-one)':
                            st.info('This model could take about 10-15 minutes to run...')
                        with st.spinner(f'Running {name}...'):
                            try:
                                model_results[name] = evalu.run_model(
                                    name, model, X, y, filtered_data, customer_code)
                            except Exception as e:
                                st.error(f"Error with {name}: {str(e)}")

                    if model_results:
                        for model, results in model_results.items():
                            model_results[model] = {metric: f"{value:.4f}" if isinstance(value, (int, float)) else value
                                                    for metric, value in results.items()}

                        comparison_df = pd.DataFrame(model_results)
                        # Transpose so models are columns and metrics are rows
                        comparison_df = comparison_df.T

                        st.subheader("Model Comparison")
                        # Create a styled dataframe - highlight minimum values for error metrics
                        error_metrics = ['RMSE', 'MAE', 'MSE']
                        styled_df = comparison_df.style.highlight_min(
                            axis=0, subset=error_metrics, color='#5fbb08')

                        # Highlight maximum values for R2 for variance capture
                        styled_df = styled_df.highlight_max(
                            axis=0, subset=['R2'], color='#6bd10a')

                        st.dataframe(styled_df)

                        # Find best model based on RMSE (Avg order quantity error value)
                        if 'RMSE' in comparison_df.columns:
                            best_model = comparison_df['RMSE'].idxmin()
                            st.success(
                                f"Best performing model: {best_model} (Lowest RMSE)")

                else:  # Run the single model the user selected
                    if chosen_model in models:  # Check if model exists in the dictionary
                        with st.spinner(f'Running {chosen_model}...'):
                            try:
                                evalu.run_model(
                                    chosen_model, models[chosen_model], X, y, filtered_data, customer_code)
                            except Exception as e:
                                st.error(
                                    f"Error with {chosen_model}: {str(e)}")
                    else:
                        st.error(
                            f"Model '{chosen_model}' is not implemented or available.")
        else:
            st.write("Please enter a customer code to view results.")
