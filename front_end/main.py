import streamlit as st
import pandas as pd

st.title('Predictive Analytics for Supply Chain Management')

st.info('This app uses machine learning models to predict a certain criteria')

with st.expander('Data'):
    st.write('**Processed Dataset**')
    product_sales = pd.read_cv('..\datasets\forecasting\final\product_sales.csv')
    product_sales.head()

    st.write('**X**')
    X = product_sales.drop('OrderQuantity', axis=1)

    st.write('**y**')
    y = product_sales.OrderQuantity