import numpy as np
import pandas as pd
from scipy import stats
import os

# define dir for the datasets to be extracted
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "datasets",
                        "forecasting", "raw", "2022-2025")

# convert datasets to csv from txt
# \t used as separator, because of raw data format and the headers as row 0
product_sales = pd.read_csv(
    os.path.join(data_dir, 'product_sales.csv'), sep=',', header=0)  # stock


print('----- DROP BOOLEAN AND NULL COLUMNS (NOT NEEDED) -----\n')

# drop columns with all rows as missing values (NaN/0)
columns_to_drop = product_sales.columns[product_sales.isnull().all()].tolist()
print(
    f'Final Product Sales Data - Columns to be dropped (Missing Vals): {columns_to_drop}\n')
product_sales = product_sales.dropna(axis=1, how='all')

# set the name attribute for each DataFrame
product_sales.name = 'Final Product Sales Data'

# drop columns with all rows containing the same values of either 0, True, or False
for df in [product_sales]:
    columns_to_drop = [col for col in df.columns if df[col].nunique(
    ) == 1 and df[col].iloc[0] in [0, True, False]]
    print(f'{df.name} - Columns to be dropped (All cols with rows same vals): {columns_to_drop}\n')
    df.drop(columns=columns_to_drop, inplace=True)

print('-------------------------------------------------------\n')

# drop 'TicQuantity' column as it is for a order (where orders can include multiple products)
product_sales.drop(columns='TicQuantity', inplace=True)
# print('Dropped TicQuantity column\n')

# drop rows with '0' values in 'OrderQuantity' column
product_sales = product_sales[product_sales.OrderQuantity != 0]

# # display a confirmation message with the rows dropped count
print(f'Dropped rows with 0 values in OrderQuantity column. Rows dropped: {len(product_sales[product_sales.OrderQuantity == 0])}\n')

# display merged data after dropping columns
print('-------------- Final Product Sales Data ----------------')
print(product_sales.head())
# display ALL columns
print(product_sales.columns)

# # Write column names to a text file
# with open('columns.txt', 'w') as f:
#     for col in product_sales.columns:
#         f.write(f"{col}\n")

'''
TRYING DIFFERENT FEATURES TO SEE HOW THEY ACT WITH MODEL PERFORMANCE
'''

# ensure date columns are properly formatted
product_sales['OrderDate'] = pd.to_datetime(
    product_sales['OrderDate'], errors='coerce')
product_sales['Ship_by_Date'] = pd.to_datetime(
    product_sales['Ship_by_Date'], errors='coerce')
product_sales['order_date'] = product_sales['OrderDate']
product_sales['ship_by_date'] = product_sales['Ship_by_Date']

# time-based features
product_sales['order_month'] = product_sales['order_date'].dt.month
product_sales['order_week'] = product_sales['order_date'].dt.isocalendar().week
product_sales['order_year'] = product_sales['order_date'].dt.year
product_sales['order_weekday'] = product_sales['OrderDate'].dt.weekday
product_sales['is_weekend'] = (product_sales['order_weekday'] >= 5).astype(int)
product_sales['quarter'] = product_sales['OrderDate'].dt.quarter
product_sales['is_end_of_month'] = (
    product_sales['OrderDate'].dt.day > 25).astype(int)

# year-over-year growth (yoy)
product_sales['prev_year_sales'] = product_sales.groupby(
    'ProductNumber')['OrderQuantity'].shift(12)
product_sales['prev_week_sales'] = product_sales.groupby(
    'ProductNumber')['OrderQuantity'].shift(1)
product_sales['yoy_growth'] = (product_sales['OrderQuantity'] -
                             product_sales['prev_year_sales']) / product_sales['prev_year_sales']

product_sales['sales_2022'] = product_sales.apply(
    lambda x: x['OrderQuantity'] if x['order_year'] == 2022 else 0, axis=1)
product_sales['sales_2023'] = product_sales.apply(
    lambda x: x['OrderQuantity'] if x['order_year'] == 2023 else 0, axis=1)
product_sales['sales_2024'] = product_sales.apply(
    lambda x: x['OrderQuantity'] if x['order_year'] == 2024 else 0, axis=1)

# growth of product sales per year (%)
product_sales['growth_2023'] = (product_sales['sales_2023'] -
                              product_sales['sales_2022']) / product_sales['sales_2022'] * 100
product_sales['growth_2024'] = (product_sales['sales_2024'] -
                              product_sales['sales_2023']) / product_sales['sales_2023'] * 100


# lag features (considers past trends via products)
product_sales['prev_month_sales'] = product_sales.groupby(
    'ProductNumber')['OrderQuantity'].shift(1)
product_sales['prev_2_month_sales'] = product_sales.groupby('ProductNumber')[
    'OrderQuantity'].shift(2)
product_sales['prev_3_month_sales'] = product_sales.groupby('ProductNumber')[
    'OrderQuantity'].shift(3)

# time difference features
# product_sales['days_since_last_order'] = product_sales.groupby('ProductNumber')['OrderDate'].diff().dt.days.fillna(30)

# rolling features
product_sales['moving_avg_3m'] = product_sales.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
product_sales['moving_avg_6m'] = product_sales.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
product_sales['moving_avg_12m'] = product_sales.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=12, min_periods=1).mean())
product_sales['moving_avg_18m'] = product_sales.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=18, min_periods=1).mean())

# variance features
product_sales['var_1m'] = product_sales.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=1, min_periods=1).var())
product_sales['var_3m'] = product_sales.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=3, min_periods=1).var())
product_sales['var_6m'] = product_sales.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=6, min_periods=1).var())
product_sales['var_12m'] = product_sales.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=12, min_periods=1).var())
product_sales['var_18m'] = product_sales.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=18, min_periods=1).var())

# log-transformed variance
product_sales['log_var_1m'] = np.log1p(
    product_sales['var_1m'])  # log1p prevents log(0) errors
product_sales['log_var_3m'] = np.log1p(product_sales['var_3m'])
product_sales['log_var_6m'] = np.log1p(product_sales['var_6m'])
product_sales['log_var_12m'] = np.log1p(product_sales['var_12m'])
product_sales['log_var_18m'] = np.log1p(product_sales['var_18m'])

# calc z-score for (possibly) helping with outliers
product_sales['z_score'] = np.abs(stats.zscore(product_sales['OrderQuantity']))

# check and fill missing values
for col in [
    'prev_month_sales', 'prev_week_sales',
    'moving_avg_3m', 'moving_avg_6m', 'moving_avg_12m', 'moving_avg_18m',
    'prev_2_month_sales', 'prev_3_month_sales',
    'var_1m', 'var_3m', 'var_6m', 'var_12m', 'var_18m'
]:
    product_sales[col] = product_sales[col].fillna(product_sales[col].mean())

# product lifestyle (upcoming, declining, mature) - REDUCES METRICS AND PREDICTION ACCURACIES
# product_sales['product_lifecycle'] = product_sales.groupby('ProductNumber')['OrderQuantity'].transform(lambda x: np.where(x.rolling(window=12, min_periods=1).mean() > x.mean(), 'mature', 'new'))

# # interaction Features
# product_sales['interaction_1'] = product_sales['prev_month_sales'] * product_sales['var_12m']
# product_sales['interaction_2'] = product_sales['prev_week_sales'] * product_sales['var_12m']
# product_sales['interaction_3'] = product_sales['moving_avg_3m'] * product_sales['moving_avg_12m']
# product_sales['interaction_4'] = product_sales['prev_2_month_sales'] * product_sales['var_18m']
# product_sales['interaction_5'] = product_sales['prev_3_month_sales'] * product_sales['var_18m']

# # demand Factors
# product_sales['inventory_ratio'] = product_sales['PhysicalInv'] / (product_sales['OnOrder'] + 1)
# product_sales['is_backordered'] = product_sales['BackOrdered'].notna().astype(int)
# product_sales['customer_order_count'] = product_sales.groupby('Customer_Num')['OrderQuantity'].transform('count')
# product_sales['customer_avg_order'] = product_sales.groupby('Customer_Num')['OrderQuantity'].transform('mean')

# aggregation
product_sales = product_sales.groupby([  # group rows by:
    'ProductNumber',
    'order_year', 'order_month', 'order_week', 'order_weekday', 'is_weekend', 'OrderDate',
    'Customer_Num'
]).agg({  # include these columns with respective data
    'OrderQuantity': 'sum',
    'prev_month_sales': 'mean',
    'prev_week_sales': 'mean',
    'prev_2_month_sales': 'mean',
    'prev_3_month_sales': 'mean',
    'var_1m': 'mean',
    'var_3m': 'mean',
    'var_6m': 'mean',
    'var_12m': 'mean',
    'var_18m': 'mean',
    'log_var_1m': 'mean',
    'log_var_3m': 'mean',
    'log_var_6m': 'mean',
    'log_var_12m': 'mean',
    'log_var_18m': 'mean',
    'yoy_growth': 'mean',
    'moving_avg_3m': 'mean',
    'moving_avg_6m': 'mean',
    'moving_avg_12m': 'mean',
    'moving_avg_18m': 'mean',
    'sales_2022': 'sum',
    'sales_2023': 'sum',
    'sales_2024': 'sum',
    'growth_2023': 'mean',
    'growth_2024': 'mean',
}).reset_index()

# define dir for the transformed datasets to be saved
final_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
final_data_dir = os.path.join(base_dir, "datasets",
                              "forecasting", "final")

# # saves outputs of products_sales to '../datasets/forecasting/final'
# product_sales.to_csv(
#     os.path.join(final_data_dir, 'product_sales.csv'), index=False)

# save the columns to a txt file
# columns.to_csv('../datasets/forecasting/2022-2025/columns.txt', index=False)