import numpy as np
import pandas as pd
from scipy import stats
import os

# define dir for the datasets to be extracted
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "datasets",
                        "stock_forecasting", "raw", "2022-2025")

# convert datasets to csv from txt
# \t used as separator, because of raw data format and the headers as row 0
products_s = pd.read_csv(
    os.path.join(data_dir, '[LT] Products [STOCK].txt'), sep='\t', header=0)  # stock
tickets_c_i = pd.read_csv(
    os.path.join(data_dir, '[LT] Tickets [CUSTOM] [ITEMS].txt'), sep='\t', header=0)  # customer order items
# customer order main data
tickets_c_m = pd.read_csv(
    os.path.join(data_dir, '[LT] Tickets [CUSTOM] [MAIN].txt'), sep='\t', header=0)
# customer order main data
sp_inv_adds = pd.read_csv(
    os.path.join(data_dir, '[LT] SP Inventory [ADDS].txt'), sep='\t', header=0)
# customer order main data
sp_inv_rel = pd.read_csv(
    os.path.join(data_dir, '[LT] SP Inventory [REL].txt'), sep='\t', header=0)

# # display the first 5 rows of the datasets
# print('-------------- Product [STOCK] ----------------')
# print(products_s.head())

# print('-------------- Tickets [CUSTOM] [ITEMS] ----------------')
# print(tickets_c_i.head())

# print('-------------- Tickets [CUSTOM] [MAIN] ----------------')
# print(tickets_c_m.head())
# print('------------------------------')

# change product stock column 'ProductNo' to 'ProductNumber' to match the other datasets
products_s.rename(columns={'ProductNo': 'ProductNumber'}, inplace=True)

# change the column name of 'Number' to 'TicketNumber'
tickets_c_m.rename(columns={'Number': 'TicketNumber'}, inplace=True)

# tickets_c_m.head()

#  merge the datasets
merged_data = pd.merge(tickets_c_m, tickets_c_i, on='TicketNumber')
merged_data = pd.merge(merged_data, products_s, on='ProductNumber')

print('----- DROP BOOLEAN AND NULL COLUMNS (NOT NEEDED) -----\n')

# drop columns with all rows as missing values (NaN/0)
columns_to_drop = merged_data.columns[merged_data.isnull().all()].tolist()
print(
    f'Merged Data - Columns to be dropped (Missing Vals): {columns_to_drop}\n')
merged_data = merged_data.dropna(axis=1, how='all')

# set the name attribute for each DataFrame
merged_data.name = 'Merged Data'

# drop columns with all rows containing the same values of either 0, True, or False
for df in [merged_data]:
    columns_to_drop = [col for col in df.columns if df[col].nunique(
    ) == 1 and df[col].iloc[0] in [0, True, False]]
    print(f'{df.name} - Columns to be dropped (All cols with rows same vals): {columns_to_drop}\n')
    df.drop(columns=columns_to_drop, inplace=True)

print('-------------------------------------------------------\n')

# drop 'TicQuantity' column as it is for a order (where orders can include multiple products)
merged_data.drop(columns='TicQuantity', inplace=True)
# print('Dropped TicQuantity column\n')

# drop rows with '0' values in 'OrderQuantity' column
merged_data = merged_data[merged_data.OrderQuantity != 0]

# # display a confirmation message with the rows dropped count
# print(f'Dropped rows with 0 values in OrderQuantity column. Rows dropped: {len(merged_data[merged_data.OrderQuantity == 0])}\n')

# # display merged data after dropping columns
# print('-------------- MERGED DATA ----------------')
# print(merged_data.head())
# # display ALL columns
# print(merged_data.columns)

# # Write column names to a text file
# with open('columns.txt', 'w') as f:
#     for col in merged_data.columns:
#         f.write(f"{col}\n")

# print('-------------------------------------------')

# # print all customername and suppliername which arent null in seperate dfs
# # print('-------------- CUSTOMER NAMES ----------------')
# # print(merged_data[merged_data.CustomerName.notnull()]['CustomerName'].unique())
# # print('----------------------------------------------')

# print('-------------- SUPPLIER NAMES ----------------')
# print(merged_data[merged_data.SupplierName.notnull()]['SupplierName'].unique())
# print('----------------------------------------------')


'''
TRYING DIFFERENT FEATURES TO SEE HOW THEY ACT WITH MODEL PERFORMANCE
'''

# ensure date columns are properly formatted
merged_data['OrderDate'] = pd.to_datetime(
    merged_data['OrderDate'], errors='coerce')
merged_data['Ship_by_Date'] = pd.to_datetime(
    merged_data['Ship_by_Date'], errors='coerce')
merged_data['order_date'] = merged_data['OrderDate']
merged_data['ship_by_date'] = merged_data['Ship_by_Date']

# time-based features
merged_data['order_month'] = merged_data['order_date'].dt.month
merged_data['order_week'] = merged_data['order_date'].dt.isocalendar().week
merged_data['order_year'] = merged_data['order_date'].dt.year
merged_data['order_weekday'] = merged_data['OrderDate'].dt.weekday
merged_data['is_weekend'] = (merged_data['order_weekday'] >= 5).astype(int)
merged_data['quarter'] = merged_data['OrderDate'].dt.quarter
merged_data['is_end_of_month'] = (
    merged_data['OrderDate'].dt.day > 25).astype(int)

# year-over-year growth (yoy)
merged_data['prev_year_sales'] = merged_data.groupby(
    'ProductNumber')['OrderQuantity'].shift(12)
merged_data['prev_week_sales'] = merged_data.groupby(
    'ProductNumber')['OrderQuantity'].shift(1)
merged_data['yoy_growth'] = (merged_data['OrderQuantity'] -
                             merged_data['prev_year_sales']) / merged_data['prev_year_sales']

merged_data['sales_2022'] = merged_data.apply(
    lambda x: x['OrderQuantity'] if x['order_year'] == 2022 else 0, axis=1)
merged_data['sales_2023'] = merged_data.apply(
    lambda x: x['OrderQuantity'] if x['order_year'] == 2023 else 0, axis=1)
merged_data['sales_2024'] = merged_data.apply(
    lambda x: x['OrderQuantity'] if x['order_year'] == 2024 else 0, axis=1)

# growth of product sales per year (%)
merged_data['growth_2023'] = (merged_data['sales_2023'] -
                              merged_data['sales_2022']) / merged_data['sales_2022'] * 100
merged_data['growth_2024'] = (merged_data['sales_2024'] -
                              merged_data['sales_2023']) / merged_data['sales_2023'] * 100


# lag features (considers past trends via products)
merged_data['prev_month_sales'] = merged_data.groupby(
    'ProductNumber')['OrderQuantity'].shift(1)
merged_data['prev_2_month_sales'] = merged_data.groupby('ProductNumber')[
    'OrderQuantity'].shift(2)
merged_data['prev_3_month_sales'] = merged_data.groupby('ProductNumber')[
    'OrderQuantity'].shift(3)

# time difference features
# merged_data['days_since_last_order'] = merged_data.groupby('ProductNumber')['OrderDate'].diff().dt.days.fillna(30)

# rolling features
merged_data['moving_avg_3m'] = merged_data.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
merged_data['moving_avg_6m'] = merged_data.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=6, min_periods=1).mean())
merged_data['moving_avg_12m'] = merged_data.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=12, min_periods=1).mean())
merged_data['moving_avg_18m'] = merged_data.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=18, min_periods=1).mean())

merged_data['var_1m'] = merged_data.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=1, min_periods=1).var())
merged_data['var_3m'] = merged_data.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=3, min_periods=1).var())
merged_data['var_6m'] = merged_data.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=6, min_periods=1).var())
merged_data['var_12m'] = merged_data.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=12, min_periods=1).var())
merged_data['var_18m'] = merged_data.groupby('ProductNumber')[
    'OrderQuantity'].transform(lambda x: x.rolling(window=18, min_periods=1).var())

# log-transformed variance
merged_data['log_var_1m'] = np.log1p(
    merged_data['var_1m'])  # log1p prevents log(0) errors
merged_data['log_var_3m'] = np.log1p(merged_data['var_3m'])
merged_data['log_var_6m'] = np.log1p(merged_data['var_6m'])
merged_data['log_var_12m'] = np.log1p(merged_data['var_12m'])
merged_data['log_var_18m'] = np.log1p(merged_data['var_18m'])

# calc z-score for (possibly) helping with outliers
merged_data['z_score'] = np.abs(stats.zscore(merged_data['OrderQuantity']))

# check and fill missing values
for col in [
    'prev_month_sales', 'prev_week_sales',
    'moving_avg_3m', 'moving_avg_6m', 'moving_avg_12m', 'moving_avg_18m',
    'prev_2_month_sales', 'prev_3_month_sales',
    'var_1m', 'var_3m', 'var_6m', 'var_12m', 'var_18m'
]:
    merged_data[col] = merged_data[col].fillna(merged_data[col].mean())

# product lifestyle (upcoming, declining, mature) - REDUCES METRICS AND PREDICTION ACCURACIES
# merged_data['product_lifecycle'] = merged_data.groupby('ProductNumber')['OrderQuantity'].transform(lambda x: np.where(x.rolling(window=12, min_periods=1).mean() > x.mean(), 'mature', 'new'))

# # interaction Features
# merged_data['interaction_1'] = merged_data['prev_month_sales'] * merged_data['var_12m']
# merged_data['interaction_2'] = merged_data['prev_week_sales'] * merged_data['var_12m']
# merged_data['interaction_3'] = merged_data['moving_avg_3m'] * merged_data['moving_avg_12m']
# merged_data['interaction_4'] = merged_data['prev_2_month_sales'] * merged_data['var_18m']
# merged_data['interaction_5'] = merged_data['prev_3_month_sales'] * merged_data['var_18m']

# # demand Factors
# merged_data['inventory_ratio'] = merged_data['PhysicalInv'] / (merged_data['OnOrder'] + 1)
# merged_data['is_backordered'] = merged_data['BackOrdered'].notna().astype(int)
# merged_data['customer_order_count'] = merged_data.groupby('Customer_Num')['OrderQuantity'].transform('count')
# merged_data['customer_avg_order'] = merged_data.groupby('Customer_Num')['OrderQuantity'].transform('mean')

# aggregation
product_sales = merged_data.groupby([  # group rows by:
    'ProductNumber',
    'order_year', 'order_month', 'order_week', 'order_weekday', 'is_weekend',
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

# add the more feature columns to the product_sales dataset
product_sales = pd.merge(
    product_sales, products_s[['ProductNumber', 'PhysicalInv']], on='ProductNumber')

# sales data for SARIMAX time series model test
# sales_data = merged_data.groupby(['OrderDate'])['OrderQuantity'].sum()

sales_data = merged_data['OrderDate'] = pd.to_datetime(
    merged_data['OrderDate'])

# 'OrderDate' to datetime
merged_data['OrderDate'] = pd.to_datetime(merged_data['OrderDate'])

# index correctly
sales_data = merged_data.set_index(['OrderDate', 'ProductNumber'])
sales_data = sales_data['OrderQuantity']

# define dir for the transformed datasets to be saved
final_base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
final_data_dir = os.path.join(base_dir, "datasets",
                              "stock_forecasting", "final")

# saves outputs of products_sales to '../datasets/stock_forecasting/final'
product_sales.to_csv(
    os.path.join(final_data_dir, 'product_sales.csv'), index=False)

# sales_data.head()

# # extract all columns from all datasets and save them to a txt file (products_s, tickets_c_i, tickets_c_m)
# # extract columns from each dataset and save them to a txt file one by one
# columns = pd.DataFrame()

# # concatenate columns from products_s
# columns = pd.concat([columns, pd.DataFrame(
#     products_s.columns, columns=['Column'])])

# # concatenate columns from tickets_c_i
# columns = pd.concat([columns, pd.DataFrame(
#     tickets_c_i.columns, columns=['Column'])])

# # concatenate columns from tickets_c_m
# columns = pd.concat([columns, pd.DataFrame(
#     tickets_c_m.columns, columns=['Column'])])

# # concatenate columns from sp_inv_adds
# columns = pd.concat([columns, pd.DataFrame(
#     sp_inv_adds.columns, columns=['Column'])])

# # concatenate columns from sp_inv_rel
# columns = pd.concat([columns, pd.DataFrame(
#     sp_inv_rel.columns, columns=['Column'])])

# save the columns to a txt file
# columns.to_csv('../datasets/stock_forecasting/2022-2025/columns.txt', index=False)
