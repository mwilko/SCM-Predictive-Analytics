import matplotlib.pyplot as plt
import seaborn as sns
from datasets import product_sales
import sys
sys.path.append('../src')


class ProductHandler:
    # Class variable to store customer code data across all instances
    custom_code_dict = {}

    @classmethod
    def custom_prod_set(cls):
        # Create a set to store unique customer codes
        products_by_customer = set()

        # Loop through all product numbers
        for index, row in product_sales.iterrows():
            product = row['ProductNumber']
            for i in range(len(product) - 2):
                # Get the first three characters of the product number (customer code)
                customer_code = product[i:i+3]

                # Add the customer code to the set
                products_by_customer.add(customer_code)

                # Create or update the list for each customer code
                if customer_code in cls.custom_code_dict:
                    cls.custom_code_dict[customer_code].append(row.to_dict())
                else:
                    cls.custom_code_dict[customer_code] = [row.to_dict()]

        print(
            f"All custom codes for existing products: \n{products_by_customer}\n------------------------------------")

    @classmethod
    def get_custom_code_data(cls, customer_code):
        # Retrieve the customer code list from the class-level dictionary
        return cls.custom_code_dict.get(customer_code, None)


def feature_importance(df):
    # Select relevant numerical features
    numerical_features = [
        'OrderQuantity', 'prev_month_sales', 'prev_week_sales', 'prev_2_month_sales',
        'prev_3_month_sales', 'var_1m', 'var_3m', 'var_6m', 'var_12m', 'var_18m',
        'log_var_1m', 'log_var_3m', 'log_var_6m', 'log_var_12m', 'log_var_18m',
        'yoy_growth', 'moving_avg_3m', 'moving_avg_6m', 'moving_avg_12m', 'moving_avg_18m',
        'sales_2022', 'sales_2023', 'sales_2024', 'growth_2023', 'growth_2024'
    ]

    # Compute correlation matrix
    corr_matrix = df[numerical_features].corr()

    # Create heatmap
    plt.figure(figsize=(12, 8))  # Adjust size for better readability
    sns.heatmap(corr_matrix[['OrderQuantity']],
                cmap='Blues', annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation with Order Quantity")
    plt.show()
