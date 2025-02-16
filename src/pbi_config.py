from powerbiclient import QuickVisualize, get_dataset_config, Report
from powerbiclient.authentication import InteractiveLoginAuthentication, DeviceCodeLoginAuthentication
import pandas as pd

print("Reading CSV file...")
prod_sales = pd.read_csv(
    '/Users/mwilko777/Desktop/Project/Software Artifact/datasets/stock_forecasting/final/product_sales.csv')
print("CSV file read successfully.")

# use DeviceCodeLoginAuthentication if want to sign in each time, else InteractiveLoginAuthentication
print("Authenticating...")
auth = DeviceCodeLoginAuthentication()
print("Authentication successful.")


def visual():
    print("Creating dataset configuration...")
    dataset_config = get_dataset_config(prod_sales)
    print("Dataset configuration created.")
    print("Creating QuickVisualize object...")
    return QuickVisualize(dataset_config, auth)
