import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
import os
import numpy as np
import torch

# Define directory for the datasets to be extracted
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_dir = os.path.join(base_dir, "raw")

# Load data structure dataset
df = pd.read_csv(os.path.join(data_dir, 'dataset_structure.csv'), sep=',', header=0)
print("Original Data Shape:", df.shape)

# Handle infinite and large values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill NaNs with appropriate values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)  # Fill categorical columns with mode
    else:
        df[col].fillna(df[col].median(), inplace=True)  # Fill numeric columns with median

print("Data Shape after Cleaning:", df.shape)

# Ensure the dataframe is not empty
if df.empty:
    raise ValueError("Dataset is empty after cleaning. Please check the input data.")

# Reduce unique categorical values by grouping rare ones
if 'ProductNumber' in df.columns:
    top_categories = df['ProductNumber'].value_counts().index[:500]  # Keep top 500 categories
    df.loc[~df['ProductNumber'].isin(top_categories), 'ProductNumber'] = "OTHER"

# Define metadata
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)

# Explicitly set 'ProductNumber' as categorical to prevent SDV anonymisation
metadata.update_column(column_name="ProductNumber", sdtype="categorical")

# Check for GPU availability (informational only)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Initialize and train the CTGAN synthesizer
model = CTGANSynthesizer(metadata, epochs=100)  # Reduced epochs from 300 to 100
model.fit(df)

# Generate synthetic data
synthetic_data = model.sample(num_rows=50000)  # Generate 50,000 rows

# Ensure 'ProductNumber' follows similar patterns
if 'ProductNumber' in df.columns:
    real_product_numbers = df['ProductNumber'].unique()
    synthetic_data['ProductNumber'] = np.random.choice(real_product_numbers, size=len(synthetic_data), replace=True)

print(synthetic_data.head())

# Save synthetic dataset
synthetic_data.to_csv("product_sales.csv", index=False)