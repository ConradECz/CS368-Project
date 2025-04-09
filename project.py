import pandas as pd
import matplotlib.pyplot as plt
try:
    data = pd.read_csv("MTA_Metro-North_On-Time_Performance__Beginning_2020.csv")
except FileNotFoundError:
    print("Error: CSV file not found. Please make sure the file path is correct.")
    exit()
# Display initial information about the dataset
print(data.head)

print("Dataset Info:")
data.info()

print("\nSummary Statistics:")
print(data.describe)

print("\nValue Counts for Object (Categorical) Features:")
for col in data.select_dtypes(include='object').columns:
    print(f"\n{col}:\n{data[col].value_counts()}")

# Identify attributes with insufficient variability
insufficient_variance_cols =[]
for col in data.columns:
    if data[col].nunique() <= 1:
        insufficient_variance_cols.append(col)
print(f"\nAttributes with insufficient variability: {insufficient_variance_cols}")

# Drop columns with insufficient variability:
data = data.drop(columns=insufficient_variance_cols)
print("\nDataset shape after dropping low variance columns:", data.shape)

print("\nValue Counts for OTP (first 20):")
print(data['OTP'].value_counts().head(20))
print("\nData type of OTP:", data['OTP'].dtype)
print("\nMin OTP:", data['OTP'].min())
print("\nMax OTP:", data['OTP'].max())

