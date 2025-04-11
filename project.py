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

# Create a binary target variable based on OTP
otp_threshold = 90.0 # Can be adjusted
data['High OTP'] = (data['OTP'] > otp_threshold).astype(int)

print("\nValue Counts for High_OTP (Target Variable):")
print(data['High_OTP'].value_counts())

# Handle potential missing values in OTP
print("\nMissing values in OTP:", data['OTP'].isnull().sum())
data.dropna(subset=['OTP'], inplace=True)

# Convert 'Month' to a numerical or categorical representation if needed
print("\nValue Counts for Month:")
print(data['Month'].value_counts())
if data['Month'].dtype == 'object':
    month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    data['Month_Numeric'] = data['Month'].replace(month_mapping)
    data.drop('Month', axis=1, inplace=True)
elif data['Month'].dtype == 'int64' or data['Month'].dtype == 'float64':
    # If already numeric, we might want to treat it as categorical for some models
    data['Month'] = data['Month'].astype('category')

