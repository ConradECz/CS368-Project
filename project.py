import pandas as pd
# import matplotlib.pyplot as plt
try:
    data = pd.read_csv("MTA_Metro-North_On-Time_Performance__Beginning_2020.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: CSV file not found. Please make sure the file path is correct.")
    exit()

#Print the columns immediately after loading
print("\nColumns in the DataFrame after loading:")
print(data.columns)

# Function to clean percentage columns
def clean_percentage_column(df, column_name):
    if column_name in df.columns and df[column_name].dtype == 'object':
        print(f"\nAttempting to clean and convert '{column_name}'...")
        try:
            df[column_name] = df[column_name].str.rstrip('%').astype('float') / 100.0
            print(f"'{column_name}' column cleaned and converted to numeric (decimal).")
            print(f"Data type of '{column_name}' after conversion:", df[column_name].dtype)
        except Exception as e:
            print(f"Error during '{column_name}' cleaning/conversion: {e}")
    elif column_name in df.columns:
        print(f"\n'{column_name}' column is already numeric.")
    else:
        print(f"\n'{column_name}' column not found!")
    return df
# Clean '
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

# Clean and convert the 'OTP' column to numeric (float)
if data['OTP'].dtype == 'object':
    data['OTP'] = data['OTP'].str.rstrip('%').astype('float') / 100.0
    print("\n'OTP' column cleaned and converted to numeric (decimal).")
else:
    print("\n'OTP' column is already numeric.")

print("\nData type of OTP after conversion:", data['OTP'].dtype)
print("\nMin OTP:", data['OTP'].min())
print("\nMax OTP:", data['OTP'].max())

# Create a binary target variable based on OTP (decimal percentage)
otp_threshold = 0.90 # Can be adjusted (90% as decimal)
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

# Encode 'Branch/Line' (categorical feature)
print("\nValue Counts for Branch/Line:")
print(data['Branch/Line'].value_counts())
data = pd.get_dummies(data, columns=['Branch/Line'], drop_first=True)

# Convert Peak hour columns to binary (if they are not already)
for col in ['AM Peak', 'PM Peak', 'Off Peak']:
    if col in data.columns and data[col].dtype == 'object':
        print(f"\nValue Counts for {col}:")
        print(data[col].value_counts())
        # Assuming values are like 'Yes'/'No' or 'True'/'False'
        data[col] = data[col].apply(lambda x: 1 if str(x).lower() in ['yes', 'true'] else 0)
    elif col in data.columns and data[col].dtype in ['int64', 'float64']:
        # Already numeric, might need to ensure it's binary (0 or 1)
        data[col] = data[col].astype(int)

print("\nMissing Values after initial handling:")
print(data.isnull().sum())

print("\nProcessed Data Info (so far):")
data.info()

# Now have binary target variable. Splitting the data into features (X) and target (y)
# and then into training and testing sets.

# Example of splitting the data:
from sklearn.model_selection import train_test_split

X = data.drop('High_OTP', axis=1)
y = data['High_OTP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nShape of training data:", X_train.shape, y_train.shape)
print("Shape of testing data:", X_test.shape, y_test.shape)