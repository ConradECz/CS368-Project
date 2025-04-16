import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt
# from aequitas.group import Group
# from aequitas.plotting import Plot
# from aequitas.bias import Bias
# from aequitas.fairness import Fairness
try:
    data = pd.read_csv("MTA_Metro-North_On-Time_Performance__Beginning_2020.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: CSV file not found. Please make sure the file path is correct.")
    exit()

print("Dataset Info:")
data.info()

print("\nSummary Statistics:")
print(data.describe)

# Identify attributes with insufficient variability
insufficient_variance_cols =[]
for col in data.columns:
    if data[col].nunique() <= 1:
        insufficient_variance_cols.append(col)
print(f"\nAttributes with insufficient variability: {insufficient_variance_cols}")

# Drop columns with insufficient variability:
data = data.drop(columns=insufficient_variance_cols)
print("\nDataset shape after dropping low variance columns:", data.shape)

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

data = clean_percentage_column(data, 'OTP')
data = clean_percentage_column(data, 'AM Peak')
data = clean_percentage_column(data, 'PM Peak')

# Create target variable 'High_OTP'
if 'OTP' in data.columns:
    otp_threshold = 0.90
    data['High_OTP'] = (data['OTP'] > otp_threshold).astype(int)
    print("\n'High_OTP' target variable created.")
else:
    print("\n'OTP' column not found, cannot create 'High_OTP'.")
    exit()

# Handle 'Month'
if 'Month' in data.columns:
    if data['Month'].dtype == 'object':
        month_mapping = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                         'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
        data['Month_Numeric'] = data['Month'].replace(month_mapping)
        data.drop('Month', axis=1, inplace=True, errors='ignore')
    elif data['Month'].dtype in ['int64', 'float64']:
        data['Month'] = data['Month'].astype('category')
else:
    print("'Month' column not found.")

# Encode 'Branch/Line'
data = pd.get_dummies(data, columns=['Branch/Line'], drop_first=True, errors='ignore')

# Drop rows with NaN in the target variable after preprocessing
data.dropna(subset=['High_OTP'], inplace=True)

# Separate features and target
X = data.drop('High_OTP', axis=1, errors='ignore')
y = data['High_OTP']

# Split data for evaluation (used in Phase III)
X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(X, y, test_size=0.3, random_state=42)

# Phase 2

# Initialize models
svm_model = SVC(random_state=42, probability=True)
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)

X = data.drop('High_OTP', axis=1)
y = data['High_OTP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nShape of training data:", X_train.shape, y_train.shape)
print("Shape of testing data:", X_test.shape, y_test.shape)