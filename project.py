import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from aequitas.group import Group
from aequitas.plotting import Plot
from aequitas.bias import Bias
from aequitas.fairness import Fairness

print("Aequitas is working!")

print("\n--- Phase I - Preprocessing Phase ---")

# Load the MTA Metro North dataset with the corrected column name
try:
    data = pd.read_csv("MTA_Metro-North_On-Time_Performance__Beginning_2020.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: CSV file not found. Please make sure the file path is correct.")
    exit()

# Rename column to match dataset structure
data.rename(columns={'Branch / Line': 'Branch_Line'}, inplace=True)

# Print initial dataset info
print("\nInitial Dataset Info:")
data.info()
print("\nInitial Summary Statistics:")
print(data.describe())

# Identify attributes with insufficient variability
insufficient_variance_cols = [col for col in data.columns if data[col].nunique() <= 1]
print(f"\nAttributes with insufficient variability: {insufficient_variance_cols}")

# Drop low-variance columns
data.drop(columns=insufficient_variance_cols, inplace=True)
print("\nDataset shape after dropping low-variance columns:", data.shape)

# Convert Month column from date string to numeric month
data['Month'] = pd.to_datetime(data['Month']).dt.month

# Clean percentage columns
def clean_percentage_column(df, column_name):
    if column_name in df.columns and df[column_name].dtype == 'object':
        try:
            df[column_name] = df[column_name].str.rstrip('%').astype('float') / 100.0
            print(f"'{column_name}' cleaned and converted to numeric.")
        except Exception as e:
            print(f"Error cleaning '{column_name}': {e}")
    return df

percentage_columns = ['AM Peak', 'PM Peak', 'Off Peak']
for col in percentage_columns:
    data = clean_percentage_column(data, col)

# Handle missing values in 'AM Peak', 'PM Peak', and 'Off Peak'
data.dropna(subset=['AM Peak', 'PM Peak', 'Off Peak'], inplace=True)

# Create target variable 'High_OTP'
otp_threshold = 0.90
data['High_OTP'] = (data['OTP'] > otp_threshold).astype(int)
print("\n'High_OTP' target variable created.")

# Encode categorical column 'Branch_Line'
data = pd.get_dummies(data, columns=['Branch_Line'], drop_first=True)

# Separate features and target
X = data.drop('High_OTP', axis=1)
y = data['High_OTP']

# Split data for evaluation
X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n--- Phase II - Classification Phase (Model Training and Cross-Validation) ---")

# Define cross-validation strategy
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Models with potential pipelines for scaling
svm_pipeline = Pipeline([('scaler', StandardScaler()), ('svm', SVC(random_state=42, probability=True))])
dt_model = DecisionTreeClassifier(random_state=42)
rf_model = RandomForestClassifier(random_state=42)

models = {'SVM': svm_pipeline, 'Decision Tree': dt_model, 'Random Forest': rf_model}
cv_results = {}

for name, model in models.items():
    print(f"\nEvaluating {name} using 10-fold cross-validation:")
    accuracy_scores = cross_val_score(model, X_train_eval, y_train_eval, cv=cv, scoring='accuracy')
    auc_scores = cross_val_score(model, X_train_eval, y_train_eval, cv=cv, scoring='roc_auc')
    cv_results[name] = {'accuracy_mean': accuracy_scores.mean(), 'accuracy_std': accuracy_scores.std(),
                        'auc_mean': auc_scores.mean(), 'auc_std': auc_scores.std()}
    print(f"Mean Accuracy: {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std():.4f})")
    print(f"Mean AUC: {auc_scores.mean():.4f} (+/- {auc_scores.std():.4f})")

print("\n--- Phase III - Evaluation Phase (Model Evaluation on Test Set) ---")

trained_models = {}
for name, model in models.items():
    print(f"\nTraining {name} on the full training set...")
    model.fit(X_train_eval, y_train_eval)
    trained_models[name] = model

print("\nEvaluating models on the Test Set:")
test_results = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_test_eval)
    y_pred_proba = model.predict_proba(X_test_eval)[:, 1]
    accuracy = accuracy_score(y_test_eval, y_pred)
    auc = roc_auc_score(y_test_eval, y_pred_proba)
    report = classification_report(y_test_eval, y_pred)
    test_results[name] = {'accuracy': accuracy, 'auc': auc, 'report': report, 'y_pred_proba': y_pred_proba}
    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print("Classification Report:\n", report)

# Plotting ROC Curves
plt.figure(figsize=(8, 6))
for name, result in test_results.items():
    fpr, tpr, _ = roc_curve(y_test_eval, result['y_pred_proba'])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {result["auc"]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curves')
plt.legend()
plt.show()

print("\n--- Phase IV - Feature Ranking/Selection ---")

best_model_name = 'Random Forest'
best_model = trained_models[best_model_name]

if isinstance(best_model, RandomForestClassifier):
    feature_importances = best_model.feature_importances_
    feature_names = X_train_eval.columns
    sorted_importances = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)

    print(f"\nFeature Importances (from {best_model_name}):")
    for feature, importance in sorted_importances:
        print(f"{feature}: {importance:.4f}")

    # Selecting top 5 features
    top_n = 5
    selected_features = [feature for feature, importance in sorted_importances[:top_n]]
    print(f"\nTop {top_n} most important features: {selected_features}")

print("\nFinalized code with preprocessing fixes and enhancements included!")
