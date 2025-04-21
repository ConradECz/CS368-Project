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
print("Aequitas is working!")
from aequitas.plotting import Plot
print("Aequitas is working!")
from aequitas.bias import Bias
print("Aequitas is working!")
from aequitas.fairness import Fairness
print("Aequitas is working!")

print("\n--- Phase I - Preprocessing Phase ---")

# Load the MTA Metro North dataset
try:
    data = pd.read_csv("MTA_Metro-North_On-Time_Performance__Beginning_2020.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: CSV file not found. Please make sure the file path is correct.")
    exit()

# Print initial info
print("\nInitial Dataset Info:")
data.info()
print("\nInitial Summary Statistics:")
print(data.describe())

# Identify attributes with insufficient variability
insufficient_variance_cols = []
for col in data.columns:
    if data[col].nunique() <= 1:
        insufficient_variance_cols.append(col)

print(f"\nAttributes with insufficient variability: {insufficient_variance_cols}")

# Drop low variance columns
data = data.drop(columns=insufficient_variance_cols, errors='ignore')
print("\nDataset shape after dropping low variance columns:", data.shape)

# Clean percentage columns
def clean_percentage_column(df, column_name):
    if column_name in df.columns and df[column_name].dtype == 'object':
        try:
            df[column_name] = df[column_name].str.rstrip('%').astype('float') / 100.0
            print(f"'{column_name}' cleaned and converted to numeric.")
        except Exception as e:
            print(f"Error cleaning '{column_name}': {e}")
    elif column_name in df.columns:
        print(f"'{column_name}' is already numeric.")
    else:
        print(f"'{column_name}' not found.")
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

# Split data for evaluation
X_train_eval, X_test_eval, y_train_eval, y_test_eval = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Baseline Model ---
print("\n--- Baseline Model (Majority Class) ---")
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train_eval, y_train_eval)
baseline_pred = dummy_clf.predict(X_test_eval)
baseline_accuracy = accuracy_score(y_test_eval, baseline_pred)
baseline_auc = roc_auc_score(y_test_eval, dummy_clf.predict_proba(X_test_eval)[:, 1])
print(f"Baseline Accuracy: {baseline_accuracy:.4f}")
print(f"Baseline AUC: {baseline_auc:.4f}")

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

print("\n--- Phase IV - Bias and Fairness ---")

# Assuming 'Branch/Line' is a potential protected attribute
branch_cols = [col for col in X_test_eval.columns if col.startswith('Branch/Line_')]
if branch_cols:
    best_model_name = 'Random Forest'  # Based on typical performance, adjust if needed
    best_model = trained_models[best_model_name]
    y_pred_proba_best = best_model.predict_proba(X_test_eval)[:, 1]
    y_pred_best = best_model.predict(X_test_eval)

    # Create a DataFrame for Aequitas
    aequitas_df = pd.DataFrame({
        'score': y_pred_proba_best,
        'label_value': y_test_eval.values,
        'predicted_value': y_pred_best,
        'Branch': X_test_eval[branch_cols].idxmax(axis=1).str.replace('Branch/Line_', '')
    })


    # Group object
    aqg = Group()
    group_df, _ = aqg.get_group_metric(aequitas_df, 'Branch', 'label_value', 'predicted_value')

    # Bias object
    aqb = Bias()
    bdf = aqb.get_bias_result(group_df, 'Branch', 'label_value', 'predicted_value')

    # Fairness object
    aqf = Fairness()
    fdf = aqf.get_fairness_result(bdf)

    print(f"\nAequitas Bias and Fairness Analysis (assuming 'Branch' is a protected attribute, using {best_model_name}):")
    print(fdf)

    # Plotting (optional)
    aqp = Plot()
    p = aqp.plot_group_metric_all(group_df, metrics=['tpr', 'fpr', 'precision'], ncols=3)
    plt.show()
    p = aqp.plot_bias_all(bdf, fairness_threshold=0.8, ncols=3)
    plt.show()

else:
    print("\n'Branch/Line' columns not found for bias and fairness analysis.")

print("\n--- Phase V - Feature Ranking/Selection ---")

best_model_name = 'Random Forest' # Adjust if a different model performed best
best_model = trained_models[best_model_name]

if isinstance(best_model, RandomForestClassifier):
    feature_importances = best_model.feature_importances_
    feature_names = X_train_eval.columns
    sorted_importances = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)

    print(f"\nFeature Importances (from {best_model_name}):")
    for feature, importance in sorted_importances:
        print(f"{feature}: {importance:.4f}")

    # Example: Select top N features (e.g., top 5)
    top_n = 5
    selected_features = [feature for feature, importance in sorted_importances[:top_n]]
    print(f"\nTop {top_n} most important features: {selected_features}")

elif isinstance(best_model, Pipeline) and isinstance(best_model.named_steps['svm'], SVC) and best_model.named_steps['svm'].kernel == 'linear':
    feature_importances = best_model.named_steps['svm'].coef_[0]
    feature_names = X_train_eval.columns
    sorted_importances = sorted(zip(feature_names, abs(feature_importances)), key=lambda x: x[1], reverse=True)

    print(f"\nFeature Importances (from Linear SVM):")
    for feature, importance in sorted_importances:
        print(f"{feature}: {importance:.4f}")

else:
    print(f"\nFeature ranking not directly applicable to the best performing model type ({type(best_model).__name__}) with the current settings.")