Phase I:
-

-Loads the MTA Metor-North On-Time Performance dataset\
-Rename columns, drops the ones with low variance\
-Convert data strings and percentage strings into numeric formats\
-Handle missing values\
-Create a binary target variable\
-Encode categorial variables into dummy variables\
-Splits data into features and target to create training sets to evaluate

Phase II:
-

-Define a 10-fold cross-validation\
-Set up the SVM, Decision Tree, and Random Tree classifiers\
-Evaluate each model using cross-validation for accuray and ROC AUC

Phase III:
-

-Training all models using the full training set\
-Evaluate models using accuracy, ROC AUC, and classification report\
(precision, recall, F1-score)\
-Plots ROC curves to visually be able to see the comparisions of the models

Phase IV:
-

-Select best model for fairness analysis\
-Use Aequitas to evaluate bias\
-Compute group-level metrics, bias metrics, and fairness metrics

Phase V:
-

For the best model:\
-Compute feature importances\
-Prints all feature importances and select top 5 most influential features


