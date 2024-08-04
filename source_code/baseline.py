import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Measure training time
start_time = time.time()

# Train the Logistic Regression model
model_lr = LogisticRegression(C=75, max_iter=175, solver='liblinear')
model_lr.fit(X_train, y_train)

end_time = time.time()
training_time_lr = end_time - start_time

# Make predictions
y_pred = model_lr.predict(X_test)

# Evaluate the model
accuracy_lr = accuracy_score(y_test, y_pred)
precision_lr = precision_score(y_test, y_pred)
recall_lr = recall_score(y_test, y_pred)
f1_lr = f1_score(y_test, y_pred)
roc_auc_lr = roc_auc_score(y_test, y_pred)

# Print the evaluation metrics and training time
print(f'Logistic Regression Accuracy: {accuracy_lr}')
print(f'Logistic Regression Precision: {precision_lr}')
print(f'Logistic Regression Recall: {recall_lr}')
print(f'Logistic Regression F1-Score: {f1_lr}')
print(f'Logistic Regression ROC-AUC: {roc_auc_lr}')
print(f'Logistic Regression Training Time: {training_time_lr} seconds')

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar

# Create a baseline model using DummyClassifier
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, y_train)
y_pred_dummy = dummy_clf.predict(X_test)

# Evaluate the baseline model
accuracy_dummy = accuracy_score(y_test, y_pred_dummy)
precision_dummy = precision_score(y_test, y_pred_dummy)
recall_dummy = recall_score(y_test, y_pred_dummy)
f1_dummy = f1_score(y_test, y_pred_dummy)
roc_auc_dummy = roc_auc_score(y_test, y_pred_dummy)

print(f'Baseline Accuracy: {accuracy_dummy}')
print(f'Baseline Precision: {precision_dummy}')
print(f'Baseline Recall: {recall_dummy}')
print(f'Baseline F1-Score: {f1_dummy}')
print(f'Baseline ROC-AUC: {roc_auc_dummy}')

# Convert y_test and y_pred_lstm to Numpy arrays
y_test_np = np.array(y_test)
y_pred_lstm_np = np.array(y_pred_lstm.flatten())

# Convert to Pandas Series for crosstab
y_test_series = pd.Series(y_test_np, name='Actual')
y_pred_lstm_series = pd.Series(y_pred_lstm_np, name='Predicted')

# Check unique values
print("Unique values in y_test:", y_test_series.unique())
print("Unique values in y_pred_lstm:", y_pred_lstm_series.unique())

# Create a detailed contingency table
contingency_table = pd.crosstab(y_test_series, y_pred_lstm_series, rownames=['Actual'], colnames=['Predicted'])
print("Detailed Contingency Table:")
print(contingency_table)

# Perform McNemar's test if the table is not empty
if not contingency_table.empty:
    result = mcnemar(contingency_table, exact=True)
    print(f'McNemar Test Statistic: {result.statistic}')
    print(f'McNemar Test p-value: {result.pvalue}')
else:
    print("Contingency table is empty. Cannot perform McNemar's test.")
