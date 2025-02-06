import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score

# Import functions from functions.py
from py.functions import (
    load_and_preprocess_data, split_data, build_decision_tree_model,
    evaluate_model, plot_confusion_matrix, decision_tree_grid_search
)

# Define file path
file_path = "Skyserver_SQL2_27_2018 6_51_39 PM.csv"

# Load and preprocess data
data = load_and_preprocess_data(file_path)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = split_data(data)

# Display the first few rows of the processed data
data.head()

# Train a Decision Tree classifier
model = build_decision_tree_model(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy, report, conf_matrix = evaluate_model(model, X_test, y_test)

# Print results
print(f"Decision Tree Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

# Plot confusion matrix
plot_confusion_matrix(y_test, y_pred, class_names=['Galaxy', 'Star', 'Quasar'])

# Define the parameter grid for optimization
param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search
best_params, best_model = decision_tree_grid_search(X_train, y_train, param_grid)

print(f"Best Parameters Found: {best_params}")

# Make predictions using the best model
y_pred_refined = best_model.predict(X_test)

# Evaluate refined model
accuracy_refined, report_refined, conf_matrix_refined = evaluate_model(best_model, X_test, y_test)

print(f"Refined Model Accuracy: {accuracy_refined:.2f}")
print("Classification Report (Refined Model):\n", report_refined)

# Plot confusion matrix for the refined model
plot_confusion_matrix(y_test, y_pred_refined, class_names=['Galaxy', 'Star', 'Quasar'])
