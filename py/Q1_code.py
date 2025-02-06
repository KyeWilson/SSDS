from py.functions import (
    load_data, preprocess_data, split_data, train_decision_tree,
    evaluate_model, plot_confusion_matrix, grid_search_decision_tree
)

# File path
file_path = "Skyserver_SQL2_27_2018 6_51_39 PM.csv"

# Load data
data = load_data(file_path)

# Preprocess data
columns_to_drop = ['objid', 'specobjid', 'fiberid', 'plate', 'mjd']
data = preprocess_data(data, columns_to_drop, target_column='class')

# Split data
feature_columns = ['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift_normalized']
X_train, X_test, y_train, y_test = split_data(data, feature_columns, 'class_encoded')

# Train Decision Tree
model = train_decision_tree(X_train, y_train)

# Evaluate model
y_pred, accuracy, report, conf_matrix = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
plot_confusion_matrix(conf_matrix)

# Grid Search for Hyperparameter Tuning
param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
best_params, best_model = grid_search_decision_tree(X_train, y_train, param_grid)
print(f"Best Parameters: {best_params}")

# Evaluate refined model
y_pred_refined, accuracy_refined, report_refined, conf_matrix_refined = evaluate_model(best_model, X_test, y_test)
print(f"Refined Model Accuracy: {accuracy_refined:.2f}")
print("Classification Report (Refined Model):")
print(report_refined)
plot_confusion_matrix(conf_matrix_refined, title="Confusion Matrix (Refined Model)")
