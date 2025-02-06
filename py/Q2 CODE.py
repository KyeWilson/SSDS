import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import functions from functions.py
from py.functions import (
    load_and_preprocess_data, split_data, standardize_features,
    one_hot_encode_labels, build_neural_network, compile_and_train_model,
    evaluate_model, plot_confusion_matrix
)

# Define file path
file_path = "Skyserver_SQL2_27_2018 6_51_39 PM.csv"

# Load and preprocess data
data = load_and_preprocess_data(file_path)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = split_data(data)

# Standardize the feature values
X_train, X_test = standardize_features(X_train, X_test)

# One-hot encode target labels
y_train, y_test = one_hot_encode_labels(y_train, y_test)

# Display dataset overview
data.head()

# Plot the distribution of classes
sns.countplot(data['class'])
plt.title("Distribution of Object Classes")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Define and build the Neural Network
model = build_neural_network(activation_function='relu', input_shape=X_train.shape[1], num_classes=3)

# Compile and train the model
history = compile_and_train_model(model, X_train, y_train, X_test, y_test)

# Display training summary
print("Neural Network training completed.")

# Make predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
y_test_classes = np.argmax(y_test, axis=1)  # Convert one-hot encoded labels to class labels

# Print classification report
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes))

# Plot confusion matrix
plot_confusion_matrix(y_test_classes, y_pred_classes, class_names=['Galaxy', 'Star', 'Quasar'])
