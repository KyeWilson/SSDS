import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import functions from functions.py
from py.functions import (
    load_and_preprocess_data, split_data, standardize_features,
    one_hot_encode_labels, build_neural_network, compile_and_train_model,
    evaluate_model
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

def test_activation_function(activation_function):
    """Train and evaluate a neural network with a given activation function."""
    print(f"\nTesting Activation Function: {activation_function}")
    
    # Build model
    model = build_neural_network(activation_function, input_shape=X_train.shape[1], num_classes=3)
    
    # Compile and train the model
    compile_and_train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate the model
    test_accuracy, test_loss = evaluate_model(model, X_test, y_test)
    
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    return test_accuracy, test_loss

# Test multiple activation functions
results = {}
for activation in ['relu', 'sigmoid', 'tanh']:
    accuracy, loss = test_activation_function(activation)
    results[activation] = {'Accuracy': accuracy, 'Loss': loss}

# Convert results to a DataFrame for easy visualization
results_df = pd.DataFrame(results).T
print(results_df)

# Plot results
plt.figure(figsize=(8, 5))
sns.barplot(x=results_df.index, y=results_df['Accuracy'], palette="viridis")
plt.title("Neural Network Accuracy by Activation Function")
plt.ylabel("Accuracy")
plt.xlabel("Activation Function")
plt.ylim(0, 1)
plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x=results_df.index, y=results_df['Loss'], palette="magma")
plt.title("Neural Network Loss by Activation Function")
plt.ylabel("Loss")
plt.xlabel("Activation Function")
plt.ylim(0, max(results_df['Loss']) * 1.2)
plt.show()
