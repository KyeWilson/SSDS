#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Neural network tools
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# In[2]:


# Load the cleaned dataset
file_path = "Skyserver_SQL2_27_2018 6_51_39 PM.csv"  # Adjust path if needed
data = pd.read_csv(file_path)

# Encode the 'class' column into numerical values
data['class_encoded'] = data['class'].astype('category').cat.codes

# Normalize the 'redshift' column
data['redshift_normalized'] = (data['redshift'] - data['redshift'].mean()) / data['redshift'].std()

# Select features and target
X = data[['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift_normalized']]  # Feature columns
y = data['class_encoded']  # Target column

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode the target labels
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)

# Display shapes to confirm
print(f"Training Features Shape: {X_train.shape}")
print(f"Testing Features Shape: {X_test.shape}")
print(f"Training Target Shape: {y_train.shape}")
print(f"Testing Target Shape: {y_test.shape}")


# In[3]:


# Define a function to build, train, and evaluate a model with a specific activation function
def build_and_evaluate_model(activation_function):
    print(f"\nTesting Activation Function: {activation_function}")
    
    # Define the model
    model = Sequential([
        Dense(64, activation=activation_function, input_shape=(X_train.shape[1],)),  # Input layer
        Dense(32, activation=activation_function),  # Hidden layer
        Dense(3, activation='softmax')  # Output layer
    ])
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                        epochs=20, batch_size=32, verbose=0)
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Return results for comparison
    return test_accuracy, test_loss


# In[4]:


# Test different activation functions
results = {}
for activation in ['relu', 'sigmoid', 'tanh']:
    accuracy, loss = build_and_evaluate_model(activation)
    results[activation] = {'Accuracy': accuracy, 'Loss': loss}

# Display results
results_df = pd.DataFrame(results).T
print(results_df)

