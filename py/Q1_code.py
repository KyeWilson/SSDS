#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[3]:


# Load the dataset
file_path = "Skyserver_SQL2_27_2018 6_51_39 PM.csv"  # Replace with the correct path if needed
data = pd.read_csv(file_path)

# Display basic information about the dataset
print(data.info())
print(data.head())


# In[4]:


# Check for missing values
print("Missing values per column:")
print(data.isnull().sum())


# In[5]:


import seaborn as sns
import matplotlib.pyplot as plt

# Distribution of the 'class' column
sns.countplot(data['class'])
plt.title('Distribution of Object Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Distribution of the 'redshift' column
sns.histplot(data['redshift'], bins=30, kde=True)
plt.title('Distribution of Redshift Values')
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.show()


# In[6]:


# Drop unnecessary columns
columns_to_drop = ['objid', 'specobjid', 'fiberid', 'plate', 'mjd']  # Adjust as needed
data = data.drop(columns=columns_to_drop, axis=1)

# Encode 'class' column
data['class_encoded'] = data['class'].astype('category').cat.codes

# Normalize redshift
data['redshift_normalized'] = (data['redshift'] - data['redshift'].mean()) / data['redshift'].std()

# Preview the cleaned dataset
print(data.head())


# In[7]:


# Distribution of the 'redshift' column
sns.histplot(data['redshift'], bins=30, kde=True)
plt.title('Distribution of Redshift Values')
plt.xlabel('Redshift')
plt.ylabel('Frequency')
plt.show()


# In[9]:


from sklearn.model_selection import train_test_split

# Select features and target
X = data[['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift_normalized']]  # Feature columns
y = data['class_encoded']  # Target column

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the datasets
print(f"Training Features Shape: {X_train.shape}")
print(f"Testing Features Shape: {X_test.shape}")
print(f"Training Target Shape: {y_train.shape}")
print(f"Testing Target Shape: {y_test.shape}")


# In[10]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[11]:


from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Perform Grid Search with cross-validation
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print(f"Best Parameters: {best_params}")


# In[12]:


# Make predictions with the refined model
y_pred_refined = best_model.predict(X_test)

# Evaluate the refined model
accuracy_refined = accuracy_score(y_test, y_pred_refined)
print(f"Refined Model Accuracy: {accuracy_refined:.2f}")

# Classification report
print("Classification Report (Refined Model):")
print(classification_report(y_test, y_pred_refined))

# Confusion matrix
conf_matrix_refined = confusion_matrix(y_test, y_pred_refined)
sns.heatmap(conf_matrix_refined, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Refined Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

