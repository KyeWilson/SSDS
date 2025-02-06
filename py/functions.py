import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Encode the 'class' column into numerical values
    data['class_encoded'] = data['class'].astype('category').cat.codes
    
    # Normalize the 'redshift' column
    data['redshift_normalized'] = (data['redshift'] - data['redshift'].mean()) / data['redshift'].std()
    
    return data


# Split data into training and testing sets
def split_data(data):
    X = data[['ra', 'dec', 'u', 'g', 'r', 'i', 'z', 'redshift_normalized']]  # Features
    y = data['class_encoded']  # Target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Standardize features
def standardize_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# One-hot encode target labels
def one_hot_encode_labels(y_train, y_test, num_classes=3):
    y_train_encoded = to_categorical(y_train, num_classes=num_classes)
    y_test_encoded = to_categorical(y_test, num_classes=num_classes)
    return y_train_encoded, y_test_encoded


# Build a Decision Tree model
def build_decision_tree_model(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


# Build a Neural Network model
def build_neural_network(activation_function, input_shape, num_classes):
    model = Sequential([
        Dense(64, activation=activation_function, input_shape=(input_shape,)),
        Dense(32, activation=activation_function),
        Dense(num_classes, activation='softmax')
    ])
    return model


# Compile and train a Neural Network
def compile_and_train_model(model, X_train, y_train, X_test, y_test, learning_rate=0.001, epochs=20, batch_size=32):
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1)
    return history


# Evaluate a model
def evaluate_model(model, X_test, y_test):
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    return test_accuracy, test_loss


# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


# Grid Search for Decision Tree
def decision_tree_grid_search(X_train, y_train, param_grid):
    grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_estimator_

