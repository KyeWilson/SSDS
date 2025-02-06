# Galaxy Zoo Neural Network Analysis

This project explores the Galaxy Zoo dataset using machine learning and deep learning techniques. The primary objective is to investigate how different models and configurations classify celestial objects (Galaxy, Star, or Quasar).

The analysis addresses three key questions:
1. **Q1**: How well can a Decision Tree classify celestial objects?
2. **Q2**: How does a Neural Network compare to a Decision Tree for this task?
3. **Q3**: How does the choice of activation function affect Neural Network performance?

The project is designed to guide users through:

1. **Data Preprocessing**: Cleaning, normalising, and encoding features.
2. **Model Design**: Building and training machine learning and neural network models.
3. **Performance Evaluation**: Assessing accuracy, precision, recall, F1-score, and generalisation.

## Dataset
### Source
The dataset originates from the Galaxy Zoo Challenge, a citizen science project aimed at classifying galaxies based on their visual properties. It provides valuable insights into celestial objects, aiding research in astronomy and machine learning.

### Features
- **Input Features**: RA, DEC, magnitudes (u, g, r, i, z), and redshift.
- **Target Labels**: Object classification (Galaxy, Star, or Quasar).

## Why This Dataset?
The Galaxy Zoo dataset is ideal for machine learning tasks due to:

1. **Complexity**: The dataset includes highly varied and non-linear relationships, making it a good testbed for advanced classification models.

2. **Real-World Application**: Understanding celestial objects aids in astronomical research and supports projects like identifying high-redshift galaxies.

3. **Large Sample Size**: Ensures meaningful results and reliable model evaluation.

## Project Structure
### Q1: Decision Tree Classifier
#### Objective: Classify celestial objects using a Decision Tree Classifier.

**Highlights**:
  - Preprocessed dataset to handle missing values and feature scaling.
  - Achieved 99% accuracy, demonstrating its simplicity and effectiveness.
  - High precision, recall, and F1-scores across all classes.

### Q2: Neural Network Classifier
#### Objective: Build a Neural Network and compare its performance to the Decision Tree.

**Highlights**:
  - Used two hidden layers with ReLU activation and softmax output.
  - Achieved 98% accuracy, with notable improvements in Quasar classification.

### Q3: Effect of Activation Functions
#### Objective: Examine the impact of different activation functions on Neural Network performance.

**Highlights**:
  - Investigated **ReLU**, **Sigmoid**, and **Tanh** activations.
  - **ReLU** and **Tanh** both achieved 99% accuracy, with **Tanh** reducing loss more effectively.
  - **Sigmoid** underperformed with 97% accuracy and slower convergence.

## How to Run the Notebooks

1. Clone the repository:
   git clone <repository-link>
   cd <repository-folder>

2. **Install the dependencies**: Use the provided `dependencies.txt` file to ensure all required libraries are installed:
  pip install -r dependencies.txt

3. **Run the notebooks**: Launch Jupyter Notebook:
  jupyter notebook

Navigate to the respective Q1, Q2, and Q3 folders to explore the notebooks.

## Dependencies
All Python libraries required to run the notebooks are listed in `dependencies.txt`:
`Python 3.8+`
`TensorFlow 2.9.0`
`Scikit-learn 1.1.1`
`Matplotlib 3.5.2`
`Pandas 1.4.3`
`NumPy 1.22.4`
`Seaborn 0.11.2`
To ensure compatibility, install the specified versions of the libraries.

## Motivation
Understanding the classification of celestial objects is crucial in astrophysics for:

1. Identifying high-redshift galaxies and their properties.
2. Distinguishing stars and quasars from galaxies for observational studies.
3. Applying machine learning techniques to large astronomical datasets, paving the way for further advancements in automated classification systems.

This project aims to provide a comprehensive learning experience for intermediate users by showcasing how machine learning and deep learning techniques can address complex classification problems.

## Acknowledgments
- **Galaxy Zoo Challenge**: For providing the dataset.
- **Libraries Used**: TensorFlow, Scikit-learn, Matplotlib, Pandas, NumPy, and Seaborn.
- **Inspiration**: This project was completed as part of a coursework assignment to deepen knowledge of machine learning in astronomy.
