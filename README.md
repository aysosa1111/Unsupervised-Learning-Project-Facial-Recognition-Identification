# UMIST Face Recognition Project

## Overview
This project implements a face recognition system using machine learning techniques on the UMIST face dataset. The implementation demonstrates a comprehensive approach to image classification, featuring preprocessing, dimensionality reduction, and neural network-based classification.

## Project Structure
- Data preprocessing
- Dimensionality reduction with PCA
- K-Means clustering
- Multi-Layer Perceptron (MLP) classification

## Dependencies
- NumPy
- SciPy
- scikit-learn
- TensorFlow
- Keras

## Key Features
- Stratified data splitting
- Standard scaling
- Principal Component Analysis (PCA)
- Balanced class weight handling
- Early stopping to prevent overfitting

## Dataset
- Source: UMIST Face Database
- Preprocessing: Image flattening, scaling, and dimensionality reduction
- Split: 70% training, 15% validation, 15% testing

## Model Architecture
### Preprocessing
- StandardScaler for feature normalization
- PCA for dimensionality reduction (95% variance retained)

### Neural Network
- Input Layer: PCA-transformed features
- Hidden Layers:
  - First layer: 128 neurons with ReLU activation
  - Dropout (40%) for regularization
  - Second layer: 64 neurons with ReLU activation
  - Dropout (40%)
- Output Layer: Softmax classification

## Training Parameters
- Optimizer: Adam (learning rate: 0.001)
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 16
- Early Stopping: 10 epochs patience

## Performance Metrics
- Accuracy measured on test set
- Detailed classification report
- Misclassification analysis

## Experimental Considerations
- Balanced class representation
- Handling high-dimensional image data
- Preventing overfitting

## Future Improvements
- Experiment with CNN architectures
- Implement cross-validation
- Deeper error analysis
- Explore alternative dimensionality reduction techniques

## How to Run
1. Ensure all dependencies are installed
2. Place the UMIST dataset at the specified path
3. Run the Python script

## Acknowledgments
- UMIST Face Database
- Scikit-learn
- TensorFlow Team
