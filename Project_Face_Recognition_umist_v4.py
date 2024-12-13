# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:32:27 2024

"""

import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models

# Parameters for experimentation
pca_variance_ratio = 0.95  # The PCA will retain 95% of the variance. Adjusting this helps see how dimensionality affects performance.
random_state = 42

# =====================
# Load and prepare data
# =====================
data_path = 'P:/data/umist_cropped.mat'
data = sio.loadmat(data_path)

facedat = data['facedat'][0]
dirnames = data['dirnames'][0]

# Extract labels
labels = np.array([str(item[0]) for item in dirnames])
unique_labels = np.unique(labels)
label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}

# Prepare X (features) and y (labels) so that each image is a separate instance
X_list = []
y_list = []

for subj_idx, subject_images in enumerate(facedat):
    num_images = subject_images.shape[2]
    for img_i in range(num_images):
        img = subject_images[:, :, img_i]
        X_list.append(img.flatten())  # Flattening the 2D image into a 1D vector
        y_list.append(label_to_idx[labels[subj_idx]])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.int32)

# ======================
# Stratified Data Splits
# ======================
# Rational for split ratio (70% train, 15% val, 15% test):
# This ensures a large portion of data for training while keeping sufficient data for validation and testing.
# Stratified splits ensure each person (class) is represented equally in all subsets.

print("Class distribution before split:")
for lbl in unique_labels:
    print(lbl, np.sum(y == label_to_idx[lbl]))

sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=random_state)
train_val_idx, test_idx = next(sss_test.split(X, y))
X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
X_test, y_test = X[test_idx], y[test_idx]

# Further split train_val into train and val
val_ratio = 0.17647  # ~15% of total data
sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
train_idx, val_idx = next(sss_val.split(X_train_val, y_train_val))
X_train, y_train = X_train_val[train_idx], y_train_val[train_idx]
X_val, y_val = X_train_val[val_idx], y_train_val[val_idx]

print("\nClass distribution after split:")
print("Train:", [np.sum(y_train == i) for i in range(len(unique_labels))])
print("Val:", [np.sum(y_val == i) for i in range(len(unique_labels))])
print("Test:", [np.sum(y_test == i) for i in range(len(unique_labels))])

# ============
# Preprocessing
# ============
# Rationale:
# 1. Normalization/Scaling:
#    We use StandardScaler to normalize features. For each feature x:
#    x_norm = (x - μ) / σ
#    where μ is the mean and σ is the standard deviation of that feature.
#    This ensures all features contribute equally and helps the model train more effectively.

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Check for NaNs to ensure data integrity
assert not np.isnan(X_train_scaled).any(), "NaN values found in training set after scaling."
assert not np.isnan(X_val_scaled).any(), "NaN values found in validation set after scaling."
assert not np.isnan(X_test_scaled).any(), "NaN values found in test set after scaling."

# 2. Dimensionality Reduction (PCA):
#    PCA finds directions of maximum variance in data:
#    X_reduced = X * W
#    where W contains eigenvectors (principal components).
#    By retaining 95% variance, we reduce noise and complexity, improving training speed and potentially performance.

pca = PCA(pca_variance_ratio, svd_solver='full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Check for NaNs after PCA
assert not np.isnan(X_train_pca).any(), "NaN values in training set after PCA."
assert not np.isnan(X_val_pca).any(), "NaN values in validation set after PCA."
assert not np.isnan(X_test_pca).any(), "NaN values in test set after PCA."

num_classes = len(unique_labels)

# ===========
# Clustering
# ===========
# We apply K-Means clustering on the training data after PCA to explore the data structure.
# Rationale for K-Means:
# - We know the number of classes (num_classes), so we set k = num_classes.
# - K-Means is simple and well-suited for roughly spherical clusters.
# Parameter Tuning:
# - k = num_classes (since we know number of subjects).
# - Default parameters for simplicity; one could tune n_init, max_iter if needed.

kmeans = KMeans(n_clusters=num_classes, random_state=random_state)
kmeans.fit(X_train_pca)
cluster_labels = kmeans.labels_
# (Optionally, we could check cluster purity by comparing cluster_labels to y_train.)

# =========================
# Model Architecture (MLP)
# =========================
# We choose a Multi-Layer Perceptron (MLP) on the PCA features for simplicity and computational efficiency.
# Architecture:
# - Input layer: size = number of PCA components
# - Hidden layers: 
#   Layer 1: Dense(128) with ReLU activation to introduce non-linearity:
#     ReLU(x) = max(0, x), helps mitigate vanishing gradients and is standard in modern NN architectures.
#   Dropout(0.4): reduces overfitting by randomly dropping neurons during training.
#   Layer 2: Dense(64) with ReLU activation + Dropout(0.4)
# - Output layer: Dense(num_classes) with softmax for probability distribution over classes.
#
# Loss Function: Sparse categorical crossentropy, standard for multi-class classification.
# Optimizer: Adam with learning_rate=0.001, commonly used for stable convergence.
# Early Stopping: Monitors val_loss, stops training if no improvement for 10 epochs.

class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_train),
                                     y=y_train)
class_weights_dict = {i: w for i, w in enumerate(class_weights)}

model = models.Sequential([
    layers.Input(shape=(X_train_pca.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  patience=10,
                                                  restore_best_weights=True)

# Training the model
history = model.fit(X_train_pca, y_train,
                    validation_data=(X_val_pca, y_val),
                    epochs=50,
                    batch_size=16,
                    class_weight=class_weights_dict,  # Adjust for any class imbalance
                    callbacks=[early_stopping],
                    verbose=1)

# =========
# Evaluation
# =========
# We evaluate on the test set to measure generalization performance.
y_pred = model.predict(X_test_pca)
y_pred_labels = np.argmax(y_pred, axis=1)

acc = accuracy_score(y_test, y_pred_labels)
report = classification_report(y_test, y_pred_labels, target_names=unique_labels)

print("Test Accuracy:", acc)
print("Classification Report:\n", report)

# Error Analysis: Identify misclassifications for further inspection
misclassified_indices = np.where(y_test != y_pred_labels)[0]
if len(misclassified_indices) > 0:
    print("Misclassified samples indices:", misclassified_indices)
    # Could visualize or analyze these further.

# Overfitting Check
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
final_train_loss = history.history['loss'][-1]
final_val_loss = history.history['val_loss'][-1]

print("\nFinal Training Accuracy:", final_train_acc)
print("Final Validation Accuracy:", final_val_acc)
print("Final Training Loss:", final_train_loss)
print("Final Validation Loss:", final_val_loss)

# =====================
# Notes and Challenges
# =====================
# - We chose a PCA-based feature reduction to manage high-dimensional image data, reducing complexity.
# - K-Means clustering applied to training data helps understand data structure; further analysis could compare cluster memberships.
# - The MLP architecture is relatively simple yet effective. For more complex tasks or larger datasets, consider CNNs.
# - The chosen split ensures balanced representation. Adjusting PCA variance or model complexity may affect performance.
# - Challenges might include:
#   - Ensuring balanced splits across classes.
#   - Determining the right PCA variance ratio.
#   - Avoiding overfitting, addressed by dropout and early stopping.
#
# Future Steps:
# - Experiment with PCA variance to see how reducing dimensions affects accuracy.
# - Implement cross-validation for more robust performance estimates.
# - Error analysis to understand why certain misclassifications occur.
# - Potentially use CNNs or other models for improved performance.
#
# These notes, along with a formal analysis report including illustrations and deeper rationales, would complete the requirements.
