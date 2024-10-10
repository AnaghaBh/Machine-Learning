import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt

# Fetch dataset
real_estate_valuation = fetch_ucirepo(id=477)

# Load and reshape the dataset
X = np.asarray(real_estate_valuation.data.features)
y = np.asarray(real_estate_valuation.data.targets).flatten()  # Ensure y is a 1D array

# Manually split the data into training and test sets
train_size = 323
indices = np.random.permutation(len(X))
train_indices = indices[:train_size]
test_indices = indices[train_size:]

X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

# Implement linear regression using matrix inversion
def linear_regression(X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

# Train the model
theta = linear_regression(X_train, y_train)

# Predict on training and test data
y_train_pred = X_train.dot(theta)
y_test_pred = X_test.dot(theta)

# Calculate mean squared error
train_mse = np.mean((y_train - y_train_pred) ** 2)
test_mse = np.mean((y_test - y_test_pred) ** 2)

print("Training MSE:", train_mse)
print("Test MSE:", test_mse)

# Normalize the features manually
def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

X_train_scaled = normalize(X_train)
X_test_scaled = normalize(X_test)

# Updated gradient descent to match feature dimensions
def gradient_descent(X, y, learning_rate, max_iterations=10000, accuracy=0.01):
    m, n = X.shape
    theta = np.array([30] + [0] * (n - 1), dtype=float)  # Initialize theta with the correct dimension
    previous_theta = np.copy(theta)
    consecutive_iterations = 0
    
    for iteration in range(max_iterations):
        predictions = X.dot(theta)
        error = predictions - y
        gradient = (1/m) * X.T.dot(error)
        theta -= learning_rate * gradient
        
        # Check convergence
        if np.all(np.abs(theta - previous_theta) < accuracy):
            consecutive_iterations += 1
            if consecutive_iterations == 3:
                print(f"Converged after {iteration + 1} iterations.")
                print("accuracy:", np.abs(theta - previous_theta))
                break
        else:
            consecutive_iterations = 0
        
        previous_theta = np.copy(theta)
    
    return theta

# Test different learning rates
learning_rates = [0.001, 0.003, 0.01, 0.1]

for lr in learning_rates:
    print(f"\nTesting learning rate: {lr}")
    theta_gd = gradient_descent(X_train_scaled, y_train, learning_rate=lr)
    
    # Predict on training and test data
    y_train_pred_gd = X_train_scaled.dot(theta_gd)
    y_test_pred_gd = X_test_scaled.dot(theta_gd)
    
    # Calculate mean squared error
    train_mse_gd = np.mean((y_train - y_train_pred_gd) ** 2)
    test_mse_gd = np.mean((y_test - y_test_pred_gd) ** 2)
    
    print(f"Training MSE :", train_mse_gd)
    print(f"Test MSE :", test_mse_gd)
