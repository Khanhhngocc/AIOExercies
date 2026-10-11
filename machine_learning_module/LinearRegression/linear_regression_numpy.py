# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 2. LOAD AND PREPARE THE ADVERTISING DATASET
# =========================================================
# Load the dataset while skipping the header row.
data = np.genfromtxt(
    "./advertising.csv",
    delimiter=",",
    skip_header=1,
)

# Separate input features and target values.
#
# X: TV, Radio, Newspaper advertising budgets
# y: Sales
X = data[:, :3]
y = data[:, 3:]


# =========================================================
# 3. MEAN NORMALIZATION
# =========================================================
def mean_normalization(X):
    """
    Apply mean normalization separately to each feature.

    Formula:
    x_scaled = (x - mean) / (max - min)

    A bias column of ones is added as the first column.
    """
    n_samples = len(X)

    # Calculate statistics for each feature column.
    max_values = np.max(X, axis=0)
    min_values = np.min(X, axis=0)
    mean_values = np.mean(X, axis=0)

    # Normalize each feature independently.
    X_scaled = (
        (X - mean_values)
        / (max_values - min_values)
    )

    # Add a bias column so theta[0] represents the intercept.
    X_with_bias = np.c_[
        np.ones((n_samples, 1)),
        X_scaled,
    ]

    return X_with_bias, max_values, min_values, mean_values


X_b, max_values, min_values, mean_values = mean_normalization(X)


# =========================================================
# 4. STOCHASTIC GRADIENT DESCENT
# =========================================================
def stochastic_gradient_descent(
    X_b,
    y,
    n_epochs=50,
    learning_rate=0.01,
):
    """
    Train Linear Regression using Stochastic Gradient Descent.

    SGD updates model parameters after processing one sample.
    """
    n_samples, n_features_with_bias = X_b.shape

    # Initialize weights:
    # theta[0] is the bias, and the remaining values are feature weights.
    thetas = np.asarray([
        [1.16270837],
        [-0.81960489],
        [1.39501033],
        [0.29763545],
    ])

    # Store parameter values and loss values during training.
    thetas_path = [thetas.copy()]
    losses = []

    for epoch in range(n_epochs):
        # Shuffle sample indices at the beginning of every epoch.
        shuffled_indices = np.random.permutation(n_samples)

        for random_index in shuffled_indices:
            # Select one training sample.
            xi = X_b[random_index:random_index + 1]
            yi = y[random_index:random_index + 1]

            # Calculate the predicted value.
            y_hat = xi.dot(thetas)

            # Calculate Squared Error loss for this sample.
            loss = (yi - y_hat) ** 2

            # Calculate the gradient for one sample.
            gradient = 2 * (y_hat - yi) * xi.T

            # Update model parameters.
            thetas -= learning_rate * gradient

            # Save the current parameters and loss.
            thetas_path.append(thetas.copy())
            losses.append(loss[0, 0])

    return thetas_path, losses


# Train the model using SGD.
sgd_thetas, sgd_losses = stochastic_gradient_descent(
    X_b,
    y,
    n_epochs=50,
    learning_rate=0.01,
)

# Plot the first 500 parameter updates.
plt.figure(figsize=(10, 6))

plt.plot(
    range(min(500, len(sgd_losses))),
    sgd_losses[:500],
    color="red",
)

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("SGD Loss per Step (First 500 Updates)")
plt.grid(alpha=0.3)
plt.show()


# =========================================================
# 5. MINI-BATCH GRADIENT DESCENT
# =========================================================
def mini_batch_gradient_descent(
    X_b,
    y,
    n_epochs=50,
    minibatch_size=20,
    learning_rate=0.01,
):
    """
    Train Linear Regression using Mini-Batch Gradient Descent.

    The model updates parameters after each small batch of samples.
    """
    n_samples, n_features_with_bias = X_b.shape

    # Initialize model parameters.
    thetas = np.asarray([
        [1.16270837],
        [-0.81960489],
        [1.39501033],
        [0.29763545],
    ])

    thetas_path = [thetas.copy()]
    losses = []

    for epoch in range(n_epochs):
        # Shuffle data before creating mini-batches.
        shuffled_indices = np.random.permutation(n_samples)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]

        # Process one mini-batch at a time.
        for start_index in range(0, n_samples, minibatch_size):
            end_index = start_index + minibatch_size

            xi = X_b_shuffled[start_index:end_index]
            yi = y_shuffled[start_index:end_index]

            # The last mini-batch may contain fewer samples.
            current_batch_size = len(xi)

            # Calculate predictions for all samples in the batch.
            y_hat = xi.dot(thetas)

            # Calculate Mean Squared Error for the current batch.
            loss = np.mean((y_hat - yi) ** 2)

            # Calculate gradients averaged over the current mini-batch.
            gradients = (
                2 / current_batch_size
            ) * xi.T.dot(y_hat - yi)

            # Update model parameters.
            thetas -= learning_rate * gradients

            # Save the current parameters and loss.
            thetas_path.append(thetas.copy())
            losses.append(loss)

    return thetas_path, losses


# Train the model using Mini-Batch Gradient Descent.
mbgd_thetas, mbgd_losses = mini_batch_gradient_descent(
    X_b,
    y,
    n_epochs=50,
    minibatch_size=20,
    learning_rate=0.01,
)

# Plot the first 200 mini-batch updates.
plt.figure(figsize=(10, 6))

plt.plot(
    range(min(200, len(mbgd_losses))),
    mbgd_losses[:200],
    color="red",
)

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Mini-Batch Gradient Descent Loss (First 200 Updates)")
plt.grid(alpha=0.3)
plt.show()


# =========================================================
# 6. BATCH GRADIENT DESCENT
# =========================================================
def batch_gradient_descent(
    X_b,
    y,
    n_epochs=100,
    learning_rate=0.01,
):
    """
    Train Linear Regression using Batch Gradient Descent.

    The model calculates one gradient using all training samples
    and updates parameters once per epoch.
    """
    n_samples, n_features_with_bias = X_b.shape

    # Initialize model parameters.
    thetas = np.asarray([
        [1.16270837],
        [-0.81960489],
        [1.39501033],
        [0.29763545],
    ])

    thetas_path = [thetas.copy()]
    losses = []

    for epoch in range(n_epochs):
        # Calculate predictions for all training samples.
        y_hat = X_b.dot(thetas)

        # Calculate Mean Squared Error over the entire dataset.
        loss = np.mean((y_hat - y) ** 2)

        # Calculate the gradient using all training samples.
        gradient = (
            2 / n_samples
        ) * X_b.T.dot(y_hat - y)

        # Update model parameters once per epoch.
        thetas -= learning_rate * gradient

        # Save the current parameters and loss.
        thetas_path.append(thetas.copy())
        losses.append(loss)

    return thetas_path, losses


# Train the model using Batch Gradient Descent.
bgd_thetas, bgd_losses = batch_gradient_descent(
    X_b,
    y,
    n_epochs=100,
    learning_rate=0.01,
)

# Plot loss values across epochs.
plt.figure(figsize=(10, 6))

plt.plot(
    range(len(bgd_losses)),
    bgd_losses,
    color="red",
)

plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Batch Gradient Descent Loss")
plt.grid(alpha=0.3)
plt.show()