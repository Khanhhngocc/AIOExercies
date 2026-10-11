# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Split data into training, validation, and testing sets.
from sklearn.model_selection import train_test_split

# Standardize numerical input features.
from sklearn.preprocessing import StandardScaler


# =========================================================
# 2. LOAD AND PREPARE THE TITANIC DATASET
# =========================================================
dataset_path = "titanic_modified_dataset.csv"

# Load the dataset and use PassengerId as the DataFrame index.
df = pd.read_csv(
    dataset_path,
    index_col="PassengerId",
)

# Convert the DataFrame into a numerical NumPy array.
dataset_arr = df.to_numpy().astype(np.float64)

# Use all columns except the last as input features.
# Use the last column as the binary target label.
X = dataset_arr[:, :-1]
y = dataset_arr[:, -1]

print(f"Feature shape before adding bias: {X.shape}")

# Add a bias column of ones as the first feature column.
intercept = np.ones(
    (X.shape[0], 1)
)

X_b = np.concatenate(
    (intercept, X),
    axis=1,
)

print(f"Feature shape after adding bias: {X_b.shape}")


# =========================================================
# 3. SPLIT DATA INTO TRAINING, VALIDATION, AND TEST SETS
# =========================================================
# Final split ratio:
# - Training: 70%
# - Validation: 20%
# - Testing: 10%
val_size = 0.2
test_size = 0.125
random_state = 2
is_shuffle = True

# Create the validation set first.
X_train, X_val, y_train, y_val = train_test_split(
    X_b,
    y,
    test_size=val_size,
    random_state=random_state,
    shuffle=is_shuffle,
    stratify=y,
)

# Create the test set from the remaining training pool.
X_train, X_test, y_train, y_test = train_test_split(
    X_train,
    y_train,
    test_size=test_size,
    random_state=random_state,
    shuffle=is_shuffle,
    stratify=y_train,
)

print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of validation samples: {X_val.shape[0]}")
print(f"Number of test samples: {X_test.shape[0]}")


# =========================================================
# 4. STANDARDIZE INPUT FEATURES
# =========================================================
# Keep the first bias column unchanged.
# Standardize only the original feature columns.
normalizer = StandardScaler()

# Fit the scaler on training data only.
X_train[:, 1:] = normalizer.fit_transform(
    X_train[:, 1:]
)

# Transform validation and test data using training statistics.
X_val[:, 1:] = normalizer.transform(
    X_val[:, 1:]
)

X_test[:, 1:] = normalizer.transform(
    X_test[:, 1:]
)


# =========================================================
# 5. LOGISTIC REGRESSION FUNCTIONS
# =========================================================
def sigmoid(z):
    """
    Convert linear logits into probabilities between 0 and 1.

    Formula:
    sigmoid(z) = 1 / (1 + exp(-z))
    """
    # Prevent numerical overflow for extremely large values.
    z = np.clip(z, -500, 500)

    return 1 / (1 + np.exp(-z))


def predict(X, theta):
    """
    Calculate survival probabilities.

    Formula:
    y_hat = sigmoid(X @ theta)
    """
    logits = X.dot(theta)

    return sigmoid(logits)


def compute_loss(y_hat, y_true):
    """
    Calculate Binary Cross-Entropy loss.

    Formula:
    -mean(y*log(y_hat) + (1-y)*log(1-y_hat))
    """
    # Avoid log(0), which is undefined.
    y_hat = np.clip(
        y_hat,
        1e-7,
        1 - 1e-7,
    )

    return np.mean(
        -y_true * np.log(y_hat)
        - (1 - y_true) * np.log(1 - y_hat)
    )


def compute_gradient(X, y_true, y_hat):
    """
    Calculate the gradient of Binary Cross-Entropy loss
    with respect to theta.
    """
    return X.T.dot(y_hat - y_true) / y_true.size


def update_theta(theta, gradient, learning_rate):
    """Update model parameters using gradient descent."""
    return theta - learning_rate * gradient


def compute_accuracy(X, y_true, theta):
    """
    Calculate classification accuracy.

    A probability greater than or equal to 0.5 is classified as 1.
    """
    probabilities = predict(X, theta)

    # Convert predicted probabilities into binary labels.
    y_pred = (probabilities >= 0.5).astype(int)

    return (y_pred == y_true).mean()


# =========================================================
# 6. TRAIN LOGISTIC REGRESSION WITH MINI-BATCH GRADIENT DESCENT
# =========================================================
learning_rate = 0.01
epochs = 100
batch_size = 16

# Initialize parameters with small random values.
np.random.seed(random_state)

theta = np.random.uniform(
    low=-0.01,
    high=0.01,
    size=X_train.shape[1],
)

# Store metrics after each epoch.
train_accs = []
train_losses = []
val_accs = []
val_losses = []

num_train_samples = X_train.shape[0]

for epoch in range(epochs):
    # Shuffle training data at the beginning of each epoch.
    indices = np.arange(num_train_samples)
    np.random.shuffle(indices)

    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    # Update parameters using one mini-batch at a time.
    for start_index in range(0, num_train_samples, batch_size):
        end_index = start_index + batch_size

        X_batch = X_train_shuffled[start_index:end_index]
        y_batch = y_train_shuffled[start_index:end_index]

        # Calculate batch predictions.
        y_hat_batch = predict(X_batch, theta)

        # Calculate gradients and update parameters.
        gradient = compute_gradient(
            X_batch,
            y_batch,
            y_hat_batch,
        )

        theta = update_theta(
            theta,
            gradient,
            learning_rate,
        )

    # Calculate training metrics after all mini-batches are completed.
    train_y_hat = predict(X_train, theta)

    train_loss = compute_loss(
        train_y_hat,
        y_train,
    )

    train_acc = compute_accuracy(
        X_train,
        y_train,
        theta,
    )

    # Calculate validation metrics after the current epoch.
    val_y_hat = predict(X_val, theta)

    val_loss = compute_loss(
        val_y_hat,
        y_val,
    )

    val_acc = compute_accuracy(
        X_val,
        y_val,
        theta,
    )

    # Save epoch metrics.
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(
        f"Epoch {epoch + 1:03d}/{epochs} | "
        f"Training Loss: {train_loss:.4f} | "
        f"Validation Loss: {val_loss:.4f} | "
        f"Training Accuracy: {train_acc:.4f} | "
        f"Validation Accuracy: {val_acc:.4f}"
    )


# =========================================================
# 7. VISUALIZE TRAINING HISTORY
# =========================================================
fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10),
)

axes[0, 0].plot(train_losses)
axes[0, 0].set_title("Training Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(val_losses, color="orange")
axes[0, 1].set_title("Validation Loss")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(train_accs)
axes[1, 0].set_title("Training Accuracy")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Accuracy")
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(val_accs, color="orange")
axes[1, 1].set_title("Validation Accuracy")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================================
# 8. FINAL EVALUATION
# =========================================================
# Evaluate performance on the validation set.
val_set_acc = compute_accuracy(
    X_val,
    y_val,
    theta,
)

print("\nEvaluation on Validation Set:")
print(f"Accuracy: {val_set_acc:.4f}")

# Evaluate the final model on the unseen test set.
test_set_acc = compute_accuracy(
    X_test,
    y_test,
    theta,
)

print("\nEvaluation on Test Set:")
print(f"Accuracy: {test_set_acc:.4f}")