# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import numpy as np
import matplotlib.pyplot as plt

# Split data into training, validation, and testing sets.
from sklearn.model_selection import train_test_split

# Standardize pixel features.
from sklearn.preprocessing import StandardScaler

# Load MNIST from TensorFlow, with OpenML as a fallback source.
from tensorflow.keras.datasets import mnist
from sklearn.datasets import fetch_openml


# =========================================================
# 2. SET RANDOM SEED
# =========================================================
# Ensure reproducible random initialization and shuffling.
np.random.seed(59)


# =========================================================
# 3. LOAD THE MNIST DATASET
# =========================================================
X_raw = None
y_raw = None
loaded_from = None

try:
    # TensorFlow provides MNIST as separate train and test splits.
    (X_tr, y_tr), (X_te, y_te) = mnist.load_data()

    # Combine both splits because this notebook creates its own split later.
    X_raw = np.concatenate([X_tr, X_te], axis=0)
    y_raw = np.concatenate([y_tr, y_te], axis=0)

    loaded_from = "tensorflow.keras.datasets.mnist"

except Exception:
    try:
        # Use OpenML if TensorFlow cannot download or load MNIST.
        mnist_data = fetch_openml(
            "mnist_784",
            version=1,
            as_frame=False,
        )

        # Reshape flattened 784-pixel vectors back into 28x28 images.
        X_raw = mnist_data.data.reshape(-1, 28, 28)
        y_raw = mnist_data.target.astype(int)

        loaded_from = "sklearn.fetch_openml('mnist_784')"

    except Exception as error:
        raise RuntimeError(
            "Failed to load MNIST via both TensorFlow and OpenML."
        ) from error

print(f"Loaded MNIST from: {loaded_from}")
print(f"Raw feature shape: {X_raw.shape}")
print(f"Raw target shape: {y_raw.shape}")


# =========================================================
# 4. PREPARE INPUT FEATURES
# =========================================================
# Flatten each 28x28 image into a vector of 784 pixel values.
X_flat = X_raw.reshape(
    X_raw.shape[0],
    -1,
).astype(np.float32)

# Scale pixel values from [0, 255] to [0, 1].
X_flat /= 255.0

n_samples, n_features = X_flat.shape

# Add a bias column of ones.
#
# Final shape:
# (number of samples, 785)
# The first column is for the bias term.
X = np.zeros(
    (n_samples, n_features + 1),
    dtype=np.float32,
)

X[:, 0] = 1.0
X[:, 1:] = X_flat

# Ensure digit labels are integer values from 0 to 9.
y = y_raw.astype(int)

print(f"X shape (with bias): {X.shape}")
print(f"y shape: {y.shape}")
print(f"Classes: {sorted(np.unique(y))}")


# =========================================================
# 5. SPLIT INTO TRAINING, VALIDATION, AND TEST SETS
# =========================================================
# First split:
# 80% training/testing pool and 20% validation data.
val_size = 0.2

# Second split:
# 12.5% of the 80% pool = 10% of the original dataset for testing.
test_size = 0.125

random_state = 2
is_shuffle = True

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=val_size,
    random_state=random_state,
    shuffle=is_shuffle,
    stratify=y,
)

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
# 6. STANDARDIZE PIXEL FEATURES
# =========================================================
# Do not scale the first bias column because it must remain equal to 1.
normalizer = StandardScaler()

# Fit scaling statistics on training pixels only.
X_train[:, 1:] = normalizer.fit_transform(
    X_train[:, 1:]
)

# Apply the same training statistics to validation and test pixels.
X_val[:, 1:] = normalizer.transform(
    X_val[:, 1:]
)

X_test[:, 1:] = normalizer.transform(
    X_test[:, 1:]
)


# =========================================================
# 7. ONE-VS-ALL LOGISTIC REGRESSION FUNCTIONS
# =========================================================
def sigmoid(z):
    """
    Convert logits into probabilities between 0 and 1.

    Formula:
    sigmoid(z) = 1 / (1 + exp(-z))
    """
    # Clip values to prevent overflow in exp(-z).
    z = np.clip(z, -500, 500)

    return 1 / (1 + np.exp(-z))


def compute_binary_loss(y_hat, y_true):
    """
    Calculate Binary Cross-Entropy loss.

    This loss is used once for each One-vs-All classifier.
    """
    # Avoid log(0), which would create undefined values.
    y_hat = np.clip(
        y_hat,
        1e-7,
        1 - 1e-7,
    )

    return (
        -y_true * np.log(y_hat)
        - (1 - y_true) * np.log(1 - y_hat)
    ).mean()


def compute_multiclass_ova_loss(
    probs,
    y_true,
    n_classes=10,
):
    """
    Calculate the average Binary Cross-Entropy loss
    across all One-vs-All classifiers.
    """
    total_loss = 0.0

    for class_index in range(n_classes):
        # Create binary labels:
        # 1 if the sample belongs to this class, otherwise 0.
        y_binary = (y_true == class_index).astype(
            np.float32
        )

        # Calculate Binary Cross-Entropy for this class.
        total_loss += compute_binary_loss(
            probs[:, class_index],
            y_binary,
        )

    # Return the average loss across all digit classes.
    return total_loss / n_classes


def predict_ova(X, theta):
    """
    Calculate One-vs-All probabilities for every class.

    theta shape:
    (n_classes, n_features_with_bias)

    logits shape:
    (batch_size, n_classes)
    """
    logits = X @ theta.T

    return sigmoid(logits)


def compute_gradient_ova(
    X,
    y_true,
    probs,
    n_classes=10,
):
    """
    Calculate gradients for all One-vs-All classifiers.
    """
    batch_size, n_features_with_bias = X.shape

    # Store one gradient vector for each class.
    gradients = np.zeros(
        (n_classes, n_features_with_bias),
        dtype=np.float32,
    )

    for class_index in range(n_classes):
        # Create binary targets for the current class.
        y_binary = (y_true == class_index).astype(
            np.float32
        )

        # Get predicted probabilities for the current class.
        y_hat_class = probs[:, class_index]

        # Calculate the average gradient for this classifier.
        gradients[class_index] = (
            X.T @ (y_hat_class - y_binary)
        ) / batch_size

    return gradients


def update_theta(theta, gradients, learning_rate):
    """Update all model parameters using gradient descent."""
    return theta - learning_rate * gradients


def compute_accuracy_multiclass(X, y_true, theta):
    """
    Predict the class with the highest OvA probability
    and calculate classification accuracy.
    """
    probs = predict_ova(X, theta)

    # Select the class with the highest predicted probability.
    y_pred = probs.argmax(axis=1)

    return (y_pred == y_true).mean()


# =========================================================
# 8. TRAIN THE OvA LOGISTIC REGRESSION MODEL
# =========================================================
learning_rate = 0.1
epochs = 20
batch_size = 256
n_classes = 10

# Reinitialize the seed before parameter initialization.
np.random.seed(random_state)

# Create one weight vector for each digit class.
theta = np.random.uniform(
    low=-0.01,
    high=0.01,
    size=(n_classes, X_train.shape[1]),
).astype(np.float32)

# Store metrics after every epoch.
train_accs = []
train_losses = []
val_accs = []
val_losses = []

num_train_samples = X_train.shape[0]

for epoch in range(epochs):
    # Shuffle training samples before mini-batch training.
    indices = np.arange(num_train_samples)
    np.random.shuffle(indices)

    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    # Train one mini-batch at a time.
    for start_index in range(0, num_train_samples, batch_size):
        end_index = start_index + batch_size

        X_batch = X_train_shuffled[start_index:end_index]
        y_batch = y_train_shuffled[start_index:end_index]

        # Calculate predicted probabilities for all digit classes.
        probs_batch = predict_ova(X_batch, theta)

        # Calculate gradients for all ten OvA classifiers.
        gradients = compute_gradient_ova(
            X_batch,
            y_batch,
            probs_batch,
            n_classes=n_classes,
        )

        # Update model parameters.
        theta = update_theta(
            theta,
            gradients,
            learning_rate,
        )

    # Calculate training metrics after the current epoch.
    probs_train = predict_ova(X_train, theta)

    train_loss = compute_multiclass_ova_loss(
        probs_train,
        y_train,
        n_classes=n_classes,
    )

    train_acc = compute_accuracy_multiclass(
        X_train,
        y_train,
        theta,
    )

    # Calculate validation metrics after the current epoch.
    probs_val = predict_ova(X_val, theta)

    val_loss = compute_multiclass_ova_loss(
        probs_val,
        y_val,
        n_classes=n_classes,
    )

    val_acc = compute_accuracy_multiclass(
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
        f"Epoch {epoch + 1:02d}/{epochs} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Validation Loss: {val_loss:.4f} | "
        f"Train Accuracy: {train_acc:.4f} | "
        f"Validation Accuracy: {val_acc:.4f}"
    )


# =========================================================
# 9. VISUALIZE TRAINING HISTORY
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
# 10. FINAL EVALUATION
# =========================================================
# Evaluate the model on validation data.
val_acc = compute_accuracy_multiclass(
    X_val,
    y_val,
    theta,
)

print("\nEvaluation on Validation Set:")
print(f"Accuracy: {val_acc:.4f}")

# Evaluate the final model once on the unseen test set.
test_acc = compute_accuracy_multiclass(
    X_test,
    y_test,
    theta,
)

print("\nEvaluation on Test Set:")
print(f"Accuracy: {test_acc:.4f}")