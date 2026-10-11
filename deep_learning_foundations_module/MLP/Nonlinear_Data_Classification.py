# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


# =========================================================
# 2. SET RANDOM SEED
# =========================================================
random_state = 59

np.random.seed(random_state)
torch.manual_seed(random_state)


# =========================================================
# 3. LOAD THE DATASET
# =========================================================
current_dir = Path(__file__).resolve().parent

data_path = current_dir / "[Data]_Ex2_NonLinear_data.npy"

# Load the dictionary stored in the .npy file.
data = np.load(
    data_path,
    allow_pickle=True,
).item()

# X contains input features, and y contains class labels.
X = data["X"]
y = data["labels"]

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Labels:", np.unique(y))


# =========================================================
# 4. SPLIT DATA INTO TRAINING, VALIDATION, AND TEST SETS
# =========================================================
# Final split ratio:
# - Training: 70%
# - Validation: 20%
# - Testing: 10%
val_size = 0.2
test_size = 0.125
is_shuffle = True

# Create the validation set first.
X_train, X_val, y_train, y_val = train_test_split(
    X,
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

# Count the number of target classes.
num_classes = len(np.unique(y))

print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of validation samples: {X_val.shape[0]}")
print(f"Number of test samples: {X_test.shape[0]}")


# =========================================================
# 5. NORMALIZE FEATURES
# =========================================================
# Fit the scaler on training data only to prevent data leakage.
normalizer = StandardScaler()

X_train = normalizer.fit_transform(X_train)
X_val = normalizer.transform(X_val)
X_test = normalizer.transform(X_test)

# Convert NumPy arrays into PyTorch tensors.
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

# CrossEntropyLoss requires integer class indices of dtype long.
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# =========================================================
# 6. CREATE PYTORCH DATASETS AND DATALOADERS
# =========================================================
class CustomDataset(Dataset):
    """Store features and labels for use with a DataLoader."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


batch_size = 32

train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
test_dataset = CustomDataset(X_test, y_test)

# Shuffle training data to create different mini-batch orders each epoch.
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

# Validation and test data do not need to be shuffled.
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)


# =========================================================
# 7. BUILD THE MLP CLASSIFICATION MODEL
# =========================================================
class MLP(nn.Module):
    """
    A one-hidden-layer Multi-Layer Perceptron for classification.

    Architecture:
    input -> hidden layer -> output logits
    """

    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()

        self.linear1 = nn.Linear(
            input_dims,
            hidden_dims,
        )

        self.output = nn.Linear(
            hidden_dims,
            output_dims,
        )

    def forward(self, x):
        # Apply a hidden layer followed by Tanh activation.
        x = torch.tanh(self.linear1(x))

        # Return raw class scores for CrossEntropyLoss.
        return self.output(x)


input_dims = X_train.shape[1]
output_dims = num_classes
hidden_dims = 512
learning_rate = 1e-1

model = MLP(
    input_dims=input_dims,
    hidden_dims=hidden_dims,
    output_dims=output_dims,
)

# CrossEntropyLoss combines softmax and negative log-likelihood loss.
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
)


# =========================================================
# 8. METRIC AND EVALUATION FUNCTIONS
# =========================================================
def compute_accuracy(logits, y_true):
    """Calculate classification accuracy from model logits."""
    predicted_classes = torch.argmax(
        logits,
        dim=1,
    )

    correct_predictions = (
        predicted_classes == y_true
    ).sum().item()

    return correct_predictions / len(y_true)


def evaluate_model(model, data_loader, criterion):
    """
    Calculate average loss and accuracy for an entire dataset.
    """
    model.eval()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for X_samples, y_samples in data_loader:
            logits = model(X_samples)
            loss = criterion(logits, y_samples)

            total_loss += loss.item() * X_samples.size(0)

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct_predictions += (
                predictions == y_samples
            ).sum().item()

            total_samples += y_samples.size(0)

    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy


# =========================================================
# 9. TRAIN THE MODEL
# =========================================================
epochs = 100

train_losses = []
val_losses = []

train_accs = []
val_accs = []

# Store the best model across ALL epochs.
best_val_loss = float("inf")
best_model_state = None

for epoch in range(epochs):
    model.train()

    total_train_loss = 0.0

    for X_samples, y_samples in train_loader:
        # Clear gradients from the previous mini-batch.
        optimizer.zero_grad()

        # Forward pass.
        logits = model(X_samples)

        # Calculate classification loss.
        loss = criterion(logits, y_samples)

        # Backpropagation.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        total_train_loss += loss.item() * X_samples.size(0)

    # Calculate average loss during the training updates.
    train_loss = (
        total_train_loss
        / len(train_loader.dataset)
    )

    # Evaluate the updated model on complete training and validation data.
    _, train_acc = evaluate_model(
        model,
        train_loader,
        criterion,
    )

    val_loss, val_acc = evaluate_model(
        model,
        val_loader,
        criterion,
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accs.append(train_acc)
    val_accs.append(val_acc)

    # Save the model with the lowest validation loss.
    if val_loss < best_val_loss:
        best_val_loss = val_loss

        best_model_state = {
            key: value.cpu().clone()
            for key, value in model.state_dict().items()
        }

    print(
        f"Epoch {epoch + 1:03d}/{epochs} | "
        f"Training Loss: {train_loss:.3f} | "
        f"Validation Loss: {val_loss:.3f} | "
        f"Training Accuracy: {train_acc:.3f} | "
        f"Validation Accuracy: {val_acc:.3f}"
    )


# =========================================================
# 10. VISUALIZE TRAINING HISTORY
# =========================================================
fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10),
)

axes[0, 0].plot(train_losses, color="green")
axes[0, 0].set_title("Training Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].grid(alpha=0.3)

axes[0, 1].plot(val_losses, color="orange")
axes[0, 1].set_title("Validation Loss")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Loss")
axes[0, 1].grid(alpha=0.3)

axes[1, 0].plot(train_accs, color="green")
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
# 11. FINAL EVALUATION
# =========================================================
# Restore the best model selected using validation loss.
if best_model_state is not None:
    model.load_state_dict(best_model_state)

val_loss, val_acc = evaluate_model(
    model,
    val_loader,
    criterion,
)

test_loss, test_acc = evaluate_model(
    model,
    test_loader,
    criterion,
)

print("\nEvaluation on Validation Set:")
print(f"Loss: {val_loss:.4f}")
print(f"Accuracy: {val_acc:.4f}")

print("\nEvaluation on Test Set:")
print(f"Loss: {test_loss:.4f}")
print(f"Accuracy: {test_acc:.4f}")