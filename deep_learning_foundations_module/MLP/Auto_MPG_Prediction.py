# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# Build custom datasets and load data in mini-batches.
from torch.utils.data import Dataset, DataLoader

# Split data into training, validation, and testing sets.
from sklearn.model_selection import train_test_split

# Standardize numerical features.
from sklearn.preprocessing import StandardScaler


# =========================================================
# 2. READ THE AUTO MPG DATASET
# =========================================================
dataset_path = "datasets/[Data]_Ex1_Auto_MPG_data.csv"

# Load the dataset into a pandas DataFrame.
dataset = pd.read_csv(dataset_path)


# =========================================================
# 3. PREPROCESSING
# =========================================================
# MPG is the target variable.
# All remaining columns are used as input features.
X = dataset.drop(columns="MPG").values
y = dataset["MPG"].values

# Final split ratio:
# - Training: 70%
# - Validation: 20%
# - Testing: 10%
val_size = 0.2
test_size = 0.125
is_shuffle = True
random_state = 59


# ---------------------------------------------------------
# 3.1 SPLIT DATA
# ---------------------------------------------------------
# Create the validation set first.
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=val_size,
    random_state=random_state,
    shuffle=is_shuffle,
)

# Create the test set from the remaining training pool.
X_train, X_test, y_train, y_test = train_test_split(
    X_train,
    y_train,
    test_size=test_size,
    random_state=random_state,
    shuffle=is_shuffle,
)


# ---------------------------------------------------------
# 3.2 FEATURE NORMALIZATION
# ---------------------------------------------------------
# Fit the scaler only on training features to prevent data leakage.
normalizer = StandardScaler()

X_train = normalizer.fit_transform(X_train)
X_val = normalizer.transform(X_val)
X_test = normalizer.transform(X_test)


# ---------------------------------------------------------
# 3.3 CONVERT NUMPY ARRAYS INTO PYTORCH TENSORS
# ---------------------------------------------------------
# Neural networks in PyTorch expect tensors, typically float32.
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# =========================================================
# 4. CREATE PYTORCH DATASETS AND DATALOADERS
# =========================================================
class CustomDataset(Dataset):
    """
    Store input features and target values in a PyTorch Dataset.

    DataLoader uses this class to retrieve one sample at a time
    and combine samples into mini-batches.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, idx):
        """Return one feature-target pair at the given index."""
        return self.X[idx], self.y[idx]


batch_size = 32

# Create Dataset objects.
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

# Shuffle training data in every epoch.
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

# Validation data should not be shuffled because model parameters
# are not updated during validation.
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
)


# =========================================================
# 5. BUILD THE MLP REGRESSION MODEL
# =========================================================
class MLP(nn.Module):
    """
    A Multi-Layer Perceptron for predicting MPG values.

    Architecture:
    input -> hidden -> hidden -> output
    """

    def __init__(self, input_dims, hidden_dims, output_dims):
        super().__init__()

        # First hidden layer.
        self.linear1 = nn.Linear(
            input_dims,
            hidden_dims,
        )

        # Second hidden layer.
        self.linear2 = nn.Linear(
            hidden_dims,
            hidden_dims,
        )

        # Output layer returns one predicted MPG value.
        self.output = nn.Linear(
            hidden_dims,
            output_dims,
        )

    def forward(self, x):
        # Apply the first linear transformation and Tanh activation.
        x = torch.tanh(self.linear1(x))

        # Apply the second linear transformation and Tanh activation.
        x = torch.tanh(self.linear2(x))

        # Remove the final dimension:
        # (batch_size, 1) -> (batch_size,)
        return self.output(x).squeeze(1)


# Get the input feature dimension automatically.
input_dims = X_train.shape[1]

# Regression predicts one continuous value.
output_dims = 1

hidden_dims = 64
learning_rate = 1e-2

# Create the neural network.
model = MLP(
    input_dims,
    hidden_dims,
    output_dims,
)

# Use Mean Squared Error as the regression loss function.
criterion = nn.MSELoss()

# Use Stochastic Gradient Descent to optimize model parameters.
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=learning_rate,
)


# =========================================================
# 6. R-SQUARED METRIC
# =========================================================
def r_squared(y_true, y_pred):
    """
    Calculate the R-squared regression metric.

    Formula:
    R² = 1 - SS_res / SS_tot
    """
    # Ensure inputs are tensors.
    y_true = torch.as_tensor(
        y_true,
        dtype=torch.float32,
    )

    y_pred = torch.as_tensor(
        y_pred,
        dtype=torch.float32,
    )

    # Calculate the total variation in actual target values.
    mean_true = torch.mean(y_true)

    ss_tot = torch.sum(
        (y_true - mean_true) ** 2
    )

    # Calculate the unexplained prediction error.
    ss_res = torch.sum(
        (y_true - y_pred) ** 2
    )

    return 1 - (ss_res / ss_tot)


# =========================================================
# 7. TRAIN AND VALIDATE THE MODEL
# =========================================================
epochs = 100

# Store metrics from every epoch.
train_losses = []
val_losses = []

train_r2_scores = []
val_r2_scores = []

for epoch in range(epochs):
    # -----------------------------------------------------
    # 7.1 TRAINING PHASE
    # -----------------------------------------------------
    # Enable training behavior, such as Dropout or BatchNorm updates.
    model.train()

    train_loss = 0.0
    train_targets = []
    train_predictions = []

    for X_samples, y_samples in train_loader:
        # Clear gradients from the previous batch.
        optimizer.zero_grad()

        # Forward pass: create predictions.
        outputs = model(X_samples)

        # Calculate MSE loss.
        loss = criterion(outputs, y_samples)

        # Backpropagation: calculate gradients.
        loss.backward()

        # Update model parameters.
        optimizer.step()

        # Multiply batch loss by batch size for correct averaging later.
        train_loss += loss.item() * X_samples.size(0)

        # Save predictions and targets for epoch-level R² calculation.
        train_predictions.append(outputs.detach().cpu())
        train_targets.append(y_samples.detach().cpu())

    # Combine all mini-batches into full training tensors.
    train_predictions = torch.cat(train_predictions)
    train_targets = torch.cat(train_targets)

    # Calculate average training loss over all training samples.
    train_loss /= len(train_loader.dataset)

    # Calculate training R².
    train_r2 = r_squared(
        train_targets,
        train_predictions,
    )

    train_losses.append(train_loss)
    train_r2_scores.append(train_r2.item())

    # -----------------------------------------------------
    # 7.2 VALIDATION PHASE
    # -----------------------------------------------------
    # Disable training-only behavior.
    model.eval()

    val_loss = 0.0
    val_targets = []
    val_predictions = []

    # Disable gradient calculation because no learning occurs in validation.
    with torch.no_grad():
        for X_samples, y_samples in val_loader:
            # Forward pass only.
            outputs = model(X_samples)

            # Calculate validation loss.
            loss = criterion(outputs, y_samples)

            # Accumulate weighted loss.
            val_loss += loss.item() * X_samples.size(0)

            # Save predictions and targets for R² calculation.
            val_predictions.append(outputs.cpu())
            val_targets.append(y_samples.cpu())

    # Combine all mini-batches into full validation tensors.
    val_predictions = torch.cat(val_predictions)
    val_targets = torch.cat(val_targets)

    # Calculate average validation loss.
    val_loss /= len(val_loader.dataset)

    # Calculate validation R².
    val_r2 = r_squared(
        val_targets,
        val_predictions,
    )

    val_losses.append(val_loss)
    val_r2_scores.append(val_r2.item())

    print(
        f"Epoch {epoch + 1:03d}/{epochs} | "
        f"Training Loss: {train_loss:.3f} | "
        f"Validation Loss: {val_loss:.3f} | "
        f"Training R²: {train_r2.item():.3f} | "
        f"Validation R²: {val_r2.item():.3f}"
    )


# =========================================================
# 8. VISUALIZE TRAINING HISTORY
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

axes[1, 0].plot(train_r2_scores, color="green")
axes[1, 0].set_title("Training R²")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("R²")
axes[1, 0].grid(alpha=0.3)

axes[1, 1].plot(val_r2_scores, color="orange")
axes[1, 1].set_title("Validation R²")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("R²")
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()


# =========================================================
# 9. FINAL EVALUATION
# =========================================================
model.eval()

with torch.no_grad():
    # Evaluate on the validation set.
    y_val_pred = model(X_val)

    val_set_r2 = r_squared(
        y_val,
        y_val_pred,
    )

    print("\nEvaluation on Validation Set:")
    print(f"R²: {val_set_r2.item():.4f}")

    # Evaluate once on the unseen test set.
    y_test_pred = model(X_test)

    test_set_r2 = r_squared(
        y_test,
        y_test_pred,
    )

    print("\nEvaluation on Test Set:")
    print(f"R²: {test_set_r2.item():.4f}")