# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
import random


# =========================================================
# 2. LOAD AND PREPARE THE ADVERTISING DATASET
# =========================================================
def get_column(data, index):
    """Extract one column from a list of rows."""
    return [row[index] for row in data]


def prepare_data(file_name_dataset):
    """
    Load the advertising dataset and separate features from the target.

    Input features:
    - TV advertising budget
    - Radio advertising budget
    - Newspaper advertising budget

    Target:
    - Sales
    """
    data = np.genfromtxt(
        file_name_dataset,
        delimiter=",",
        skip_header=1,
    ).tolist()

    tv_data = get_column(data, 0)
    radio_data = get_column(data, 1)
    newspaper_data = get_column(data, 2)
    sales_data = get_column(data, 3)

    # X has shape: [TV feature, Radio feature, Newspaper feature].
    X = [tv_data, radio_data, newspaper_data]

    # y contains the sales values.
    y = sales_data

    return X, y


X, y = prepare_data("./advertising.csv")

# Verify the first five values of each feature and target.
my_list = [
    sum(X[0][:5]),
    sum(X[1][:5]),
    sum(X[2][:5]),
    sum(y[:5]),
]

print(my_list)


# =========================================================
# 3. LINEAR REGRESSION FROM SCRATCH
# =========================================================
def initialize_params():
    """
    Initialize model weights with small random values
    and initialize bias with zero.
    """
    w1 = random.gauss(mu=0.0, sigma=0.01)
    w2 = random.gauss(mu=0.0, sigma=0.01)
    w3 = random.gauss(mu=0.0, sigma=0.01)
    b = 0.0

    return w1, w2, w3, b


def predict(x1, x2, x3, w1, w2, w3, b):
    """
    Calculate the predicted sales value.

    Formula:
    y_hat = x1*w1 + x2*w2 + x3*w3 + b
    """
    return x1 * w1 + x2 * w2 + x3 * w3 + b


def compute_loss(y_hat, y):
    """
    Calculate Squared Error loss for one training sample.

    Formula:
    loss = (y_hat - y)^2
    """
    return (y_hat - y) ** 2


def compute_gradient_wi(xi, y, y_hat):
    """
    Calculate the gradient of the loss with respect to one weight.

    Formula:
    dL/dwi = 2 * (y_hat - y) * xi
    """
    return 2 * (y_hat - y) * xi


def compute_gradient_b(y, y_hat):
    """
    Calculate the gradient of the loss with respect to the bias.

    Formula:
    dL/db = 2 * (y_hat - y)
    """
    return 2 * (y_hat - y)


def update_weight_wi(wi, dl_dwi, learning_rate):
    """Update one weight using gradient descent."""
    return wi - dl_dwi * learning_rate


def update_weight_b(b, dl_db, learning_rate):
    """Update the bias using gradient descent."""
    return b - dl_db * learning_rate


# =========================================================
# 4. TEST INDIVIDUAL FUNCTIONS
# =========================================================
# Test the loss calculation.
loss_example = compute_loss(y_hat=1, y=0.5)

# Test a weight gradient calculation.
gradient_weight_example = compute_gradient_wi(
    xi=1.0,
    y=1.0,
    y_hat=0.5,
)
print(gradient_weight_example)

# Test a bias gradient calculation.
gradient_bias_example = compute_gradient_b(
    y=2.0,
    y_hat=0.5,
)
print(gradient_bias_example)

# Test the weight update function.
updated_weight_example = update_weight_wi(
    wi=1.0,
    dl_dwi=-0.5,
    learning_rate=1e-5,
)
print(updated_weight_example)

# Test the bias update function.
updated_bias_example = update_weight_b(
    b=0.5,
    dl_db=-1.0,
    learning_rate=1e-5,
)
print(updated_bias_example)


# =========================================================
# 5. TRAIN THE LINEAR REGRESSION MODEL
# =========================================================
def implement_linear_regression(
    X_data,
    y_data,
    epoch_max=50,
    learning_rate=1e-5,
):
    """
    Train a multiple linear regression model using stochastic
    gradient descent.

    The model updates its parameters after each training sample.
    """
    losses = []

    # Initialize model parameters.
    w1, w2, w3, b = initialize_params()

    # Get the number of training samples.
    n_samples = len(y_data)

    for epoch in range(epoch_max):
        for i in range(n_samples):
            # Get feature values for the current sample.
            x1 = X_data[0][i]
            x2 = X_data[1][i]
            x3 = X_data[2][i]

            # Get the actual sales value.
            y_true = y_data[i]

            # Predict sales using the current parameters.
            y_hat = predict(
                x1,
                x2,
                x3,
                w1,
                w2,
                w3,
                b,
            )

            # Calculate the loss for the current sample.
            loss = compute_loss(y_hat, y_true)

            # Calculate gradients for weights and bias.
            dl_dw1 = compute_gradient_wi(x1, y_true, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y_true, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y_true, y_hat)
            dl_db = compute_gradient_b(y_true, y_hat)

            # Update model parameters.
            w1 = update_weight_wi(
                w1,
                dl_dw1,
                learning_rate,
            )
            w2 = update_weight_wi(
                w2,
                dl_dw2,
                learning_rate,
            )
            w3 = update_weight_wi(
                w3,
                dl_dw3,
                learning_rate,
            )
            b = update_weight_b(
                b,
                dl_db,
                learning_rate,
            )

            # Save the loss after each parameter update.
            losses.append(loss)

    return w1, w2, w3, b, losses


w1, w2, w3, b, losses = implement_linear_regression(
    X,
    y,
)

# Display the first 100 loss values.
print(losses[:100])

# Visualize loss changes during the first 100 iterations.
plt.plot(losses[:100])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss During Training")
plt.grid(alpha=0.3)
plt.show()


# =========================================================
# 6. PREDICT SALES FOR A NEW ADVERTISING RECORD
# =========================================================
tv = 19.2
radio = 35.9
newspaper = 51.3

# Train the model again before making a prediction.
w1, w2, w3, b, losses = implement_linear_regression(
    X,
    y,
    epoch_max=50,
    learning_rate=1e-5,
)

# Predict sales for the new advertising budgets.
sales = predict(
    tv,
    radio,
    newspaper,
    w1,
    w2,
    w3,
    b,
)

print(f"Predicted sales: {sales:.4f}")


# =========================================================
# 7. FEATURE NORMALIZATION FUNCTIONS
# =========================================================
def min_max_scaling(X):
    """
    Scale each feature into the range [0, 1].

    Formula:
    x_scaled = (x - min) / (max - min)
    """
    mins = []
    maxs = []
    X_scaled = []

    for feature in X:
        min_val = min(feature)
        max_val = max(feature)

        mins.append(min_val)
        maxs.append(max_val)

        scaled_feature = []

        for x in feature:
            scaled_x = (x - min_val) / (max_val - min_val)
            scaled_feature.append(scaled_x)

        X_scaled.append(scaled_feature)

    return X_scaled, mins, maxs


def inverse_min_max_scaling(X_scaled, mins, maxs):
    """
    Restore Min-Max scaled features to their original values.

    Formula:
    x_original = x_scaled * (max - min) + min
    """
    X_recovered = []

    for i in range(len(X_scaled)):
        feature = X_scaled[i]
        min_val = mins[i]
        max_val = maxs[i]

        recovered_feature = []

        for x in feature:
            original_x = x * (max_val - min_val) + min_val
            recovered_feature.append(original_x)

        X_recovered.append(recovered_feature)

    return X_recovered


def z_score_scaling(X):
    """
    Standardize each feature using Z-score normalization.

    Formula:
    x_scaled = (x - mean) / standard_deviation
    """
    means = []
    stds = []
    X_scaled = []

    for feature in X:
        mean_val = sum(feature) / len(feature)
        means.append(mean_val)

        # Calculate sample standard deviation.
        std_val = (
            sum((x - mean_val) ** 2 for x in feature)
            / (len(feature) - 1)
        ) ** 0.5

        stds.append(std_val)

        scaled_feature = []

        for x in feature:
            scaled_x = (x - mean_val) / std_val
            scaled_feature.append(scaled_x)

        X_scaled.append(scaled_feature)

    return X_scaled, means, stds


def inverse_z_score_scaling(X_scaled, means, stds):
    """
    Restore Z-score scaled features to their original values.

    Formula:
    x_original = x_scaled * standard_deviation + mean
    """
    X_recovered = []

    for i in range(len(X_scaled)):
        feature = X_scaled[i]
        mean_val = means[i]
        std_val = stds[i]

        recovered_feature = []

        for x in feature:
            original_x = x * std_val + mean_val
            recovered_feature.append(original_x)

        X_recovered.append(recovered_feature)

    return X_recovered


# =========================================================
# 8. TEST NORMALIZATION FUNCTIONS
# =========================================================
data1 = [1, 2, 3]
data2 = [4, 5, 6]
data3 = [7, 8, 9]

# Use a different variable name so the original advertising data X is preserved.
X_demo = [data1, data2, data3]

# Test Min-Max scaling and its inverse transformation.
X_minmax, mins, maxs = min_max_scaling(X_demo)

print("Min-Max Scaled:", X_minmax)
print(
    "Recovered Min-Max:",
    inverse_min_max_scaling(X_minmax, mins, maxs),
)

# Test Z-score scaling and its inverse transformation.
X_zscore, means, stds = z_score_scaling(X_demo)

print("\nZ-Score Scaled:", X_zscore)
print(
    "Recovered Z-Score:",
    inverse_z_score_scaling(X_zscore, means, stds),
)