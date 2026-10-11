# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Standardize input features.
from sklearn.preprocessing import StandardScaler

# Split data into training and testing sets.
from sklearn.model_selection import train_test_split

# Evaluation metrics.
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)

# sklearn implementation of Linear Regression.
from sklearn.linear_model import LinearRegression


# =========================================================
# 2. LOAD AND PREPARE THE BITCOIN DATASET
# =========================================================
# Load daily Bitcoin price data.
df = pd.read_csv("./BTC-Daily.csv")

# Remove duplicate rows, if any.
df = df.drop_duplicates()

# Convert the date column from string to datetime format.
df["date"] = pd.to_datetime(df["date"])

# Display the available date range in the dataset.
date_range = (
    str(df["date"].dt.date.min())
    + " to "
    + str(df["date"].dt.date.max())
)

print(f"Dataset date range: {date_range}")

# Extract year, month, and day for yearly visualization.
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day


# =========================================================
# 3. VISUALIZE BITCOIN CLOSING PRICES BY YEAR
# =========================================================
# Get all years available in the dataset.
unique_years = df["year"].unique()

for year in unique_years:
    # Create a complete daily date range for the current year.
    dates = pd.date_range(
        start=f"{year}-01-01",
        end=f"{year}-12-31",
        freq="D",
    )

    # Create a temporary DataFrame containing every calendar day.
    year_month_day = pd.DataFrame({"date": dates})
    year_month_day["year"] = year_month_day["date"].dt.year
    year_month_day["month"] = year_month_day["date"].dt.month
    year_month_day["day"] = year_month_day["date"].dt.day

    # Merge actual Bitcoin records with the full calendar.
    # Missing trading dates will have NaN close values.
    merged_data = pd.merge(
        year_month_day,
        df.drop(columns="date"),
        on=["year", "month", "day"],
        how="left",
    )

    # Plot daily closing prices for the current year.
    plt.figure(figsize=(10, 6))

    plt.plot(
        merged_data["date"],
        merged_data["close"],
    )

    plt.title(f"Bitcoin Closing Prices - {year}")
    plt.xlabel("Date")
    plt.ylabel("Closing Price (USD)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# =========================================================
# 4. CUSTOM LINEAR REGRESSION FUNCTIONS
# =========================================================
def predict(X, w, b):
    """
    Calculate predictions using the linear regression equation.

    Formula:
    y_hat = Xw + b
    """
    return X.dot(w) + b


def gradient(y_hat, y, X):
    """
    Calculate gradients for weights and bias.

    The loss function is Mean Squared Error:

    Cost = sum((y_hat - y)^2) / (2 * n_samples)
    """
    # Calculate prediction errors.
    error = y_hat - y

    # Calculate gradient for weights.
    dw = X.T.dot(error) / len(y)

    # Calculate gradient for bias.
    db = np.sum(error) / len(y)

    # Calculate the current cost.
    cost = np.sum(error ** 2) / (2 * len(y))

    return dw, db, cost


def update_weight(w, b, learning_rate, dw, db):
    """
    Update weights and bias using gradient descent.
    """
    w_new = w - learning_rate * dw
    b_new = b - learning_rate * db

    return w_new, b_new


def linear_regression_vectorized(
    X,
    y,
    learning_rate=0.01,
    num_iterations=200,
):
    """
    Train a Linear Regression model from scratch
    using vectorized gradient descent.
    """
    # Get the number of samples and input features.
    n_samples, n_features = X.shape

    # Initialize weights and bias with zeros.
    w = np.zeros(n_features)
    b = 0.0

    # Store the loss value after every iteration.
    losses = []

    for _ in range(num_iterations):
        # Make predictions using the current parameters.
        y_hat = predict(X, w, b)

        # Calculate gradients and current loss.
        dw, db, cost = gradient(y_hat, y, X)

        # Update the parameters.
        w, b = update_weight(
            w,
            b,
            learning_rate,
            dw,
            db,
        )

        # Save the current loss.
        losses.append(cost)

    return w, b, losses


# =========================================================
# 5. PREPARE FEATURES AND TARGET
# =========================================================
# Use opening, lowest, and highest prices as input features.
X = df[["open", "low", "high"]].values

# Use closing price as the prediction target.
y = df["close"].values


# =========================================================
# 6. SPLIT AND SCALE THE DATA
# =========================================================
# Split data into training and testing sets.
#
# shuffle=True is acceptable for this basic regression exercise.
# For real time-series forecasting, use chronological splitting instead.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    shuffle=True,
)

# Create a feature scaler.
scaler = StandardScaler()

# Fit the scaler only on training features.
X_train_scaled = scaler.fit_transform(X_train)

# Transform test features using training statistics.
X_test_scaled = scaler.transform(X_test)


# =========================================================
# 7. TRAIN CUSTOM LINEAR REGRESSION MODEL
# =========================================================
w, b, losses = linear_regression_vectorized(
    X_train_scaled,
    y_train,
    learning_rate=0.1,
    num_iterations=200,
)


# =========================================================
# 8. VISUALIZE THE TRAINING LOSS
# =========================================================
plt.figure(figsize=(10, 6))

plt.plot(losses)

plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Loss Function During Gradient Descent")
plt.grid(alpha=0.3)
plt.show()


# =========================================================
# 9. EVALUATE THE CUSTOM LINEAR REGRESSION MODEL
# =========================================================
# Predict test values using SCALED test features.
y_pred = predict(X_test_scaled, w, b)

# Calculate regression metrics.
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(
    np.abs((y_test - y_pred) / y_test)
) * 100

# Calculate R-squared scores on training and testing data.
y_train_pred = predict(X_train_scaled, w, b)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_pred)

print("\nCustom Linear Regression Evaluation:")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Training R-squared: {train_r2:.4f}")
print(f"Testing R-squared: {test_r2:.4f}")


# =========================================================
# 10. VISUALIZE CUSTOM MODEL PREDICTIONS FOR 2019 Q1
# =========================================================
# Select Bitcoin records from January 1 to March 31, 2019.
df_2019_q1 = df[
    (df["date"] >= "2019-01-01")
    & (df["date"] <= "2019-03-31")
].copy()

# Extract and scale input features.
X_2019 = df_2019_q1[["open", "low", "high"]].values
X_2019_scaled = scaler.transform(X_2019)

# Generate predicted closing prices using the custom model.
y_pred_2019_custom = predict(
    X_2019_scaled,
    w,
    b,
)

df_2019_q1["predicted_close"] = y_pred_2019_custom

# Plot actual and predicted closing prices.
plt.figure(figsize=(12, 6))

plt.plot(
    df_2019_q1["date"],
    df_2019_q1["close"],
    label="Actual Close Price",
    marker="o",
)

plt.plot(
    df_2019_q1["date"],
    df_2019_q1["predicted_close"],
    label="Predicted Close Price (Custom Model)",
    marker="x",
)

plt.title("Actual vs. Predicted Bitcoin Close Price (2019 Q1)")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# =========================================================
# 11. EVALUATE WITH SCIKIT-LEARN LINEAR REGRESSION
# =========================================================
# Create sklearn's Linear Regression model.
linear_regressor = LinearRegression()

# Train the model using the same scaled training data.
linear_regressor.fit(
    X_train_scaled,
    y_train,
)

# Predict values using scaled test features.
sklearn_y_pred = linear_regressor.predict(
    X_test_scaled
)

# Calculate regression metrics.
sklearn_mse = mean_squared_error(
    y_test,
    sklearn_y_pred,
)

sklearn_mae = mean_absolute_error(
    y_test,
    sklearn_y_pred,
)

sklearn_r2 = r2_score(
    y_test,
    sklearn_y_pred,
)

print("\nsklearn Linear Regression Evaluation:")
print(f"Mean Squared Error (MSE): {sklearn_mse:.4f}")
print(f"Mean Absolute Error (MAE): {sklearn_mae:.4f}")
print(f"R-squared: {sklearn_r2:.4f}")


# =========================================================
# 12. VISUALIZE SKLEARN MODEL PREDICTIONS FOR 2019 Q1
# =========================================================
# Predict 2019 Q1 prices using the sklearn model.
y_pred_2019_sklearn = linear_regressor.predict(
    X_2019_scaled
)

df_2019_q1["sklearn_predicted_close"] = y_pred_2019_sklearn

plt.figure(figsize=(12, 6))

plt.plot(
    df_2019_q1["date"],
    df_2019_q1["close"],
    label="Actual Close Price",
    marker="o",
)

plt.plot(
    df_2019_q1["date"],
    df_2019_q1["sklearn_predicted_close"],
    label="Predicted Close Price (sklearn Model)",
    marker="x",
)

plt.title("Actual vs. Predicted Bitcoin Close Price (sklearn, 2019 Q1)")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.xticks(rotation=45)
plt.legend()
plt.grid(alpha=0.3)
plt.show()