# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CPU performance dataset from OpenML.
from sklearn.datasets import fetch_openml

# Split data into training and testing sets.
from sklearn.model_selection import train_test_split

# Evaluate regression performance.
from sklearn.metrics import mean_squared_error, r2_score

# Build a Decision Tree regression model.
from sklearn.tree import DecisionTreeRegressor

# Standardize numerical features.
from sklearn.preprocessing import StandardScaler


# =========================================================
# 2. LOAD THE MACHINE CPU DATASET
# =========================================================
# Load the Machine CPU dataset from OpenML.
#
# The target is the CPU Relative Performance (PRP),
# which represents the performance level of a computer CPU.
machine_cpu = fetch_openml(
    name="machine_cpu",
    version=1,
)

# Extract input features and target values.
cpu_X = machine_cpu.data
cpu_y = machine_cpu.target.astype(float)

# Store feature names.
cpu_feature_names = machine_cpu.feature_names

# Create a DataFrame for exploratory data analysis.
cpu_df = pd.DataFrame(
    cpu_X,
    columns=cpu_feature_names,
)

# Add the target column to the DataFrame.
cpu_df["performance"] = cpu_y

# Display basic dataset information.
print(f"Dataset shape: {cpu_df.shape}")
print(f"Feature names: {cpu_feature_names}")
print("Target: Performance (CPU Relative Performance)")


# =========================================================
# 3. EXPLORATORY DATA ANALYSIS
# =========================================================
# Create a 3x3 grid for:
# - Six feature histograms
# - One target histogram
# - One correlation heatmap
# - One boxplot
fig, axes = plt.subplots(
    3,
    3,
    figsize=(18, 15),
)

# Flatten the 2D axes array into a 1D array for easier indexing.
axes = axes.flatten()


# ---------------------------------------------------------
# 3.1 FEATURE HISTOGRAMS
# ---------------------------------------------------------
# Plot the distribution of each input feature.
for i, col in enumerate(cpu_feature_names):
    axes[i].hist(
        cpu_df[col],
        bins=20,
        alpha=0.7,
        color="lightblue",
        edgecolor="black",
    )

    axes[i].set_title(col, fontsize=12)
    axes[i].set_xlabel(col, fontsize=10)
    axes[i].set_ylabel("Frequency", fontsize=10)
    axes[i].grid(True, alpha=0.3)


# ---------------------------------------------------------
# 3.2 TARGET HISTOGRAM
# ---------------------------------------------------------
# Plot the distribution of CPU performance values.
axes[6].hist(
    cpu_df["performance"],
    bins=20,
    alpha=0.7,
    color="lightblue",
    edgecolor="black",
)

axes[6].set_title("CPU Relative Performance", fontsize=12)
axes[6].set_xlabel("Performance", fontsize=10)
axes[6].set_ylabel("Frequency", fontsize=10)
axes[6].grid(True, alpha=0.3)


# ---------------------------------------------------------
# 3.3 CORRELATION HEATMAP
# ---------------------------------------------------------
# Calculate correlations between all features and the target.
correlation_matrix_cpu = cpu_df[
    cpu_feature_names + ["performance"]
].corr()

# Visualize the correlation matrix.
sns.heatmap(
    correlation_matrix_cpu,
    annot=True,
    cmap="coolwarm",
    center=0,
    square=True,
    ax=axes[7],
    fmt=".3f",
)

axes[7].set_title("Feature Correlation Matrix", fontsize=12)


# ---------------------------------------------------------
# 3.4 FEATURE BOXPLOT
# ---------------------------------------------------------
# Visualize the spread and possible outliers of input features.
cpu_df.boxplot(
    column=cpu_feature_names,
    ax=axes[8],
)

axes[8].set_title("Feature Boxplot")
axes[8].set_ylabel("Value", fontsize=10)
axes[8].tick_params(
    axis="x",
    rotation=45,
)

# Adjust subplot spacing automatically.
plt.tight_layout()
plt.show()


# =========================================================
# 4. SPLIT DATA INTO TRAINING AND TESTING SETS
# =========================================================
# Use 80% of samples for training and 20% for testing.
#
# random_state=42 ensures reproducible results.
X_train_cpu, X_test_cpu, y_train_cpu, y_test_cpu = train_test_split(
    cpu_X,
    cpu_y,
    test_size=0.2,
    random_state=42,
)


# =========================================================
# 5. FEATURE SCALING
# =========================================================
# Standardize input features using the training-set mean
# and standard deviation.
#
# Scaling is not required for Decision Trees, but it is included
# here to keep the preprocessing pipeline consistent.
scaler_cpu = StandardScaler()

# Fit the scaler on training features and transform them.
X_train_cpu_scaled = scaler_cpu.fit_transform(X_train_cpu)

# Transform test features using training-set scaling parameters.
X_test_cpu_scaled = scaler_cpu.transform(X_test_cpu)


# =========================================================
# 6. TRAIN THE DECISION TREE REGRESSOR
# =========================================================
# max_depth=5 limits the tree depth to help reduce overfitting.
dt_regressor = DecisionTreeRegressor(
    max_depth=5,
    random_state=42,
)

# Train the model to predict CPU relative performance.
dt_regressor.fit(
    X_train_cpu_scaled,
    y_train_cpu,
)


# =========================================================
# 7. MAKE PREDICTIONS
# =========================================================
# Predict CPU performance values for the testing set.
y_pred_cpu = dt_regressor.predict(X_test_cpu_scaled)


# =========================================================
# 8. EVALUATE REGRESSION PERFORMANCE
# =========================================================
# Mean Squared Error:
# average squared difference between actual and predicted values.
mse = mean_squared_error(
    y_test_cpu,
    y_pred_cpu,
)

# Root Mean Squared Error:
# square root of MSE, expressed in the target's original unit.
rmse = np.sqrt(mse)

# Mean Absolute Error:
# average absolute difference between actual and predicted values.
mae = np.mean(
    np.abs(y_test_cpu - y_pred_cpu)
)

# R-squared:
# proportion of target variance explained by the model.
r2 = r2_score(
    y_test_cpu,
    y_pred_cpu,
)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared (R²): {r2:.4f}")