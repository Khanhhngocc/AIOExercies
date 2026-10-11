# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import pandas as pd

# Ensemble regression models.
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
)

# Convert categorical columns into numerical values.
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Split data into training and validation sets.
from sklearn.model_selection import train_test_split

# Evaluate regression performance.
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# 2. LOAD THE HOUSING DATASET
# =========================================================
dataset_path = "./Housing.csv"

# Load the dataset into a DataFrame.
df = pd.read_csv(dataset_path)

# The target variable is the house price.
target_column = "price"

# Separate input features from the target.
X = df.drop(columns=target_column)
y = df[target_column]


# =========================================================
# 3. SPLIT DATA INTO TRAINING AND VALIDATION SETS
# =========================================================
# Use 70% of data for training and 30% for validation.
#
# Split before fitting preprocessing tools to prevent data leakage.
test_size = 0.3
random_state = 1
is_shuffle = True

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    shuffle=is_shuffle,
)

print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of validation samples: {X_val.shape[0]}")


# =========================================================
# 4. ENCODE CATEGORICAL FEATURES
# =========================================================
# Find categorical columns, such as furnishingstatus,
# mainroad, guestroom, and airconditioning.
categorical_cols = X_train.select_dtypes(
    include=["object"]
).columns.to_list()

# Find numerical columns.
numerical_cols = X_train.select_dtypes(
    exclude=["object"]
).columns.to_list()

# OrdinalEncoder converts each category into a numerical ID.
ordinal_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,
)

# Fit the encoder only on training data.
X_train_categorical = ordinal_encoder.fit_transform(
    X_train[categorical_cols]
)

# Transform validation data using the categories learned from training data.
X_val_categorical = ordinal_encoder.transform(
    X_val[categorical_cols]
)

# Convert encoded categorical arrays back into DataFrames.
X_train_categorical_df = pd.DataFrame(
    X_train_categorical,
    columns=categorical_cols,
    index=X_train.index,
)

X_val_categorical_df = pd.DataFrame(
    X_val_categorical,
    columns=categorical_cols,
    index=X_val.index,
)

# Combine numerical and encoded categorical features.
X_train_encoded = pd.concat(
    [
        X_train[numerical_cols],
        X_train_categorical_df,
    ],
    axis=1,
)

X_val_encoded = pd.concat(
    [
        X_val[numerical_cols],
        X_val_categorical_df,
    ],
    axis=1,
)


# =========================================================
# 5. FEATURE SCALING
# =========================================================
# Tree-based models do not require scaling.
# It is included here only to keep a general preprocessing pipeline.
scaler = StandardScaler()

# Fit the scaler on training features only.
X_train_scaled = scaler.fit_transform(X_train_encoded)

# Transform validation features using training statistics.
X_val_scaled = scaler.transform(X_val_encoded)


# =========================================================
# 6. TRAIN AND EVALUATE RANDOM FOREST REGRESSOR
# =========================================================
random_forest_regressor = RandomForestRegressor(
    random_state=random_state,
)

# Train the model.
random_forest_regressor.fit(
    X_train_scaled,
    y_train,
)

# Predict house prices for the validation set.
rf_y_pred = random_forest_regressor.predict(X_val_scaled)

# Calculate evaluation metrics.
rf_mae = mean_absolute_error(y_val, rf_y_pred)
rf_mse = mean_squared_error(y_val, rf_y_pred)

print("\nRandom Forest Evaluation Results:")
print(f"Mean Absolute Error: {rf_mae:.2f}")
print(f"Mean Squared Error: {rf_mse:.2f}")


# =========================================================
# 7. TRAIN AND EVALUATE ADABOOST REGRESSOR
# =========================================================
ada_boost_regressor = AdaBoostRegressor(
    random_state=random_state,
)

# Train the model.
ada_boost_regressor.fit(
    X_train_scaled,
    y_train,
)

# Predict house prices for the validation set.
ada_y_pred = ada_boost_regressor.predict(X_val_scaled)

# Calculate evaluation metrics.
ada_mae = mean_absolute_error(y_val, ada_y_pred)
ada_mse = mean_squared_error(y_val, ada_y_pred)

print("\nAdaBoost Evaluation Results:")
print(f"Mean Absolute Error: {ada_mae:.2f}")
print(f"Mean Squared Error: {ada_mse:.2f}")