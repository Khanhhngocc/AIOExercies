# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import built-in datasets such as the Iris dataset.
from sklearn import datasets

# Split data into training and testing sets.
from sklearn.model_selection import train_test_split

# Evaluate classification performance.
from sklearn.metrics import accuracy_score, classification_report

# Build and visualize a Decision Tree classifier.
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Standardize feature values.
from sklearn.preprocessing import StandardScaler


# =========================================================
# 2. LOAD AND INSPECT THE IRIS DATASET
# =========================================================
# Load the Iris dataset.
iris = datasets.load_iris()

# Extract feature values and target labels.
iris_X = iris.data
iris_y = iris.target

# Store feature names and class names.
feature_names = iris.feature_names
target_names = iris.target_names

# Display basic dataset information.
print(f"Dataset shape: {iris_X.shape}")
print(f"Number of features: {len(feature_names)}")
print(f"Feature names: {feature_names}")
print(f"Number of classes: {len(target_names)}")
print(f"Class names: {target_names}")


# =========================================================
# 3. CREATE A DATAFRAME FOR EXPLORATORY DATA ANALYSIS
# =========================================================
# Create a DataFrame containing the four numerical features.
iris_df = pd.DataFrame(
    iris_X,
    columns=feature_names,
)

# Add the numerical target label.
iris_df["target"] = iris_y

# Add readable species names:
# 0 -> setosa, 1 -> versicolor, 2 -> virginica.
iris_df["species"] = [
    target_names[i]
    for i in iris_y
]


# =========================================================
# 4. VISUALIZE THE DATASET
# =========================================================
# Create a 3x2 grid:
# - Four boxplots for the four features
# - One correlation heatmap
# - One scatter plot
fig, axes = plt.subplots(
    3,
    2,
    figsize=(15, 18),
)

# ---------------------------------------------------------
# 4.1 BOXPLOTS FOR EACH FEATURE BY SPECIES
# ---------------------------------------------------------
for i, feature in enumerate(feature_names):
    # Select subplot position.
    ax = axes[i // 2, i % 2]

    # Compare the distribution of one feature across species.
    iris_df.boxplot(
        column=feature,
        by="species",
        ax=ax,
    )

    ax.set_title(
        f"{feature} Distribution by Species",
        fontsize=12,
        pad=20,
    )
    ax.set_xlabel("Species", fontsize=10)
    ax.set_ylabel(feature, fontsize=10)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------
# 4.2 CORRELATION HEATMAP
# ---------------------------------------------------------
# Replace the fifth subplot with a correlation heatmap.
axes[2, 0].remove()
axes[2, 0] = plt.subplot(3, 2, 5)

# Calculate correlations between numerical features.
correlation_matrix = iris_df[feature_names].corr()

# Visualize feature correlations.
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    center=0,
    square=True,
    ax=axes[2, 0],
    fmt=".3f",
)

axes[2, 0].set_title(
    "Feature Correlation Matrix",
    fontsize=12,
)


# ---------------------------------------------------------
# 4.3 SCATTER PLOT
# ---------------------------------------------------------
# Replace the sixth subplot with a scatter plot.
axes[2, 1].remove()
axes[2, 1] = plt.subplot(3, 2, 6)

# Use one color for each Iris species.
colors = ["red", "blue", "green"]

for i, species in enumerate(target_names):
    # Select samples belonging to the current species.
    species_data = iris_df[
        iris_df["species"] == species
    ]

    # Plot sepal length against petal length.
    axes[2, 1].scatter(
        species_data["sepal length (cm)"],
        species_data["petal length (cm)"],
        label=species,
        alpha=0.7,
        color=colors[i],
    )

axes[2, 1].set_xlabel("Sepal Length (cm)", fontsize=10)
axes[2, 1].set_ylabel("Petal Length (cm)", fontsize=10)
axes[2, 1].set_title("Sepal Length vs. Petal Length", fontsize=12)

axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

# Adjust spacing between all subplots.
plt.tight_layout()
plt.show()


# =========================================================
# 5. SPLIT DATA INTO TRAINING AND TESTING SETS
# =========================================================
# Use 80% of the samples for training and 20% for testing.
#
# stratify=iris_y keeps the class proportions similar
# in both training and testing sets.
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    iris_X,
    iris_y,
    test_size=0.2,
    random_state=42,
    stratify=iris_y,
)

# Display split sizes and class distributions.
print(f"Training set shape: {X_train_iris.shape}")
print(f"Testing set shape: {X_test_iris.shape}")
print(
    f"Class distribution in training set: "
    f"{np.bincount(y_train_iris)}"
)
print(
    f"Class distribution in testing set: "
    f"{np.bincount(y_test_iris)}"
)


# =========================================================
# 6. FEATURE SCALING
# =========================================================
# Standardize features using training-set statistics:
# mean = 0 and standard deviation = 1.
#
# Scaling is not required for Decision Trees because they split
# values by thresholds, but it does not negatively affect this example.
scaler_iris = StandardScaler()

# Learn scaling parameters from training data and transform it.
X_train_iris_scaled = scaler_iris.fit_transform(X_train_iris)

# Transform test data using the same training-set parameters.
X_test_iris_scaled = scaler_iris.transform(X_test_iris)


# =========================================================
# 7. TRAIN THE DECISION TREE CLASSIFIER
# =========================================================
# max_depth=5 limits the maximum depth of the tree
# to reduce the risk of overfitting.
#
# min_samples_split=10 means a node needs at least 10 samples
# before it can be split into child nodes.
dt_classifier = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=10,
    random_state=42,
)

# Train the model using scaled training features.
dt_classifier.fit(
    X_train_iris_scaled,
    y_train_iris,
)


# =========================================================
# 8. MAKE PREDICTIONS AND EVALUATE THE MODEL
# =========================================================
# Predict species labels for the testing set.
y_pred_iris = dt_classifier.predict(X_test_iris_scaled)

# Print precision, recall, F1-score, and support for each class.
print(
    classification_report(
        y_test_iris,
        y_pred_iris,
        target_names=target_names,
    )
)

# Calculate and display overall classification accuracy.
accuracy = accuracy_score(
    y_test_iris,
    y_pred_iris,
)

print(f"Accuracy: {accuracy:.4f}")


# =========================================================
# 9. VISUALIZE THE TRAINED DECISION TREE
# =========================================================
plt.figure(figsize=(16, 10))

# Visualize the learned decision rules.
plot_tree(
    dt_classifier,
    feature_names=[
        "Sepal_Length",
        "Sepal_Width",
        "Petal_Length",
        "Petal_Width",
    ],
    class_names=target_names,
    filled=True,
    rounded=True,
    fontsize=10,
)

plt.title("Decision Tree for Iris Classification")
plt.show()