# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import the built-in Iris dataset from scikit-learn.
from sklearn import datasets

# Split data into training and testing sets.
from sklearn.model_selection import train_test_split

# Calculate classification accuracy.
from sklearn.metrics import accuracy_score

# Standardize numerical features before applying KNN.
from sklearn.preprocessing import StandardScaler

# Import the K-Nearest Neighbors classifier.
from sklearn.neighbors import KNeighborsClassifier


# =========================================================
# 2. LOAD THE IRIS DATASET
# =========================================================
# iris_X contains the input features:
# sepal length, sepal width, petal length, and petal width.
#
# iris_y contains the target class labels:
# 0 = setosa, 1 = versicolor, 2 = virginica.
iris_X, iris_y = datasets.load_iris(return_X_y=True)


# =========================================================
# 3. SPLIT DATA INTO TRAINING AND TESTING SETS
# =========================================================
# Use 80% of the data for training and 20% for testing.
#
# random_state=42 ensures that the same split is created
# every time the code is run.
X_train, X_test, y_train, y_test = train_test_split(
    iris_X,
    iris_y,
    test_size=0.2,
    random_state=42,
)


# =========================================================
# 4. FEATURE SCALING
# =========================================================
# KNN calculates distance between samples.
# Therefore, features should be standardized so that
# large-scale features do not dominate smaller-scale features.
scaler = StandardScaler()

# Learn the mean and standard deviation from training data,
# then transform the training features.
X_train = scaler.fit_transform(X_train)

# Transform test data using the same scaling parameters
# learned from the training data.
X_test = scaler.transform(X_test)


# =========================================================
# 5. TRAIN A KNN MODEL WITH K = 3
# =========================================================
# n_neighbors=3 means the model uses the 3 nearest training samples
# to decide the class of a new sample.
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the KNN classifier.
knn_classifier.fit(X_train, y_train)

# Predict labels for the test set.
y_pred = knn_classifier.predict(X_test)

# Calculate the accuracy of the K=3 model.
initial_accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy with K = 3: {initial_accuracy:.4f}")


# =========================================================
# 6. FIND THE BEST K VALUE
# =========================================================
# Test K values from 1 to 30.
k_values = [i for i in range(1, 31)]

# Store the test accuracy obtained from each K value.
scores = []

for k in k_values:
    # Create a new KNN model with the current K value.
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Train the model using the scaled training data.
    knn_classifier.fit(X_train, y_train)

    # Predict labels for the test set.
    y_pred = knn_classifier.predict(X_test)

    # Calculate test accuracy.
    score = accuracy_score(y_test, y_pred)

    # Append the accuracy score for the current K.
    # np.mean(score) is unnecessary here because score is already one number.
    scores.append(score)


# =========================================================
# 7. VISUALIZE K VALUES AND ACCURACY SCORES
# =========================================================
# Plot the relationship between K and test accuracy.
sns.lineplot(
    x=k_values,
    y=scores,
    marker="o",
)

plt.xlabel("K Values")
plt.ylabel("Accuracy Score")
plt.title("KNN Accuracy for Different K Values")
plt.grid(alpha=0.3)
plt.show()