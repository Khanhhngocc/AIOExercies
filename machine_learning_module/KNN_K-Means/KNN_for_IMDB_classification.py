# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import numpy as np

# Load the IMDB dataset from Hugging Face Datasets.
from datasets import load_dataset

# K-Nearest Neighbors classifier.
from sklearn.neighbors import KNeighborsClassifier

# Standardize numerical feature values.
from sklearn.preprocessing import StandardScaler

# Convert text documents into Bag-of-Words feature vectors.
from sklearn.feature_extraction.text import CountVectorizer

# Evaluate classification performance.
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


# =========================================================
# 2. LOAD THE IMDB SENTIMENT DATASET
# =========================================================
# The IMDB dataset contains movie reviews with binary sentiment labels:
# 0 = negative review
# 1 = positive review
imdb = load_dataset("stanfordnlp/imdb")

# Separate the dataset into training and testing splits.
imdb_train = imdb["train"]
imdb_test = imdb["test"]

# Display the number of reviews in each split.
print("Number of training reviews:", len(imdb_train["text"]))
print("Number of testing reviews :", len(imdb_test["text"]))


# =========================================================
# 3. CONVERT TEXT INTO NUMERICAL FEATURES
# =========================================================
# CountVectorizer creates a Bag-of-Words representation.
#
# Each review becomes a vector in which every position represents
# the count of one vocabulary word in that review.
#
# max_features=1000 keeps only the 1,000 most frequent words
# in the training dataset.
vectorizer = CountVectorizer(max_features=1000)

# Learn the vocabulary from training reviews and transform them
# into a document-term matrix.
#
# toarray() converts the sparse matrix into a dense NumPy array.
X_train = vectorizer.fit_transform(
    imdb_train["text"]
).toarray()

# Transform test reviews using the vocabulary learned from training data.
X_test = vectorizer.transform(
    imdb_test["text"]
).toarray()


# =========================================================
# 4. PREPARE TARGET LABELS
# =========================================================
# Convert Hugging Face Dataset labels into NumPy arrays.
y_train = np.array(imdb_train["label"])
y_test = np.array(imdb_test["label"])


# =========================================================
# 5. FEATURE SCALING
# =========================================================
# KNN is distance-based, so feature scaling is usually applied.
#
# However, for Bag-of-Words data, StandardScaler can make the
# feature matrix dense and consume a large amount of RAM.
# This approach is suitable only for a small experiment.
scaler = StandardScaler()

# Learn scaling statistics from training data and transform it.
X_train = scaler.fit_transform(X_train)

# Transform test data using the same training statistics.
X_test = scaler.transform(X_test)


# =========================================================
# 6. TRAIN THE KNN CLASSIFIER
# =========================================================
# Use the 3 nearest training reviews to predict the sentiment
# of each test review.
knn_classifier = KNeighborsClassifier(n_neighbors=3)

# Train the model with vectorized and scaled training reviews.
knn_classifier.fit(X_train, y_train)


# =========================================================
# 7. MAKE PREDICTIONS
# =========================================================
# Predict sentiment labels for all test reviews.
y_pred = knn_classifier.predict(X_test)


# =========================================================
# 8. EVALUATE THE MODEL
# =========================================================
# Accuracy:
# proportion of correctly classified reviews.
accuracy = accuracy_score(y_test, y_pred)

# F1 score:
# harmonic mean of precision and recall.
# This is useful when class balance is a concern.
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1:.4f}")

# Display precision, recall, F1 score, and support for each sentiment class.
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Negative (0)", "Positive (1)"],
    )
)