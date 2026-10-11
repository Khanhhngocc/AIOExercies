# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# Split data into training, validation, and testing sets.
from sklearn.model_selection import train_test_split

# Standardize numerical features.
from sklearn.preprocessing import StandardScaler

# Tokenize social-media text such as tweets.
from nltk.tokenize import TweetTokenizer

# Store word frequencies with a default value of zero.
from collections import defaultdict


# =========================================================
# 2. LOAD THE SENTIMENT ANALYSIS DATASET
# =========================================================
dataset_path = "./sentiment_analysis.csv"

# Load the dataset and use the id column as the DataFrame index.
df = pd.read_csv(
    dataset_path,
    index_col="id",
)

# Display dataset structure and numerical statistics.
df.info()
print(df.describe())


# =========================================================
# 3. TEXT NORMALIZATION AND TOKENIZATION
# =========================================================
def text_normalize(text):
    """
    Clean a tweet and convert it into a list of normalized tokens.

    Processing steps:
    - Remove the retweet prefix, such as "RT ".
    - Remove URLs.
    - Remove punctuation.
    - Convert text to lowercase.
    - Remove @mentions.
    - Reduce repeated characters.
    """
    # Remove the retweet prefix at the start of text.
    text = re.sub(r"^RT[\s]+", "", text)

    # Remove URLs.
    text = re.sub(r"https?://\S+", "", text)

    # Remove punctuation while preserving letters, numbers, and whitespace.
    text = re.sub(r"[^\w\s]", "", text)

    # Create a tokenizer suitable for tweet-like text.
    tokenizer = TweetTokenizer(
        preserve_case=False,
        strip_handles=True,
        reduce_len=True,
    )

    return tokenizer.tokenize(text)


# Test the text normalization function.
example_text = """We love this! Would you go?
#talk #makememories #unplug
#relax #iphone #smartphone #wifi #connect...
http://fb.me/6N3LsUpCu
"""

print(text_normalize(example_text))


# =========================================================
# 4. SPLIT RAW TEXT DATA FIRST
# =========================================================
# Final split ratio:
# - Training: 70%
# - Validation: 20%
# - Testing: 10%
val_size = 0.2
test_size = 0.125
random_state = 2
is_shuffle = True

# Separate raw tweets from labels before building word frequencies.
tweets = df["tweet"].to_numpy()
labels = df["label"].to_numpy()

# Create the validation split.
tweets_train, tweets_val, y_train, y_val = train_test_split(
    tweets,
    labels,
    test_size=val_size,
    random_state=random_state,
    shuffle=is_shuffle,
    stratify=labels,
)

# Create the test split from the remaining training pool.
tweets_train, tweets_test, y_train, y_test = train_test_split(
    tweets_train,
    y_train,
    test_size=test_size,
    random_state=random_state,
    shuffle=is_shuffle,
    stratify=y_train,
)

print(f"Number of training samples: {len(tweets_train)}")
print(f"Number of validation samples: {len(tweets_val)}")
print(f"Number of test samples: {len(tweets_test)}")


# =========================================================
# 5. BUILD WORD FREQUENCIES FROM TRAINING DATA ONLY
# =========================================================
def get_freqs(tweets, labels):
    """
    Count how frequently each token appears in each sentiment class.

    Example key:
    ("happy", 1) -> number of times "happy" appears in positive tweets.
    """
    freqs = defaultdict(int)

    for tweet, label in zip(tweets, labels):
        tokens = text_normalize(tweet)

        for token in tokens:
            freqs[(token, label)] += 1

    return freqs


# Build vocabulary statistics only from training tweets.
# This prevents information leakage from validation and test data.
freqs = get_freqs(
    tweets_train,
    y_train,
)


# =========================================================
# 6. CONVERT TWEETS INTO NUMERICAL FEATURES
# =========================================================
def get_feature(text, freqs):
    """
    Convert one tweet into a three-value feature vector.

    Features:
    - X[0]: Bias term, always equal to 1.
    - X[1]: Total frequency of tweet tokens in negative training tweets.
    - X[2]: Total frequency of tweet tokens in positive training tweets.
    """
    tokens = text_normalize(text)

    # Initialize [bias, negative_frequency, positive_frequency].
    X = np.zeros(3)
    X[0] = 1.0

    for token in tokens:
        # Add the word frequency in negative tweets.
        X[1] += freqs[(token, 0)]

        # Add the word frequency in positive tweets.
        X[2] += freqs[(token, 1)]

    return X


def create_feature_matrix(tweets, freqs):
    """Convert a collection of tweets into a feature matrix."""
    return np.array(
        [get_feature(tweet, freqs) for tweet in tweets],
        dtype=np.float64,
    )


# Generate feature matrices using training-only word frequencies.
X_train = create_feature_matrix(
    tweets_train,
    freqs,
)

X_val = create_feature_matrix(
    tweets_val,
    freqs,
)

X_test = create_feature_matrix(
    tweets_test,
    freqs,
)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")


# =========================================================
# 7. STANDARDIZE NON-BIAS FEATURES
# =========================================================
# Keep the bias column unchanged.
# Standardize only negative and positive frequency features.
normalizer = StandardScaler()

# Fit the scaler on training features only.
X_train[:, 1:] = normalizer.fit_transform(
    X_train[:, 1:]
)

# Apply the same transformation to validation and test data.
X_val[:, 1:] = normalizer.transform(
    X_val[:, 1:]
)

X_test[:, 1:] = normalizer.transform(
    X_test[:, 1:]
)


# =========================================================
# 8. LOGISTIC REGRESSION FUNCTIONS
# =========================================================
def sigmoid(z):
    """Convert logits into probabilities between 0 and 1."""
    # Prevent overflow for extremely large positive or negative values.
    z = np.clip(z, -500, 500)

    return 1 / (1 + np.exp(-z))


def compute_loss(y_hat, y_true):
    """
    Calculate Binary Cross-Entropy loss.
    """
    # Avoid log(0), which is undefined.
    y_hat = np.clip(
        y_hat,
        1e-7,
        1 - 1e-7,
    )

    return (
        -y_true * np.log(y_hat)
        - (1 - y_true) * np.log(1 - y_hat)
    ).mean()


def predict(X, theta):
    """
    Calculate the positive-sentiment probability.

    Formula:
    y_hat = sigmoid(X @ theta)
    """
    logits = X.dot(theta)

    return sigmoid(logits)


def compute_gradient(X, y_true, y_hat):
    """
    Calculate the Binary Cross-Entropy gradient.
    """
    return X.T.dot(y_hat - y_true) / y_true.size


def update_theta(theta, gradient, learning_rate):
    """Update model parameters using gradient descent."""
    return theta - learning_rate * gradient


def compute_accuracy(X, y_true, theta):
    """
    Calculate binary classification accuracy.

    Probabilities greater than or equal to 0.5 are predicted as class 1.
    """
    probabilities = predict(X, theta)
    y_pred = (probabilities >= 0.5).astype(int)

    return (y_pred == y_true).mean()


# =========================================================
# 9. TRAIN WITH MINI-BATCH GRADIENT DESCENT
# =========================================================
learning_rate = 0.01
epochs = 200
batch_size = 128

# Initialize weights with small random values.
np.random.seed(random_state)

theta = np.random.uniform(
    low=-0.01,
    high=0.01,
    size=X_train.shape[1],
)

# Store model metrics after every epoch.
train_accs = []
train_losses = []
val_accs = []
val_losses = []

num_train_samples = X_train.shape[0]

for epoch in range(epochs):
    # Shuffle training data at the beginning of every epoch.
    indices = np.arange(num_train_samples)
    np.random.shuffle(indices)

    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    # Update parameters one mini-batch at a time.
    for start_index in range(0, num_train_samples, batch_size):
        end_index = start_index + batch_size

        X_batch = X_train_shuffled[start_index:end_index]
        y_batch = y_train_shuffled[start_index:end_index]

        # Calculate predicted probabilities for this batch.
        y_hat_batch = predict(X_batch, theta)

        # Calculate the gradient and update model weights.
        gradient = compute_gradient(
            X_batch,
            y_batch,
            y_hat_batch,
        )

        theta = update_theta(
            theta,
            gradient,
            learning_rate,
        )

    # Calculate metrics once after all mini-batches in the epoch.
    train_y_hat = predict(X_train, theta)
    train_loss = compute_loss(train_y_hat, y_train)
    train_acc = compute_accuracy(X_train, y_train, theta)

    val_y_hat = predict(X_val, theta)
    val_loss = compute_loss(val_y_hat, y_val)
    val_acc = compute_accuracy(X_val, y_val, theta)

    # Save epoch metrics.
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(
        f"Epoch {epoch + 1:03d}/{epochs} | "
        f"Training Loss: {train_loss:.4f} | "
        f"Validation Loss: {val_loss:.4f} | "
        f"Training Accuracy: {train_acc:.4f} | "
        f"Validation Accuracy: {val_acc:.4f}"
    )


# =========================================================
# 10. VISUALIZE TRAINING HISTORY
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
# 11. FINAL EVALUATION
# =========================================================
val_set_acc = compute_accuracy(
    X_val,
    y_val,
    theta,
)

print("\nEvaluation on Validation Set:")
print(f"Accuracy: {val_set_acc:.4f}")

test_set_acc = compute_accuracy(
    X_test,
    y_test,
    theta,
)

print("\nEvaluation on Test Set:")
print(f"Accuracy: {test_set_acc:.4f}")