# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import numpy as np


# =========================================================
# 2. CREATE TRAINING DATA
# =========================================================
def create_training_data():
    data = [
        ["Sunny",    "Hot",  "High",   "Weak",   "No"],
        ["Sunny",    "Hot",  "High",   "Strong", "No"],
        ["Overcast", "Hot",  "High",   "Weak",   "Yes"],
        ["Rain",     "Mild", "High",   "Weak",   "Yes"],
        ["Rain",     "Cool", "Normal", "Weak",   "Yes"],
        ["Rain",     "Cool", "Normal", "Strong", "No"],
        ["Overcast", "Cool", "Normal", "Strong", "Yes"],
        ["Overcast", "Mild", "High",   "Weak",   "No"],
        ["Sunny",    "Cool", "Normal", "Weak",   "Yes"],
        ["Rain",     "Mild", "Normal", "Weak",   "Yes"],
    ]

    # Use dtype=object because the array contains strings.
    return np.array(data, dtype=object)

train_data = create_training_data()


# =========================================================
# 3. SIMPLE CATEGORICAL NAIVE BAYES MODEL
# =========================================================
class SimpleCategoricalNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    # =====================================================
    # 3.1 TRAIN THE MODEL
    # =====================================================
    def fit(self, X, y):
        # Convert inputs into NumPy object arrays.
        X = np.asarray(X, dtype=object)
        y = np.asarray(y, dtype=object)

        # Validate that X contains rows and columns.
        if X.ndim != 2:
            raise ValueError("X must be a 2D array: shape (n_samples, n_features).")

        # Validate that each feature row has one corresponding target label.
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

        # Save dataset dimensions.
        self.n_samples_, self.n_features_ = X.shape

        # Find every unique target class, for example: ["No", "Yes"].
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Find all possible values for each feature.
        #
        # Example:
        # feature_values_[0] = ["Overcast", "Rain", "Sunny"]
        # feature_values_[1] = ["Cool", "Hot", "Mild"]
        self.feature_values_ = [
            np.unique(X[:, feature_idx])
            for feature_idx in range(self.n_features_)
        ]

        # Store the number of training samples in each class.
        #
        # Example:
        # class_count_[0] = number of "No" samples
        # class_count_[1] = number of "Yes" samples
        self.class_count_ = np.zeros(self.n_classes_)

        # Store category frequencies for each feature.
        self.category_count_ = []

        # Create an empty count matrix for each feature.
        for feature_idx in range(self.n_features_):
            n_values = len(self.feature_values_[feature_idx])
            counts = np.zeros((self.n_classes_, n_values))
            self.category_count_.append(counts)

        # Count feature values inside each target class.
        for class_idx, class_name in enumerate(self.classes_):

            # Select training rows belonging to the current class.
            class_mask = y == class_name
            X_class = X[class_mask]

            # Count how many samples belong to this class.
            self.class_count_[class_idx] = len(X_class)

            # Count each categorical value for every feature.
            for feature_idx in range(self.n_features_):
                values = self.feature_values_[feature_idx]

                for value_idx, value in enumerate(values):
                    # Count how often this value appears in the current class.
                    count = np.sum(X_class[:, feature_idx] == value)
                    # Save the count.
                    self.category_count_[feature_idx][class_idx, value_idx] = count

        # Convert counts into log probabilities.
        self._compute_log_probabilities()

        # Return self to support method chaining.
        return self

    # =====================================================
    # 3.2 COMPUTE CLASS PRIORS AND CONDITIONAL PROBABILITIES
    # =====================================================
    def _compute_log_probabilities(self):

        # Calculate the prior probability of each class.
        #
        # Example:
        # P(Yes) = number_of_Yes_samples / total_samples
        class_prior = self.class_count_ / np.sum(self.class_count_)

        # Store log P(class).
        self.class_log_prior_ = np.log(class_prior)

        # Store log P(feature_value | class) for every feature.
        self.feature_log_prob_ = []

        for feature_idx in range(self.n_features_):
            # Get the frequency matrix for the current feature.
            counts = self.category_count_[feature_idx]

            # Get the number of possible values for this feature.
            n_values = len(self.feature_values_[feature_idx])

            # Apply Laplace smoothing to each category count.
            #
            # Formula:
            # smoothed_count = count + alpha
            smoothed_counts = counts + self.alpha

            # Apply Laplace smoothing to the denominator.
            #
            # Formula:
            # total_for_class + alpha * number_of_categories
            smoothed_totals = (
                    self.class_count_[:, None]
                    + self.alpha * n_values
            )

            # Calculate conditional probabilities:
            # P(feature_value | class)
            probs = smoothed_counts / smoothed_totals

            # Convert probabilities to log probabilities.
            log_probs = np.log(probs)

            # Save log probabilities for this feature.
            self.feature_log_prob_.append(log_probs)

    # =====================================================
    # 3.3 FIND THE INDEX OF A CATEGORICAL VALUE
    # =====================================================
    def _get_value_index(self, feature_idx, value):

        values = self.feature_values_[feature_idx]

        # Find all positions where the feature value matches.
        matches = np.where(values == value)[0]

        # Raise an error when the test value was not seen in training data.
        if len(matches) == 0:
            raise ValueError(
                f"Unknown value {value!r} for feature index {feature_idx}. "
                f"Known values: {values.tolist()}"
            )

        # Return the first matching index.
        return matches[0]

    # =====================================================
    # 3.4 CALCULATE LOG PROBABILITIES FOR EACH CLASS
    # =====================================================
    def predict_log_proba(self, X):
        X = np.asarray(X, dtype=object)

        # If one sample is passed as a 1D array,
        # reshape it into a 2D array with one row.
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Check that test data has the same number of features as training data.
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was trained "
                f"with {self.n_features_} features."
            )

        # Store log scores for all input samples.
        all_log_scores = []

        for sample in X:
            # Start with log P(class).
            log_scores = self.class_log_prior_.copy()

            # Add log P(feature_value | class) for every feature.
            for feature_idx, value in enumerate(sample):
                value_idx = self._get_value_index(feature_idx, value)

                log_scores += self.feature_log_prob_[feature_idx][:, value_idx]

            # Save class scores for the current sample.
            all_log_scores.append(log_scores)

        return np.array(all_log_scores)


    # =====================================================
    # 3.5 CALCULATE NORMALIZED CLASS PROBABILITIES
    # =====================================================
    def predict_proba(self, X):
        log_scores = self.predict_log_proba(X)

        # Shift values before exponentiation for numerical stability.
        shifted = log_scores - np.max(
            log_scores,
            axis=1,
            keepdims=True,
        )

        # Convert shifted log scores back to positive values.
        exp_scores = np.exp(shifted)

        # Normalize scores so that probabilities sum to 1.
        probs = exp_scores / np.sum(
            exp_scores,
            axis=1,
            keepdims=True,
        )

        return probs

    # =====================================================
    # 3.6 PREDICT THE MOST LIKELY CLASS
    # =====================================================
    def predict(self, X):
        log_scores = self.predict_log_proba(X)

        # Get the index of the class with the highest score.
        class_indices = np.argmax(log_scores, axis=1)

        # Convert class indices back to original labels.
        return self.classes_[class_indices]

    # =====================================================
    # 3.7 CALCULATE ACCURACY
    # =====================================================
    def score(self, X, y):

        y = np.asarray(y, dtype=object)
        y_pred = self.predict(X)

        return np.mean(y_pred == y)


# =========================================================
# 4. COMPARE WITH SCIKIT-LEARN CategoricalNB
# =========================================================
def compare_with_sklearn(X_train, y_train, X_test, alpha=1.0):
    try:
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.naive_bayes import CategoricalNB
    except ImportError:
        print(
            "\nscikit-learn is not installed, "
            "so sklearn comparison is skipped."
        )
        return

    print("\n" + "=" * 60)
    print("COMPARE WITH SKLEARN CategoricalNB")
    print("=" * 60)

    # Convert string categories into numeric category IDs.
    encoder = OrdinalEncoder()

    # Learn category mappings from training data and encode it.
    X_train_encoded = encoder.fit_transform(X_train)

    # Encode test data using the same mappings learned from training data.
    X_test_encoded = encoder.transform(X_test)

    # Create and train sklearn's implementation.
    sklearn_model = CategoricalNB(alpha=alpha)
    sklearn_model.fit(X_train_encoded, y_train)

    # Generate predictions and class probabilities.
    sklearn_pred = sklearn_model.predict(X_test_encoded)
    sklearn_proba = sklearn_model.predict_proba(X_test_encoded)

    print("\nsklearn predictions:")
    print(sklearn_pred)

    print("\nsklearn probabilities:")
    print(np.round(sklearn_proba, 4))


# =========================================================
# 5. MAIN PROGRAM
# =========================================================
def main():
    # Create the full dataset.
    data = create_training_data()

    # Split the dataset into input features and target labels.
    #
    # X_train contains:
    # Outlook, Temperature, Humidity, Wind
    X_train = data[:, :-1]

    # y_train contains:
    # Play Tennis label: Yes or No
    y_train = data[:, -1]

    # Define a few unseen samples for testing.
    X_test = np.array([
        ["Sunny", "Cool", "High", "Strong"],
        ["Rain", "Mild", "Normal", "Weak"],
        ["Overcast", "Hot", "High", "Weak"],
    ], dtype=object)

    # Create the model.
    # alpha=1.0 means standard Laplace smoothing.
    model = SimpleCategoricalNB(alpha=1.0)

    # Learn class priors and conditional probabilities from training data.
    model.fit(X_train, y_train)

    # Optional helper function for printing learned probabilities.
    # print_learned_probabilities(model)

    # =====================================================
    # 5.1 PREDICTION
    # =====================================================
    print("\n" + "=" * 60)
    print("PREDICTION")
    print("=" * 60)

    # Predict classes and calculate class probabilities.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Display the result for every test sample.
    for sample, pred, prob in zip(
            X_test,
            predictions,
            probabilities,
    ):
        # Build a readable dictionary:
        # {"No": probability, "Yes": probability}
        prob_dict = {
            class_name: round(float(p), 4)
            for class_name, p in zip(model.classes_, prob)
        }

    print(f"\nX = {sample.tolist()}")
    print(f"Prediction = {pred}")
    print(f"Probabilities = {prob_dict}")

    # =====================================================
    # 5.2 TRAINING ACCURACY
    # =====================================================
    print("\n" + "=" * 60)
    print("TRAINING ACCURACY")
    print("=" * 60)

    # Evaluate the model on its training data.
    print(
        "Accuracy on training data:",
        round(model.score(X_train, y_train), 4),
    )

    # =====================================================
    # 5.3 COMPARE WITH SKLEARN
    # =====================================================
    compare_with_sklearn(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        alpha=1.0,
    )


if __name__ == "__main__":
    main()