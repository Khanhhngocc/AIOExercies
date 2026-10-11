# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# KMeans clustering algorithm.
from sklearn.cluster import KMeans

# Encode categorical values and scale numerical features.
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Evaluate clustering quality.
from sklearn.metrics import silhouette_score


# =========================================================
# 2. LOAD AND INSPECT THE DATASET
# =========================================================
# Load the Mall Customers dataset from a CSV file.
df = pd.read_csv("./Mall_Customers.csv")

# Remove CustomerID because it is only an identifier.
# It does not describe customer behavior and should not affect clusters.
df.drop("CustomerID", axis=1, inplace=True)

# Display column names, data types, and missing-value information.
df.info()


# =========================================================
# 3. VISUALIZE NUMERICAL FEATURE DISTRIBUTIONS
# =========================================================
def plot_all_histograms(df, title_prefix=""):
    """
    Plot a histogram for every numerical column in the DataFrame.

    Parameters:
    - df: Input DataFrame.
    - title_prefix: Optional text displayed before each chart title.
    """

    # Select only numerical columns for histogram visualization.
    num_cols = df.select_dtypes(include=[np.number]).columns

    # Display at most three histograms per row.
    n_cols = 3

    # Calculate the number of rows needed for all numerical columns.
    n_rows = math.ceil(len(num_cols) / n_cols)

    # Create one figure containing multiple subplots.
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))

    # Create one histogram for each numerical column.
    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)

        # Draw the distribution and its estimated density curve.
        sns.histplot(
            df[col],
            kde=True,
            bins=30,
        )

        plt.title(f"{title_prefix}{col}")
        plt.xlabel("")
        plt.ylabel("")

    # Adjust subplot spacing automatically.
    plt.tight_layout()
    plt.show()


# Visualize the distributions before feature scaling.
plot_all_histograms(df)


# =========================================================
# 4. ENCODE CATEGORICAL FEATURES
# =========================================================
# KMeans requires numerical input, so encode the Genre column:
# Female / Male -> numerical category IDs.
label_encoder = LabelEncoder()

df["Genre"] = label_encoder.fit_transform(df["Genre"])


# =========================================================
# 5. SCALE ALL FEATURES
# =========================================================
# KMeans is distance-based.
# MinMaxScaler transforms every feature into the range [0, 1],
# preventing a large-scale feature from dominating the clustering result.
scaler = MinMaxScaler()

# Fit the scaler and transform all feature columns.
df_scaled = scaler.fit_transform(df)


# =========================================================
# 6. TEST DIFFERENT NUMBERS OF CLUSTERS
# =========================================================
# Store the inertia and silhouette score for each K value.
inertia = []
silhouette_scores = []

# Test cluster counts from 2 to 10.
# Silhouette score cannot be calculated when K = 1.
k_values = range(2, 11)

for k in k_values:
    # Create a KMeans model with k clusters.
    # random_state=42 makes results reproducible.
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    # Fit KMeans on the scaled customer data.
    kmeans.fit(df_scaled)

    # Inertia is the sum of squared distances from samples
    # to their assigned cluster centroids.
    inertia.append(kmeans.inertia_)

    # Silhouette score measures how well samples fit their own cluster
    # compared with other clusters. Higher is better.
    silhouette_scores.append(
        silhouette_score(df_scaled, kmeans.labels_)
    )


# =========================================================
# 7. VISUALIZE THE ELBOW METHOD
# =========================================================
plt.plot(
    k_values,
    inertia,
    marker="o",
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(alpha=0.3)
plt.show()


# =========================================================
# 8. VISUALIZE SILHOUETTE SCORES
# =========================================================
plt.plot(
    k_values,
    silhouette_scores,
    marker="o",
    color="green",
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Each K Value")
plt.grid(alpha=0.3)
plt.show()


# =========================================================
# 9. SELECT THE BEST NUMBER OF CLUSTERS
# =========================================================
# Find the K value with the highest silhouette score.
optimal_k = list(k_values)[np.argmax(silhouette_scores)]

print(f"Optimal number of clusters based on Silhouette Score: {optimal_k}")
print(f"Highest Silhouette Score: {max(silhouette_scores):.3f}")


# =========================================================
# 10. TRAIN THE FINAL KMEANS MODEL
# =========================================================
# Train the final model using the selected number of clusters.
kmeans_final = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10,
)

# Fit the model and return a cluster label for each customer.
cluster_labels = kmeans_final.fit_predict(df_scaled)

# Calculate the final silhouette score.
final_silhouette_score = silhouette_score(
    df_scaled,
    cluster_labels,
)

print(f"Final Silhouette Score: {final_silhouette_score:.3f}")