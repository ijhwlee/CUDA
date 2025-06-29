import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Generate synthetic data
np.random.seed(0)
n_samples = 300

# Create three different clusters
data1 = np.random.normal(loc=0, scale=1.0, size=(n_samples, 2))
data2 = np.random.normal(loc=5, scale=1.5, size=(n_samples, 2))
data3 = np.random.normal(loc=10, scale=0.5, size=(n_samples, 2))

# Combine the data
X = np.vstack((data1, data2, data3))

# Fit a Gaussian Mixture Model
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=0)
gmm.fit(X)

# Predict cluster labels
labels = gmm.predict(X)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=10)
plt.title("Gaussian Mixture Model Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.savefig("gmm.png")
# plt.show()
