import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
# Training loda 
X, y = make_blobs(n_samples=500, n_features=2, centers= [(0,0)], cluster_std=1.0)
proj_vec = []
for _ in range(5):
    # Unit vectors
    vector = np.random.multivariate_normal(mean=[0,0], cov=np.eye(2))
    vector = vector / np.linalg.norm(vector)
    proj_vec.append(vector)

projections = [X.dot(vec) for vec in proj_vec]
projections = np.array(projections)
scores = np.zeros(500)
histograms = []
bin_edges = []
for proj in projections:
    hist, edges = np.histogram(proj,bins=10, range=(proj.min() - 1, proj.max() + 1), density=True)
    bin_probs = hist * np.diff(edges)
    inds = np.digitize(proj, edges) - 1
    probs = bin_probs[inds]
    scores += probs
    histograms.append(hist)
    bin_edges.append(edges)
scores = scores / 5
histograms = np.array(histograms)
bin_edges = np.array(bin_edges)

# Testing 
X_test = np.random.uniform(-3, 3, (500,2))
test_projections = [X_test.dot(vec) for vec in proj_vec]
test_scores = np.zeros(500)
# Using trained histograms to predict anomaly scores
for i,test_proj in enumerate(test_projections):
    bin_probs = histograms[i]
    edges = bin_edges[i]
    test_inds = np.digitize(test_proj, edges) - 1
    test_inds = np.clip(test_inds, 0, len(bin_probs) - 1)
    test_probs = bin_probs[test_inds]
    test_scores += test_probs 
test_scores = test_scores / 5
print(test_scores.shape)
# Plot test points with predicted anomaly scores as color map
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_scores)
plt.colorbar(label="Predicted anomaly score")
plt.show()

    