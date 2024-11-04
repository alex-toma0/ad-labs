import numpy as np
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=[200,100], n_features=2, centers=[(-10,10), (10,10)], cluster_std=[2,6])
knn = KNN(n_neighbors=20, contamination=0.07)
knn.fit(X)
knn_y_pred = knn.labels_

lof = LOF(n_neighbors=20, contamination=0.07)
lof.fit(X)
lof_y_pred = lof.labels_

fig, axs = plt.subplots(1,2, figsize=(7,5))
knn_outliers = X[knn_y_pred == 1]
knn_inliers = X[knn_y_pred == 0]

lof_outliers = X[lof_y_pred == 1]
lof_inliers = X[lof_y_pred == 0]

axs[0].scatter(knn_inliers[:, 0], knn_inliers[:, 1], color="blue")
axs[0].scatter(knn_outliers[:, 0], knn_outliers[:, 1], color="red")
axs[0].set_title("KNN")

axs[1].scatter(lof_inliers[:, 0], lof_inliers[:, 1], color="blue")
axs[1].scatter(lof_outliers[:, 0], lof_outliers[:, 1], color="red")
axs[1].set_title("LOF")

plt.show()
