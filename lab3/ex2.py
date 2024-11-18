import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
# Generating data
def ex2(dimensions="2d"):
    if dimensions=="2d":
        X, y = make_blobs(n_samples=500, n_features=2, centers=[(10,0), (0,10)], cluster_std=1.0)
        X_test = np.random.uniform(-10, 20, (1000, 2))
    elif dimensions=="3d":
        X, y = make_blobs(n_samples=500, n_features=3, centers=[(0,10,0), (10,0,10)], cluster_std=1.0)
        X_test = np.random.uniform(-10, 20, (1000, 3))
    else:
        return
    # Fitting the models
    clf_iforest = IForest(contamination=0.02)
    clf_deepforest = DIF(contamination=0.02)
    clf_loda = LODA(contamination=0.02)
    clf_iforest.fit(X)
    clf_deepforest.fit(X)
    clf_loda.fit(X)
    # Calculate and plot anomaly scores
    iforest_scores = clf_iforest.decision_function(X_test)
    deepforest_scores = clf_deepforest.decision_function(X_test)
    loda_scores = clf_loda.decision_function(X_test)

    if dimensions == "2d":
        fig, ax = plt.subplots(3, figsize=(8,10))
        sc = ax[0].scatter(X_test[:, 0], X_test[:, 1], c=iforest_scores)
        ax[0].set_title("Isolation Forest")
        ax[1].scatter(X_test[:, 0], X_test[:, 1], c=deepforest_scores)
        ax[1].set_title("Deep Isolation Forest")
        ax[2].scatter(X_test[:, 0], X_test[:, 1], c=loda_scores)
        ax[2].set_title("LODA")
        fig.colorbar(sc,ax=ax,label="Anomaly score")
        plt.show()
    elif dimensions == "3d":
        fig = plt.figure(figsize=(14,10))
        ax1 = fig.add_subplot(131, projection="3d")
        sc = ax1.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=iforest_scores)
        ax1.set_title("Isolation Forest")
        ax2 = fig.add_subplot(132, projection="3d")
        ax2.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=deepforest_scores)
        ax2.set_title("Deep Isolation Forest")
        ax3 = fig.add_subplot(133, projection="3d")
        ax3.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=loda_scores)
        ax3.set_title("LODA")
        fig.colorbar(sc, ax=[ax1,ax2,ax3], label="Anomaly score")
        plt.show()

if __name__ == "__main__":
    ex2(dimensions="2d")
    #ex2(dimensions="3d")