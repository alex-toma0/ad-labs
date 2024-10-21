import numpy as np
from pyod.utils.data import generate_data
from sklearn.metrics import balanced_accuracy_score
X_train,y_train = generate_data(n_features=1, train_only=True, n_train=1000, contamination=0.1)
mean = np.mean(X_train)
std= np.std(X_train)
z_scores = (X_train - mean ) / std
# Contamination rate of 10%
threshold = np.quantile(np.abs(z_scores), 0.9)
# If the zscore exceeds the threshold, classify it as an anomaly
y_pred = np.where(np.abs(z_scores) > threshold, 1, 0)
ba = balanced_accuracy_score(y_train, y_pred)
print(f"Balanced accuracy={ba}")