import numpy as np
from sklearn.metrics import balanced_accuracy_score
n_features = 2
n_train = 1000
contamination = 0.1
normal_mean = np.array([1,-1])
normal_variance = np.array([2,3])
anomaly_mean = np.array([0,1])
anomaly_variance = np.array([5,4])

# Build dataset by combining normal distributions
X_train_normal = np.column_stack(
    (np.random.normal(loc=normal_mean[0], scale=np.sqrt(normal_variance[0]), size=int(n_train * (1-contamination))),
    np.random.normal(loc=normal_mean[1], scale=np.sqrt(normal_variance[1]), size=int(n_train * (1-contamination))))
)

X_train_anomaly = np.column_stack(
    (np.random.normal(loc=anomaly_mean[0], scale=np.sqrt(anomaly_variance[0]), size=int(n_train * contamination)),
    np.random.normal(loc=anomaly_mean[1], scale=np.sqrt(anomaly_variance[1]), size=int(n_train * contamination)))
)

X_train = np.concatenate([X_train_normal, X_train_anomaly])

# Create labels, 0 = normal, 1 = anomaly
y_train_normal = np.zeros(int(n_train * (1-contamination)))
y_train_anomaly = np.ones(int(n_train * contamination))

y_train = np.concatenate([y_train_normal, y_train_anomaly])

# Shuffle to prevent bias
order = np.arange(n_train)
np.random.shuffle(order)
X_train = X_train[order]
y_train = y_train[order]

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
z_scores = (X_train - mean ) / std

threshold = np.quantile(np.abs(z_scores ), 1 - contamination)

# Predicts for each feature
y_pred_initial = np.where(np.abs(z_scores) > threshold, 1, 0)

# If any feature is anomalous, classify the data point as an anomaly
y_pred = np.any(y_pred_initial, axis=1)
ba = balanced_accuracy_score(y_train, y_pred)
print(f"Balanced accuracy={ba}")