from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
X_train, X_test, y_train, y_test = generate_data_clusters(n_features=2, n_train=400, n_test=200, n_clusters = 2,contamination=0.1)
knn = KNN(contamination=0.1, n_neighbors=5)
knn.fit(X_train)
y_train_pred = knn.labels_
y_test_pred = knn.predict(X_test)

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(221)
ax1.scatter(X_train[:,0], X_train[:,1], c=y_train)
ax1.set_title("Ground truth labels for training data")

ax2 = fig.add_subplot(222)
ax2.scatter(X_train[:,0], X_train[:,1], c=y_train_pred)
ax2.set_title("Predicted labels for training data")

ax3 = fig.add_subplot(223)
ax3.scatter(X_test[:,0], X_test[:,1], c=y_test)
ax3.set_title("Ground truth labels for test data")

ax4 = fig.add_subplot(224)
ax4.scatter(X_test[:,0], X_test[:,1], c=y_test_pred)
ax4.set_title("Predicted labels for test data")

print(f"Balanced accuracy score = {balanced_accuracy_score(y_train, y_train_pred)}")
plt.show()