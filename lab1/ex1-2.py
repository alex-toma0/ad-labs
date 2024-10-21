import matplotlib.pyplot as plt
import numpy as np
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, roc_curve, auc


# Ex 1
X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1)
X_train_normal = X_train[y_train == 0]
X_train_outlier = X_train[y_train == 1]
plt.figure(1)
plt.scatter(X_train_normal[:,0], X_train_normal[:, 1], c="blue")
plt.scatter(X_train_outlier[:, 0], X_train_outlier[:, 1], c="red")
plt.show()

# Ex 2

clf_name = 'KNN'
clf = KNN(contamination=0.1)
clf.fit(X_train)

y_train_pred = clf.labels_
y_train_scores = clf.decision_scores_
y_test_pred = clf.predict(X_test)
y_test_scores = clf.decision_function(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
ba = (tpr + tnr) / 2
print(f"Balanced accuracy ={ba}")
fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)

plt.figure(2)
plt.plot([0,1], [0,1], "r--")
plt.plot(fpr,tpr, "b-")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title(f"ROC curve, with an area of {auc(fpr,tpr):.2f}")
plt.show()

