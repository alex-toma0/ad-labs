import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD

def plot_data(y_train_pred, y_test_pred):

    fig = plt.figure(figsize=(14,10))
    ax1 = fig.add_subplot(221, projection="3d")
    ax1.set_title("Training data (Ground truth)")
    ax1.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], X_train[y_train == 0][:, 2],
                c="blue")
    ax1.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], X_train[y_train == 1][:, 2],
                c="red")
    
    ax2 = fig.add_subplot(222, projection="3d")
    ax2.set_title("Training data (Predicted)")
    ax2.scatter(X_train[y_train_pred == 0][:, 0], X_train[y_train_pred == 0][:, 1], X_train[y_train_pred == 0][:, 2],
                c="blue")
    ax2.scatter(X_train[y_train_pred == 1][:, 0], X_train[y_train_pred == 1][:, 1], X_train[y_train_pred == 1][:, 2],
                c="red")
    
    ax3 = fig.add_subplot(223, projection="3d")
    ax3.set_title("Test data (Ground truth)")
    ax3.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], X_test[y_test == 0][:, 2],
                c="blue")
    ax3.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], X_test[y_test == 1][:, 2],
                c="red")
    
    ax4 = fig.add_subplot(224, projection="3d")
    ax4.set_title("Test data (Predicted)")
    ax4.scatter(X_test[y_test_pred == 0][:, 0], X_test[y_test_pred == 0][:, 1], X_test[y_test_pred == 0][:, 2],
                c="blue")
    ax4.scatter(X_test[y_test_pred == 1][:, 0], X_test[y_test_pred == 1][:, 1], X_test[y_test_pred == 1][:, 2],
                c="red")
    
    plt.tight_layout()
    plt.show()
    
    
X_train, X_test, y_train, y_test = generate_data(n_features=3, n_train=300, n_test=200, contamination=0.15, random_state=40)
ocsvm = OCSVM(kernel="linear", contamination=0.15)
ocsvm.fit(X_train)
y_test_pred_ocsvm = ocsvm.predict(X_test)
y_test_scores_ocsvm = ocsvm.decision_function(X_test)
y_train_pred_ocsvm = ocsvm.labels_
ba_ocsvm = balanced_accuracy_score(y_test, y_test_pred_ocsvm)
roc_auc_ocsvm = roc_auc_score(y_test, y_test_scores_ocsvm)

dsvdd = DeepSVDD(contamination = 0.15, n_features=3)
dsvdd.fit(X_train)
y_train_pred_dsvdd = dsvdd.labels_
y_test_pred_dsvdd = dsvdd.predict(X_test)
y_test_scores_dsvdd = dsvdd.decision_function(X_test)
ba_dsvdd = balanced_accuracy_score(y_test, y_test_pred_dsvdd)
roc_auc_dsvdd = roc_auc_score(y_test, y_test_scores_dsvdd)
print(f"OCSVM: Balanced accuracy: {ba_ocsvm}\nROC AUC: {roc_auc_ocsvm}")
print(f"Deep SVDD: Balanced accuracy: {ba_dsvdd}\nROC AUC: {roc_auc_dsvdd}")
plot_data(y_train_pred_ocsvm, y_test_pred_ocsvm)
plot_data(y_train_pred_dsvdd, y_test_pred_dsvdd)
