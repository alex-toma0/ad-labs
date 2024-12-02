import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD

data = loadmat("shuttle.mat")
X = data["X"]
y = data["y"].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, train_size=0.5)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

ocsvm = OCSVM()
ocsvm.fit(X_train)
y_test_pred_ocsvm = ocsvm.predict(X_test)
y_test_scores_ocsvm = ocsvm.decision_function(X_test)
ba_ocsvm = balanced_accuracy_score(y_test, y_test_pred_ocsvm)
roc_auc_ocsvm = roc_auc_score(y_test, y_test_scores_ocsvm)
print(f"OCSVM:\nBalanced accuracy: {ba_ocsvm}\nROC AUC: {roc_auc_ocsvm}")

neuron_config=[[64,32],
         [128,64,32],
         [256,128,64,32]]

for index, config in enumerate(neuron_config): 
    print(f"Architecture {index + 1}: ")
    dsvdd = DeepSVDD(n_features=9, hidden_neurons=config, epochs=40)
    dsvdd.fit(X_train)
    y_test_pred = dsvdd.predict(X_test)
    y_test_scores = dsvdd.decision_function(X_test)
    ba = balanced_accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_scores)
    print(f"Balanced accuracy: {ba}\nROC AUC:{roc_auc}")