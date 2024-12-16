import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA

data = loadmat("shuttle.mat")
X = data["X"]
y = data["y"].ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, train_size=0.6)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pca = PCA()
pca.fit(X_train)

expl_var = pca.explained_variance_
cum_expl_var = np.cumsum(expl_var)

plt.bar(range(0, len(expl_var)), expl_var, label="Individual variance", align="center")
plt.step(range(0, len(cum_expl_var)), cum_expl_var, label="Cumulative variance", where="mid")

pca.decision_function
y_train_pred = pca.labels_
y_test_pred = pca.predict(X_test)
ba_pca_train = balanced_accuracy_score(y_train, y_train_pred)
ba_pca_test = balanced_accuracy_score(y_test, y_test_pred)

print(f"PCA Train ba: {ba_pca_train}")
print(f"PCA Test ba: {ba_pca_test}")

kpca = KPCA()
kpca.fit(X_train)

kpca_train_pred = kpca.labels_
kpca_test_pred = kpca.predict(X_test)

ba_kpca_train = balanced_accuracy_score(y_train, kpca_train_pred)
ba_kpca_test = balanced_accuracy_score(y_test, kpca_test_pred)

print(f"KPCA Train ba: {ba_kpca_train}")
print(f"KPCA Test ba: {ba_kpca_test}")
plt.show()