import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.models.dif import DIF
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
def evaluate(model, X, y):
    ba_scores = []
    auc_scores = []
    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
        model.fit(X_train)
        y_test_pred = model.predict(X_test)
        y_test_scores = model.decision_function(X_test)
        ba = balanced_accuracy_score(y_test, y_test_pred)
        auc = roc_auc_score(y_test, y_test_scores)
        ba_scores.append(ba)
        auc_scores.append(auc)
    return np.mean(ba_scores), np.mean(auc_scores)
if __name__ == "__main__":
    data = loadmat("shuttle.mat")
    X = data["X"]
    y = data["y"].ravel()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    iforest = IForest()
    dif = DIF()
    loda = LODA()
    iforest_ba, iforest_auc = evaluate(iforest, X, y)
    dif_ba, dif_auc = evaluate(dif, X, y)
    loda_ba, loda_auc = evaluate(loda, X, y)
    print(f"IForest - mean BA: {iforest_ba}, mean AUC: {iforest_auc}")
    print(f"DIF - mean BA: {dif_ba}, mean AUC: {dif_auc}")
    print(f"LODA - mean BA: {loda_ba}, mean AUC: {loda_auc}")

