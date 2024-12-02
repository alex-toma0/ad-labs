import numpy as np
from scipy.io import loadmat
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
data =  loadmat("cardio.mat")
X = data["X"]
y = data["y"].ravel()
# Convert pyod anomaly labels to sklearn format
y = 2 * y - 1

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.4, test_size=0.6)

pipeline=Pipeline([
    ("scaler", StandardScaler()),
    ("ocsvm", OneClassSVM())
])

params = {
    "ocsvm__kernel": ["linear", "poly", "rbf", "sigmoid"],
    "ocsvm__gamma": ["scale", "auto", 0.001,0.005, 0.01, 0.05, 0.1],
    "ocsvm__nu": [0.05, 0.1, 0.2, 0.3,0.4,0.5]
}

grid_search = GridSearchCV(pipeline, params, scoring="balanced_accuracy",cv=5)
grid_search.fit(X_train, y_train)

print(f"Best params found by grid search: {grid_search.best_params_}")
print(f"Best balanced accuracy score found by grid search: {grid_search.best_score_}")

best_ocsvm = grid_search.best_estimator_
y_pred = best_ocsvm.predict(X_test)
ba = balanced_accuracy_score(y_test, y_pred)

print(f"Best model's Balanced accuracy: {ba}")


