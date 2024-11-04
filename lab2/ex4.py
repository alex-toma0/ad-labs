import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.knn import KNN
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization
from sklearn.metrics import balanced_accuracy_score
from scipy.io import loadmat

# Load data from the cardio dataset
data = loadmat("cardio.mat")
X = data["X"]
y = data["y"].ravel()

# Split and normalize data set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

train_scores = []
test_scores = []
for neighbors in range(10, 120, 12):
    knn = KNN(n_neighbors=neighbors)
    knn.fit(X_train)
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)
    train_ba = balanced_accuracy_score(y_train, y_train_pred)
    test_ba = balanced_accuracy_score(y_test, y_test_pred)
    train_scores.append(train_ba)
    test_scores.append(test_ba)

# Normalize scores
train_scores = standardizer(np.array(train_scores).reshape(-1,1))
test_scores = standardizer(np.array(test_scores).reshape(-1,1))

# Final scores
avg_train_score = average(train_scores.T)
max_train_score = maximization(train_scores.T)
avg_test_score = average(test_scores.T)
max_test_score = maximization(test_scores.T)




