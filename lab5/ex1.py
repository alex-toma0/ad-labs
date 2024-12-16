import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
X = np.random.multivariate_normal([5,10,2], [[3,2,2], [2,10,1], [2,1,2]], 500)

# PCA steps
mean_X = np.mean(X, axis=0)
X = X - mean_X

cov_matrix = np.cov(X, rowvar=False)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
desc_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[desc_idx] 
eigenvectors = eigenvectors[:,desc_idx]
expl_var = eigenvalues / np.sum(eigenvalues)
cum_expl_var = np.cumsum(expl_var)

proj_data = X @ eigenvectors
# 3rd component
threshold_third = np.quantile(np.abs(proj_data[:, 2] - np.mean(proj_data[:, 2])), 0.9)
outliers_third = np.abs(proj_data[:, 2] - np.mean(proj_data[:, 2])) > threshold_third


# 2nd component
threshold_second = np.quantile(np.abs(proj_data[:, 1] - np.mean(proj_data[:, 1])), 0.9)
outliers_second = np.abs(proj_data[:, 1] - np.mean(proj_data[:, 1])) > threshold_second

fig = plt.figure(figsize=(8,18))
ax1 = fig.add_subplot(311)
ax1.bar(range(0, len(expl_var)), expl_var, label="Individual variance", align="center")
ax1.step(range(0, len(cum_expl_var)), cum_expl_var, label="Cumulative variance", where="mid")
ax1.legend(loc="best")  

ax2 = fig.add_subplot(312)
ax2.scatter(X[:, 0], X[:, 1], X[:, 2], c=np.where(outliers_third, 'red', 'blue'), label="Outliers Third Component")
ax2.legend()

ax3 = fig.add_subplot(313)
ax3.scatter(X[:, 0], X[:, 1], X[:, 2], c=np.where(outliers_second, 'red', 'blue'), label="Outliers Second Component")
ax3.legend()

plt.show()



