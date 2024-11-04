import numpy as np
import matplotlib.pyplot as plt
a = np.random.normal(0,10)
b = np.random.normal(0,10)
c = np.random.normal(0,10)
means = [0,1,2,3]
variances=[0.5,1,2,3]

def generate_data_1D(mean, variance, size, type="regular"):
    std_dev = np.sqrt(variance)
    noise = np.random.normal(mean, std_dev, size)
    x1 = np.random.normal(0,10,size)
    match type:
        case "high_x":
            x1 += np.random.normal(0,4 * std_dev, size)
            
        case "high_y":
            noise += np.random.normal(mean, 4 * std_dev, size)
            
        case "high_both":
            x1 += np.random.normal(0,4 * std_dev, size)
            noise += np.random.normal(mean, 4 * std_dev, size)
            
    y = a * x1 + b + noise
    return x1, y

def generate_data_2D(mean, variance, size, type="regular"):
    std_dev = np.sqrt(variance)
    noise = np.random.normal(mean, std_dev, size)
    x1 = np.random.normal(0,10,size)
    x2 = np.random.normal(0,10,size)
    match type:
        case "high_x":
            x1 += np.random.normal(0, 4 * std_dev, size)
            x2 += np.random.normal(0, 4 * std_dev, size)
        case "high_y":
            noise += np.random.normal(0, 4 * std_dev, size)
        case "high_both":
            x1 += np.random.normal(0, 4 * std_dev, size)
            x2 += np.random.normal(0, 4 * std_dev, size)
            noise += np.random.normal(0, 4 * std_dev, size)
    y = a * x1 + b * x2 + c + noise
    return x1,x2,y

def calc_leverage(X):
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    return np.diag(H)

def main():
    fig, axs = plt.subplots(1,len(variances), figsize=(15,5))
    # 1D
    for i, (mean, variance) in enumerate(zip(means, variances)):
        x_reg,y_reg = generate_data_1D(mean, variance, 100)
        x_highx, y_highx = generate_data_1D(mean,variance,100,"high_x")
        x_highy, y_highy = generate_data_1D(mean,variance,100,"high_y")
        x_both, y_both = generate_data_1D(mean,variance,100,"high_both")
        x = np.concatenate([x_reg,x_highx, x_highy, x_both])
        y = np.concatenate([y_reg,y_highx, y_highy, y_both])
        X = np.vstack([x, np.ones_like(x)]).T
        leverage_scores = calc_leverage(X)
    
        axs[i].set_title(f"$\\sigma^2$ = {variance}")
        axs[i].set_xlabel("x1")
        axs[i].set_ylabel("y")
        axs[i].scatter(x,y, c=leverage_scores, cmap="viridis")

        # Mark highest leverage score points
        high_indices = np.argsort(leverage_scores)[-5:]
        axs[i].scatter(x[high_indices], y[high_indices], c="red")

    # 2D 
    fig, axs = plt.subplots(1,len(variances), figsize=(15,5), constrained_layout=True)
    for i, (mean, variance) in enumerate(zip(means, variances)):
        x1_reg, x2_reg, y_reg = generate_data_2D(mean, variance, 100)
        x1_highx, x2_highx, y_highx = generate_data_2D(mean, variance, 100, "high_x")
        x1_highy, x2_highy, y_highy = generate_data_2D(mean, variance, 100, "high_y")
        x1_both, x2_both, y_both = generate_data_2D(mean, variance, 100, "high_both")
        
        x1 = np.concatenate([x1_reg, x1_highx, x1_highy, x1_both])
        x2 = np.concatenate([x2_reg, x2_highx, x2_highy, x2_both])
        y_2D = np.concatenate([y_reg, y_highx, y_highy, y_both])
        X_2D = np.vstack([x1, x2, np.ones_like(x1)]).T
        leverage_scores_2D = calc_leverage(X_2D)

        axs[i].set_title(f"2D case, $\\sigma^2$ = {variance}")
        axs[i].set_xlabel("x1")
        axs[i].set_ylabel("x2")
        sc = axs[i].scatter(x1,x2, c=y_2D, cmap="viridis")
        cbar = plt.colorbar(sc)

        # Mark highest leverage score points
        high_indices_2D = np.argsort(leverage_scores_2D)[-5:]
        axs[i].scatter(x1[high_indices_2D], x2[high_indices_2D], c="red")
        

        
    plt.show() 

if __name__ == "__main__":
    main()

