import math, random
import numpy as np
import matplotlib.pyplot as plt
from setup import plot_series


# 1) Generate data from y = a*x + b + noise
n = 200
true_a, true_b = 2.5, -0.7
x = np.linspace(-2, 2, n)
noise = 0.25 * np.random.randn(n)
y = true_a * x + true_b + noise

# 2) Initialize parameters
w = np.random.randn()
b = np.random.randn()

# 3) Training loop (gradient descent)
lr = 0.05
losses = []
for step in range(800):
    yhat = w * x + b                    # forward
    err = yhat - y                      # residuals
    loss = np.mean(err**2)              # MSE
    # gradients dMSE/dw, dMSE/db
    dw = 2 * np.mean(err * x)
    db = 2 * np.mean(err)
    # update
    w -= lr * dw
    b -= lr * db
    losses.append(loss)

print({"w": w, "b": b, "true_a": true_a, "true_b": true_b})
plot_series(range(len(losses)), losses, title="MSE over steps")