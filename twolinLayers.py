import numpy as np
import matplotlib.pyplot as plt
from setup import plot_series
from linearlayer import Linear

n = 200
true_a, true_b = 2.5, -0.7
x = np.linspace(-2, 2, n)
noise = 0.25 * np.random.randn(n)
y = true_a * x + true_b + noise


lin1 = Linear(in_dim=1, out_dim=4)
lin2 = Linear(in_dim=4, out_dim=1)

lr = 0.05
losses = []

# Reuse data from Stage 1
X = x.reshape(-1, 1)  # (n,1)
Y = y.reshape(-1, 1)  # (n,1)

for epoch in range(200):
    # forward
    h = lin1.forward(X)
    yhat = lin2.forward(h)
    # loss
    err = yhat - Y
    loss = np.mean(err**2)
    losses.append(loss)
    # backward (dMSE/dyhat = 2*err/bs)
    bs = X.shape[0]
    dyhat = (2.0/bs) * err
    dh = lin2.backward(dyhat)
    _ = lin1.backward(dh)
    # SGD update
    for layer in (lin1, lin2):
        layer.W -= lr * layer.dW
        layer.b -= lr * layer.db

plot_series(range(len(losses)), losses, title="Two-layer linear: MSE")