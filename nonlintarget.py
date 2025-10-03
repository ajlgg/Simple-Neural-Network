import numpy as np
import math
from MLP1Lay import MLP
from setup import plot_series

# Nonlinear data: y = sin(2*pi*x) + noise
n = 400
x2 = np.linspace(-1, 1, n)
y2 = np.sin(2*math.pi*x2) + 0.1*np.random.randn(n)
X2 = x2.reshape(-1,1)
Y2 = y2.reshape(-1,1)

model = MLP(in_dim=1, hidden_dim=32, out_dim=1, nonlin="relu")
lr = 0.05
losses = []

for epoch in range(1500):
    yhat, h = model.forward(X2)
    err = yhat - Y2
    loss = np.mean(err**2)
    losses.append(loss)
    # backward
    dyhat = (2.0/X2.shape[0]) * err
    _ = model.backward(dyhat, h)
    # update
    for layer in (model.l1, model.l2):
        layer.W -= lr * layer.dW
        layer.b -= lr * layer.db

plot_series(range(len(losses)), losses, title="Nonlinear MLP: MSE")