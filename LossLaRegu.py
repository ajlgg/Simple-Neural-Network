import numpy as np
from setup import plot_series
import random
from linearlayer import Linear

np.random.seed(3)
lin = Linear(in_dim=3, out_dim=1)
Xn = np.random.randn(5,3)
yc = np.random.randn(5,1)

# forward
yhat = lin.forward(Xn)
err = yhat - yc
bs = Xn.shape[0]
dy = (2.0/bs) * err

# grads without reg
_ = lin.backward(dy)
dW_no_reg = lin.dW.copy()

# grads with reg
lam = 0.01
_ = lin.backward(dy)            # recompute base grads
lin.dW += 2*lam*lin.W
dW_with_reg = lin.dW

assert_close("L2 grad delta", dW_with_reg - dW_no_reg, 2*lam*lin.W)


lam = 1e-2  # try 0, 1e-2, 1e-1
lr = 0.01
losses = []
for epoch in range(1000):
    yhat = lin.forward(Xn)
    err = np.mean(yhat - yc)**2
    

    # TODO: Compute the total loss, and append to the losses array
    l2 = np.sum(lin.w)**2 * lam
    loss = l2 + err
    losses.append(loss)

    # backward
    bs = Xn.shape[0]
    dy = (2.0/bs) * err
    _ = lin.backward(dy)
    
    # TODO: Keep a running total of lin.dW (add to it) with the gradient from L2: d/dW lam||W||^2 = 2*lam*W
    lin.dW += 2 * lam * lin.W

    
    # update
    lin.W -= lr * lin.dW
    lin.b -= lr * lin.db

plot_series(range(len(losses)), losses, title="Credit score with L2")

print("weights (on normalized features):\n", lin.W)
print("bias:", lin.b)