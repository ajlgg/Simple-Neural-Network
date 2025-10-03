import numpy as np
from linearlayer import Linear
from setup import plot_series


def assert_close(name, got, want, atol=1e-7, rtol=1e-5):
    ok = np.allclose(got, want, atol=atol, rtol=rtol)
    print(f"[{name}] match:", ok, f"(max abs err ~ {np.max(np.abs(got - want)) if got.shape == want.shape else 'shape mismatch'})")
    if not ok:
        print("   shapes:", got.shape, "vs", want.shape)

# Example synthetic credit-like features 
# Features: [utilization, age, late_payments]
Xc = np.array([
    [0.10, 45, 0],
    [0.90, 22, 3],
    [0.40, 31, 1],
    [0.20, 52, 0],
    [0.70, 28, 2],
], dtype=float)

# Target score (higher is better)
yc = np.array([720, 580, 660, 740, 610], dtype=float).reshape(-1,1)

# Normalize features for stability
mu, sigma = Xc.mean(axis=0, keepdims=True), Xc.std(axis=0, keepdims=True) + 1e-8
Xn = (Xc - mu)/sigma

lin = Linear(in_dim=Xn.shape[1], out_dim=1)





lam = 1e-2  # try 0, 1e-2, 1e-1
lr = 0.01
losses = []
for epoch in range(2000):
    yhat = lin.forward(Xn)
    err = (yhat - yc)
    lossval = np.mean((err)**2)
    

    # TODO: Compute the total loss, and append to the losses array
    l2 = np.sum((lin.W)**2) * lam
    loss = l2 + lossval
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