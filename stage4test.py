import numpy as np
from linearlayer import Linear


def assert_close(name, got, want, atol=1e-7, rtol=1e-5):
    ok = np.allclose(got, want, atol=atol, rtol=rtol)
    print(f"[{name}] match:", ok, f"(max abs err ~ {np.max(np.abs(got - want)) if got.shape == want.shape else 'shape mismatch'})")
    if not ok:
        print("   shapes:", got.shape, "vs", want.shape)

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