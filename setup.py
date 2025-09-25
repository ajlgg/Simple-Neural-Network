import math, random
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
random.seed(42)

def plot_series(xs, ys, title=""):
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("step/epoch")
    plt.ylabel("value")
    if title:
        plt.title(title)
    plt.show()
    
# Check that two matrices are approximately equal in value, with some tolerance for rounding error    pip
def assert_close(name, got, want, atol=1e-7, rtol=1e-5):
    ok = np.allclose(got, want, atol=atol, rtol=rtol)
    print(f"[{name}] match:", ok, f"(max abs err ~ {np.max(np.abs(got - want)) if got.shape == want.shape else 'shape mismatch'})")
    if not ok:
        print("   shapes:", got.shape, "vs", want.shape)