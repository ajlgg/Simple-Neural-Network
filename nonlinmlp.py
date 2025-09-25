import numpy as np
from linearlayer import Linear


def relu(x):
    return np.maximum(0, x)

def relu_backward(x, dout):
    result = []
    for arr in x:
        if arr < 0:
            result.append(0)
        else:
            result.append(1)
    return dout * result

def tanh(x):
    return np.tanh(x)

def tanh_backward(x, dout):
    return dout * (1 - np.tanh(x)**2)
