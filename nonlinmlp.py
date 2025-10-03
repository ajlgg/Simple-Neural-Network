import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_backward(x, dout):
    # Gradient of ReLU is 1 for x > 0, else 0
    dx = dout * (x > 0).astype(float)
    return dx

def tanh(x):
    return np.tanh(x)

def tanh_backward(x, dout):
    return dout * (1 - np.tanh(x) ** 2)