import numpy as np
import math


class Linear:
    def __init__(self, in_dim, out_dim):
        # Xavier/Glorot for linear-only is fine
        limit = math.sqrt(6/(in_dim+out_dim))
        self.W = np.random.uniform(-limit, limit, size=(out_dim, in_dim))
        self.b = np.zeros((out_dim,))
        # grads
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # cache
        self.x = None

    def forward(self, x):
        """x: (batch, in_dim) -> out: (batch, out_dim)"""
        # make sure x is a numpy array with shape (batch, in_dim)
        self.x = x
        # TODO: return x @ W^T + b 
        out = x @ self.W.T + self.b
        return out

    def backward(self, dout):
        """dout: (batch, out_dim) -> returns dx: (batch, in_dim)
        Fills self.dW, self.db."""
        # TODO: compute grads wrt W, b, and return dx
        self.dw = dout.T @ self.x
        self.db = np.sum(dout, axis=0)
        dx = dout @ self.W
        return dx