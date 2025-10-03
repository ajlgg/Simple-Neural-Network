import numpy as np
from linearlayer import Linear
from nonlinmlp import relu, relu_backward, tanh, tanh_backward


class MLP:
    def __init__(self, in_dim, hidden_dim, out_dim, nonlin="relu"):
        self.l1 = Linear(in_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        self.h_pre = None  # pre-activation cache

    def forward(self, x):
        z1 = self.l1.forward(x)
        self.h_pre = z1
        if self.nonlin == "relu":
            h = relu(z1)            
        else:
            h = tanh(z1)           
        yhat = self.l2.forward(h)
        return yhat, h

    def backward(self, dyhat, h):
        dh = self.l2.backward(dyhat)
        if self.nonlin == "relu":
            dz1 = relu_backward(self.h_pre, dh) 
        else:
            dz1 = tanh_backward(self.h_pre, dh) 
        dx = self.l1.backward(dz1)
        return dx