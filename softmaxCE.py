import numpy as np

def softmax(logits):
    # logits: (batch, C)
    z = logits - logits.max(axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / expz.sum(axis=1, keepdims=True)

def cross_entropy(probs, Y_true):
   # probs: (batch, C), softmax outputs
    # Y_true: (batch, C), one-hot encoded
    eps = 1e-9
    # select log of predicted probabilities at the true class positions
    log_likelihood = -np.sum(Y_true * np.log(probs + eps), axis=1)
    return np.mean(log_likelihood)
   
def softmax_cross_entropy_backward(probs, Y_true):
    # d/dlogits of CE with softmax fusion: (probs - Y)/batch
    bs = probs.shape[0]
    return (probs - Y_true) / bs