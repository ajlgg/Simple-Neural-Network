import numpy as np
from sklearn.datasets import fetch_openml


mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist['data'].astype(np.float32) / 255.0
y = mnist['target'].astype(np.int64)

# train/val split (small subset to speed up)
N = 20000
X, y = X[:N], y[:N]
num_classes = 10

# one-hot helper
def one_hot(y, C):
    oh = np.zeros((y.shape[0], C), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh
Y = one_hot(y, num_classes)