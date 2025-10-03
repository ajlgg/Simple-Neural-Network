import numpy as np
from softmaxCE import softmax, cross_entropy, softmax_cross_entropy_backward
from setup import plot_series
from dataloading import X, num_classes, Y
from MLP1Lay import MLP


D = X.shape[1]     # e.g. 784 for MNIST
H = 64             # you can try other sizes: 32,128 etc.
C = num_classes    # 10 usually for the number of digits being classified
EPOCHS = 500

model = MLP(in_dim=D, hidden_dim=H, out_dim=C, nonlin="relu")

lr = 0.01
batch = 128
losses_mlp = []
accs_mlp = []

for epoch in range(EPOCHS):
    perm = np.random.permutation(X.shape[0]) # shuffle the data
    X_shuf, Y_shuf = X[perm], Y[perm]
    for i in range(0, X.shape[0], batch):
        Xb = X_shuf[i:i+batch]
        Yb = Y_shuf[i:i+batch]

        # Forward through model
        logits, h = model.forward(Xb)   # logits shape: (batch, C)
        probs = softmax(logits)

        # Loss
        loss = cross_entropy(probs, Yb)
        losses_mlp.append(loss)

        # Backward
        dyhat = softmax_cross_entropy_backward(probs, Yb)
        _ = model.backward(dyhat, h)

        # Update parameters
        for layer in (model.l1, model.l2):
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

    # Optional: evaluate accuracy on a validation / subset
    probs_all, _ = model.forward(X[:2000])
    pred = probs_all.argmax(axis=1)
    acc = (pred == Y[:2000].argmax(axis=1)).mean()
    accs_mlp.append(acc)
    print(f"(MLP) epoch {epoch} acc ~ {acc:.3f}")

# Plotting
plot_series(range(len(losses_mlp)), losses_mlp, title="MNIST MLP: loss over steps")
plot_series(range(len(accs_mlp)), accs_mlp, title="MNIST MLP: running accuracy (subset)")