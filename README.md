# Simple-Neural-Network
A lab assignment for CS357 Foundations of Artificial Intelligence, building and training small neural networks from scratch, including linear estimators, MLPs, and a tiny MNIST model.

Creating a Simple Neural Network

assignment goals
The goals of this assignment are:
1. Build intuition for forward pass, loss, gradients, and backpropagation by implementing tiny networks from scratch.
2. Progress from a fully‑worked linear estimator to partially scaffolded implementations (multilayer, nonlinear, credit score weights, MNIST).
3. Explain each coding step in your own words and validate results with small tests/plots.


STAGE 0 - Setup (Shared Utilities)
Before each stage, you may want simple helpers for reproducibility and plotting.

STAGE 1 - Linear Function Estimator
Fit a line y = mx+b to noisy data. You will not implement anything—just run, read, and interpret it.

Concepts: forward pass, mean squared error (MSE), gradients, update rule.

STAGE 2 - Linear, Multi-Layer
We now stack linear layers without nonlinearities. A depth-2 linear network is still linear end-to-end—but this is great practice for shapes and chaining.

x @ self.W.T multiplies each input row by the columns of W (so the output has one column per output unit). The @ operator performs matrix multiplication in Python. If your version of Python does not support the @ operator, you can substitute it with np.matmul like this: np.matmul(x, self.W.T).
+ self.b adds the bias to every row.
backward Function
What to do: (inside def backward(self, dout):):
This computes gradients for weights and bias, and the gradient to pass downstream.

Set self.dW to dout.T matrix multiplied by self.x
Set self.db to dout.sum(axis=0) to add these elements together
Set dx to dout matrix multiplied by self.W and return dx
Why:

self.dW asks: “if I nudge a weight, how much does the loss change?” For matrices, the rule gives dout.T @ X.
self.db is the sum across the batch because b is added to every row.
dx tells the previous layer how the loss changes if inputs change: multiply by W.

We will create a two-layer linear model: x -> Linear1 -> Linear2 -> yhat. No activation yet. Just run this code.

Define two layers
lin1 = Linear(in_dim=1, out_dim=4) expands a single input feature into a 4-dimensional hidden representation.
lin2 = Linear(in_dim=4, out_dim=1) collapses that hidden vector back to a single prediction.
Forward pass
First call h = lin1.forward(X) to compute hidden features.
Then yhat = lin2.forward(h) to produce predictions.
This illustrates how modules are chained together to form deeper networks.
Loss calculation
Compute residuals err = yhat - Y and the mean squared error (MSE).
This is the same loss as Stage 1, but now passed through two layers.
Backward pass
Start with gradient of MSE wrt predictions: dyhat = (2.0/bs) * err.
Push it through lin2.backward(dyhat) and then lin1.backward(dh).
This shows how the chain rule is implemented in code: each layer returns gradients for the one before it.
Parameter update
After gradients are computed, each layer’s weights and biases are updated with SGD:
layer.W -= lr * layer.dW, layer.b -= lr * layer.db.
Summary
Even with two layers, the composition is still just a single linear function overall (because no nonlinearity is added yet). This stage is practice for chaining layers and verifying backprop mechanics before introducing nonlinear activations in Stage 3.
Checkpoints:

If loss steadily decreases, your Linear layer is working.
Verify lin1.backward and lin2.backward by finite differences on a tiny batch (e.g., 3 points). Briefly describe the result.


Why does stacking linear layers still represent a linear map overall?



STAGE 3 - Nonlinear MLP (Implement activations + their grads)

Introduce /sigma to gain expressivity. We will add ReLU by default and let you try tanh.

In the relu function, use the np.maximum(0, x) function to which returns max(a, b) for every element of x.
In the relu_backward function, we want to pass through the values corresponding to z elements that were not trimmed by the ReLU function. In other words, those values of dout that correspond to positive values of z. For each element of x, append an element to result (initially an empty array): that value should be 0 if the current element of x is less than 0, and should be the current element of dout (the gradient value for that element) otherwise. This is our ReLU gradient, and you should return that result array.
Fill in the formula for tanh_backward and return that function.


Forward pass (MLP.forward)
Why:

z1 = self.l1.forward(x)
Pass input x through the first linear layer.
Computes z1 = X W1^T + b1.
Shape: (B, in_dim) @ (in_dim, hidden_dim) = (B, hidden_dim).
self.h_pre = z1
Cache the raw pre-activation values for use in backprop.
Needed because ReLU/tanh derivatives depend on the input to the nonlinearity.
if self.nonlin == "relu":
    h = relu(z1)
else:
    h = tanh(z1)
Apply the chosen nonlinearity elementwise.
ReLU: pass positives, zero-out negatives.
Tanh: squashes values smoothly into [-1, 1].
yhat = self.l2.forward(h)
Pass hidden activations through the second linear layer.
Computes predictions yhat = H W2^T + b2.
return yhat, h
Return both the predictions and hidden activations.
h is useful for visualization or debugging.
Backward pass (MLP.backward)
Why:

dh = self.l2.backward(dyhat)
Backprop through the second linear layer.
dyhat is ∂L/∂ŷ from the loss.
Returns ∂L/∂h.
if self.nonlin == "relu":
    dz1 = relu_backward(self.h_pre, dh)
else:
    dz1 = tanh_backward(self.h_pre, dh)
Backprop through the nonlinearity.
Uses the cached z1 (self.h_pre).
Implements chain rule: ∂L/∂z1 = ∂L/∂h ⊙ σ′(z1).
dx = self.l1.backward(dz1)
Backprop through the first linear layer.
Produces ∂L/∂x, useful if this MLP sits inside a larger model.
return dx
Return ∂L/∂x so the gradient can keep flowing backward.


Swap nonlin="tanh". Which fits better here and why? Reference bias/variance.
Try increasing the number of epochs from 1500 to 15000. What happens, and why?
Plot predictions versus ground truth to visually confirm fit.



STAGE 4 - Credit Score Feature Weight Estimator

L2 regularization in the loss and gradients.
A simple interpretability pass over learned weights.

Begin by defining sone input features about the people (credit utilization percentage, age, number of late payments), and a parallel array of their credit scores. We will try to estimate the function that maps these features to credit scores for new people (for whom their actual credit score is not yet known). We normalize the features so that they are all on the same scale by subtracting the mean of each feature from each element of that feature, and dividing by the standard deviation of that overall feature.

Inside your training loop, compute the MSE loss (as a scalar from the error vector) so that we can plot it. The error vector is yhat - yc. Compute this, square it (you can square an entire vector just like you would square a single scalar value), and compute the mean with np.mean. This is the Mean Squared Error (MSE). We square the values so that they are always positive, so that direction doesn’t offset the error artificially.
Next, compute the L2 penalty. This is the sum of the squares of all elements of lin.W. Multiply that sum by the regularization strength multiplier lam. This is referred to as lambda, and is a hyperparameter we use to tune training. A lambda value of 0 disables regularlization and just uses MSE to calculate loss, while larger values of lambda penalize large weights and incentivize smaller weight values.
The total loss is the MSE loss plus the L2 regularlization loss that you just computed in the prior two steps. Add these two terms together. Call this loss so that it appends to the losses array in the template above. By adding this to the loss, we consider a result with large weights as more lossy than a result with smaller weights, with the hope that this will allow our model to better generalize to new data.
Later, keep the running total of lin.dW, right before you reference it in the template. It is computed with the formula given in the TODO comment. At each step, you will add to the existing value of lin.dW using that formula.
Why:

L2 regularization (also called weight decay) discourages large weights by adding a penalty proportional to the squared magnitude of all weights.
Mathematically, if
L_total = L_data + λ ||W||²
then
∂L_total/∂W = ∂L_data/∂W + 2λW
That’s why you add 2 * lam * W to the gradient.
The effect is to shrink weights toward zero:
Smaller weights reduce variance and help generalization.
Larger lam increases the penalty (more bias, less variance).
Note: only the weights (W) are regularized, not the biases (b).



Which features carry the largest magnitude weights? What does the sign imply?
How does increasing lam affect weights and fit? Connect to bias–variance and overfitting risk.
By hand, use the weights and biases to calculate the credit score of someone with [0.30, 21, 0]. How does this compare to a person with [0.30, 65, 0], and what does this mean?



STAGE 5 - MNIST - Tiny Neural Net (Classification)

Implement a minimal two-layer classifier with softmax + cross-entropy, minibatching, and accuracy. We provide most code; you fill in the key formulas.

This code is complete. It imports the MNIST dataset, and creates a function one_hot which creates an array of all 0 values except for the one target element whose value is 1.

When training neural networks for classification tasks such as MNIST digit recognition, a common combination is the softmax activation in the output layer with a cross-entropy loss function.

The network outputs raw logits 
, which can be any real numbers. To interpret them as probabilities over classes (that add up to 100% probability, or 1.0), we apply the softmax function:

\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}}

Each output 
 \hat{y}_i is in [0,1]
 The probabilities sum to 1, making them interpretable as class likelihoods.

 Cross-Entropy Loss: Comparing Probabilities to Labels: For a one-hot label vector 
, the cross-entropy loss is: L = -\sum_{i=1}^C y_i \log(\hat{y}_i)

This is the difference between the predicted probabilities and the true labels, without computing the derivatives of the softmax function.'

Implement the mean negative log-likelihood (use a tiny epsilon value eps to avoid accidentally computing log(0)).

Compute the log (use np.log) of the probability vector probs. You can add eps to this to add the epsilon value to all elements in one line of code, just as if probs was a scalar value. Similarly, you can pass that whole result to np.log which will calculate on all elements of the vector.
Calculate the log-likelihood of each element being the correct class by calculating the product of Y_true by the log probability vector you just computed. Since the log of a probability should always be negative (since it is the log of a value between 0 and 1), multiply this result by -1 to make it a negative log likelihood. Then, call .sum(axis=1) on that vector. You will end up with a vector that is 0 in all positions except the one-hot position, since we multiplied it by the one-hot vector Y_true earlier. The one-hot position will have a likelihood corresponding to that correct classification. To spread this likelihood out over the remaining positions, return the mean of the negative log likelihood vector, which corresponds to a measure of how “surprised” the model is by this classification. For example:

Cross-entropy for one-hot Y_true is the negative log-probability assigned to the true class, averaged over the batch.
Probabilities are between 0 and 1, so their logarithms are ≤ 0.
If we summed raw log-probabilities, we would be maximizing a negative number, which doesn’t fit the idea of a “loss”.
By taking the negative log, we turn the objective into a positive quantity that can be minimized with gradient descent.
Mathematically:
Likelihood:
L(\theta) = \prod_i p_\theta(y_i \mid x_i)

Log-likelihood (to maximize):
\log L(\theta) = \sum_i \log p_\theta(y_i \mid x_i)

Negative log-likelihood (NLL) (to minimize):
\sum_i \log p_\theta(y_i \mid x_i)



Intuition:
If the model assigns high probability to the correct class (e.g. 0.99), log(prob) ≈ –0.01 → NLL ≈ 0.01 (good, small loss).
If the model assigns low probability (e.g. 0.01), log(prob) ≈ –4.6 → NLL ≈ 4.6 (large penalty).
.

This section puts everything together — linear layers, ReLU activation, softmax, and cross-entropy — into a trainable classifier.
The structure X → fc1 → ReLU → fc2 → softmax is the simplest neural net that can do non-linear classification.
We use minibatch SGD to train efficiently on subsets of MNIST data.
Each minibatch gives a noisy gradient estimate, but that noise helps escape poor local minima.
The loss curve (from cross_entropy) shows whether the model is learning; the accuracy curve checks generalization.
Forward and backward steps reuse the building blocks from earlier stages (Linear.forward, relu, softmax_cross_entropy_backward), reinforcing the idea that complex models are built from simple, composable parts.
By coding the loop manually, students see how deep-learning libraries like PyTorch or TensorFlow automate these mechanics.
Intuition: after Stage 5.3, they’ve basically built a mini-PyTorch for MNIST — proving they understand the computational graph and gradient flow.



