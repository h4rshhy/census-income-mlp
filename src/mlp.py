"""Multilayer Perceptron written from first principles with NumPy only.

Forward propagation, binary cross-entropy + L2 loss, backpropagation and the
Adam optimizer are all implemented by hand here.
"""
import copy

import numpy as np


class MLP:
    def __init__(self, layer_sizes, lr=0.001, l2=1e-4, seed=42):
        self.layer_sizes = list(layer_sizes)
        self.rng = np.random.RandomState(seed)
        self.lr, self.l2 = lr, l2
        self.params = {}
        # He initialisation for ReLU layers
        for i in range(1, len(layer_sizes)):
            self.params[f"W{i}"] = self.rng.randn(
                layer_sizes[i - 1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i - 1])
            self.params[f"b{i}"] = np.zeros((1, layer_sizes[i]))
        self.L = len(layer_sizes) - 1
        # Adam state
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.t = 0
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

    # ---------- activations ----------
    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_grad(z):
        return (z > 0).astype(float)

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    # ---------- forward ----------
    def forward(self, X):
        cache = {"A0": X}
        A = X
        for i in range(1, self.L + 1):
            Z = A @ self.params[f"W{i}"] + self.params[f"b{i}"]
            A = self.relu(Z) if i < self.L else self.sigmoid(Z)
            cache[f"Z{i}"], cache[f"A{i}"] = Z, A
        return A, cache

    def compute_loss(self, y_pred, y_true):
        eps = 1e-9
        bce = -np.mean(y_true * np.log(y_pred + eps) +
                       (1 - y_true) * np.log(1 - y_pred + eps))
        l2_term = sum(np.sum(self.params[f"W{i}"] ** 2) for i in range(1, self.L + 1))
        return bce + 0.5 * self.l2 * l2_term

    # ---------- backward ----------
    def backward(self, y_true, cache):
        grads = {}
        m = y_true.shape[0]
        # d(BCE)/dZ_L for sigmoid output simplifies to (A_L - y)
        dZ = cache[f"A{self.L}"] - y_true
        for i in range(self.L, 0, -1):
            A_prev = cache[f"A{i-1}"]
            grads[f"W{i}"] = (A_prev.T @ dZ) / m + self.l2 * self.params[f"W{i}"]
            grads[f"b{i}"] = np.sum(dZ, axis=0, keepdims=True) / m
            if i > 1:
                dA_prev = dZ @ self.params[f"W{i}"].T
                dZ = dA_prev * self.relu_grad(cache[f"Z{i-1}"])
        return grads

    # ---------- Adam ----------
    def update(self, grads):
        self.t += 1
        for k in self.params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[k] ** 2)
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def predict_proba(self, X):
        return self.forward(X)[0]

    def n_params(self):
        return int(sum(v.size for v in self.params.values()))

    # ---------- persistence ----------
    def save(self, path, **extra):
        np.savez(path, layer_sizes=np.array(self.layer_sizes),
                 **{k: v for k, v in self.params.items()}, **extra)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        model = cls(d["layer_sizes"].tolist())
        model.params = {k: d[k] for k in model.params}
        return model, d


def train(model, X, y, X_val, y_val, max_epochs=100, batch_size=256, patience=10,
          verbose=False):
    """Mini-batch training with early stopping on validation loss."""
    from sklearn.metrics import accuracy_score

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val, best_params, best_epoch, wait = np.inf, None, 0, 0
    n = X.shape[0]
    for epoch in range(1, max_epochs + 1):
        perm = model.rng.permutation(n)
        X_shuf, y_shuf = X[perm], y[perm]
        for start in range(0, n, batch_size):
            Xb, yb = X_shuf[start:start + batch_size], y_shuf[start:start + batch_size]
            _, cache = model.forward(Xb)
            model.update(model.backward(yb, cache))
        tr_pred, val_pred = model.predict_proba(X), model.predict_proba(X_val)
        tr_loss = model.compute_loss(tr_pred, y)
        val_loss = model.compute_loss(val_pred, y_val)
        val_acc = accuracy_score(y_val, val_pred >= 0.5)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        improved = val_loss < best_val - 1e-5
        if improved:
            best_val, best_epoch, wait = val_loss, epoch, 0
            best_params = copy.deepcopy(model.params)
        else:
            wait += 1
        if verbose:
            print(f"  epoch {epoch:3d}  train_loss {tr_loss:.4f}  val_loss {val_loss:.4f}  "
                  f"val_acc {val_acc*100:5.2f}%{'  *best*' if improved else ''}")
        if wait >= patience:
            if verbose:
                print(f"  early stopping: no improvement for {patience} epochs")
            break
    model.params = best_params
    history["best_epoch"], history["epochs_run"] = best_epoch, epoch
    return history
