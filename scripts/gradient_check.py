"""Numerical gradient check: proves the hand-written backpropagation is correct.

Compares each analytic gradient from MLP.backward() with a central finite
difference (L(w+h) - L(w-h)) / 2h on a small random network.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.mlp import MLP  # noqa: E402


def main():
    rng = np.random.RandomState(0)
    X = rng.randn(20, 6)
    y = (rng.rand(20, 1) > 0.5).astype(float)
    net = MLP([6, 5, 4, 1], l2=1e-2, seed=1)
    # shift biases so no ReLU sits exactly at its kink
    for k in net.params:
        if k.startswith("b"):
            net.params[k] += 0.1

    _, cache = net.forward(X)
    grads = net.backward(y, cache)
    h = 1e-6
    worst = 0.0
    print(f"{'param':>5} {'shape':>8} {'max rel. error':>15}")
    for k, P in net.params.items():
        num = np.zeros_like(P)
        it = np.nditer(P, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = P[idx]
            P[idx] = old + h; lp = net.compute_loss(net.forward(X)[0], y)
            P[idx] = old - h; lm = net.compute_loss(net.forward(X)[0], y)
            P[idx] = old
            num[idx] = (lp - lm) / (2 * h)
        rel = np.max(np.abs(num - grads[k]) / np.maximum(1e-8, np.abs(num) + np.abs(grads[k])))
        worst = max(worst, rel)
        print(f"{k:>5} {str(P.shape):>8} {rel:15.2e}")
    ok = worst < 1e-5
    print(f"\nWorst relative error: {worst:.2e} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
