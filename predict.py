"""Predict income class for a person using the trained NumPy MLP.

Example:
  python predict.py --age 45 --workclass Private --education-num 14 \
      --marital-status Married-civ-spouse --occupation Exec-managerial \
      --relationship Husband --race White --sex Male --capital-gain 15000 \
      --hours-per-week 50 --native-country United-States
"""
import argparse
import os
import pickle

import pandas as pd

from src.data import clean
from src.mlp import MLP

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_model():
    model, d = MLP.load(os.path.join(ROOT, "models", "mlp_weights.npz"))
    with open(os.path.join(ROOT, "models", "preprocessor.pkl"), "rb") as f:
        prep = pickle.load(f)
    return model, prep, float(d["threshold"])


def predict(person: dict):
    model, prep, thr = load_model()
    X = prep.transform(clean(pd.DataFrame([person])))
    p = float(model.predict_proba(X)[0, 0])
    return p, thr, (">50K" if p >= thr else "<=50K")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--age", type=int, default=38)
    ap.add_argument("--workclass", default="Private")
    ap.add_argument("--education-num", type=int, default=10)
    ap.add_argument("--marital-status", default="Never-married")
    ap.add_argument("--occupation", default="Adm-clerical")
    ap.add_argument("--relationship", default="Not-in-family")
    ap.add_argument("--race", default="White")
    ap.add_argument("--sex", default="Female")
    ap.add_argument("--capital-gain", type=float, default=0)
    ap.add_argument("--capital-loss", type=float, default=0)
    ap.add_argument("--hours-per-week", type=int, default=40)
    ap.add_argument("--native-country", default="United-States")
    a = vars(ap.parse_args())
    person = {k.replace("_", "-"): v for k, v in a.items()}
    p, thr, label = predict(person)
    print("Input:", person)
    print(f"P(income > 50K) = {p:.4f}   threshold = {thr:.2f}   ->  prediction: {label}")


if __name__ == "__main__":
    main()
