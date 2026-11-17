"""Data loading, cleaning, splitting and preprocessing (Section 2 & 3 of the report)."""
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "adult-all.csv")

COLS = ["age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "income"]
NUM = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT = ["workclass", "marital-status", "occupation", "relationship", "race",
       "sex", "native-country"]


def clean(df):
    """Apply the same cleaning to raw rows (used for training and for new predictions)."""
    df = df.replace("?", "Unknown")
    df = df.drop(columns=[c for c in ["fnlwgt", "education"] if c in df.columns])
    df = df.copy()
    for c in ["capital-gain", "capital-loss"]:
        df[c] = np.log1p(df[c].astype(float))
    return df


def load_data(path=DATA_PATH):
    """Return (features DataFrame with 12 columns, binary label array)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run: python scripts/download_data.py")
    df = pd.read_csv(path, header=None, names=COLS, skipinitialspace=True)
    y = df["income"].str.strip().str.startswith(">50K").astype(int).values
    X = clean(df.drop(columns=["income"]))
    return X, y


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])


def make_split(df, y_all, seed):
    """Stratified 70 / 15 / 15 split; preprocessing is fitted on the training set only."""
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        df, y_all, test_size=0.30, random_state=seed, stratify=y_all)
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=seed, stratify=y_tmp)
    prep = make_preprocessor()
    X_tr = prep.fit_transform(X_tr)
    X_va, X_te = prep.transform(X_va), prep.transform(X_te)
    return (X_tr, X_va, X_te,
            y_tr.reshape(-1, 1), y_va.reshape(-1, 1), y_te.reshape(-1, 1), prep)
