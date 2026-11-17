"""Interactive demo of the from-scratch NumPy MLP.  Run:  streamlit run app.py"""
import json
import os

import numpy as np
import pandas as pd
import streamlit as st

from predict import load_model
from src.data import clean

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "results", "figures")

st.set_page_config(page_title="Census Income MLP", layout="wide")
st.title("Census Income Classification — MLP from scratch (NumPy)")
st.caption("UCI Adult dataset · 91-64-32-1 network · ReLU + sigmoid · BCE + L2 · hand-written Adam · early stopping")

model, prep, thr = load_model()
with open(os.path.join(ROOT, "results", "results.json")) as f:
    R = json.load(f)

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Results", "🧠 Inside the network"])

with tab1:
    st.subheader("Enter a person's attributes")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.slider("Age", 17, 90, 38)
        edu = st.slider("Education level (education-num, 1=Preschool … 16=Doctorate)", 1, 16, 10)
        hours = st.slider("Hours per week", 1, 99, 40)
        sex = st.selectbox("Sex", ["Male", "Female"])
    with c2:
        workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                                               "Local-gov", "State-gov", "Without-pay", "Never-worked", "Unknown"])
        occupation = st.selectbox("Occupation", ["Exec-managerial", "Prof-specialty", "Tech-support", "Sales",
                                                 "Adm-clerical", "Craft-repair", "Machine-op-inspct", "Transport-moving",
                                                 "Handlers-cleaners", "Farming-fishing", "Other-service",
                                                 "Protective-serv", "Priv-house-serv", "Armed-Forces", "Unknown"])
        marital = st.selectbox("Marital status", ["Married-civ-spouse", "Never-married", "Divorced", "Separated",
                                                  "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
        relationship = st.selectbox("Relationship", ["Husband", "Wife", "Not-in-family", "Own-child",
                                                     "Unmarried", "Other-relative"])
    with c3:
        race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
        country = st.selectbox("Native country", ["United-States", "Mexico", "Philippines", "Germany", "Canada",
                                                  "India", "England", "China", "Cuba", "Unknown"])
        gain = st.number_input("Capital gain ($)", 0, 99999, 0, step=500)
        loss = st.number_input("Capital loss ($)", 0, 4356, 0, step=100)

    person = {"age": age, "workclass": workclass, "education-num": edu, "marital-status": marital,
              "occupation": occupation, "relationship": relationship, "race": race, "sex": sex,
              "capital-gain": gain, "capital-loss": loss, "hours-per-week": hours, "native-country": country}
    X = prep.transform(clean(pd.DataFrame([person])))
    p = float(model.predict_proba(X)[0, 0])
    label = ">50K" if p >= thr else "<=50K"
    st.divider()
    a, b, c = st.columns(3)
    a.metric("P(income > $50K)", f"{p*100:.1f}%")
    b.metric("Decision threshold (tuned on validation)", f"{thr:.2f}")
    c.metric("Prediction", label)
    st.progress(min(max(p, 0.0), 1.0))
    st.caption("⚠️ Educational demo trained on 1994 census data with sensitive attributes — not for real decisions.")

with tab2:
    m = R["mlp"]
    st.subheader("Test-set performance (7,327 held-out samples)")
    cols = st.columns(6)
    for col, (k, n) in zip(cols, [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                                  ("f1", "F1"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]):
        col.metric(n, f"{m[k]*100:.2f}%")
    st.subheader("Model comparison")
    names = {"logreg": "Logistic Regression", "mlp": "MLP (from scratch)", "xgboost": "XGBoost"}
    st.table(pd.DataFrame({names[k]: {q: f"{R[k][q]*100:.2f}%" for q in ["accuracy", "f1", "roc_auc", "pr_auc"]}
                           for k in names}).T)
    if "five_splits" in R:
        st.markdown("**Mean ± std over 5 random splits**")
        st.table(pd.DataFrame({names[k]: {q: f"{np.mean(v):.2f} ± {np.std(v, ddof=1):.2f}"
                                          for q, v in R["five_splits"][k].items()} for k in names}).T)
    g1, g2 = st.columns(2)
    g1.image(os.path.join(FIG, "learning_curves.png"))
    g2.image(os.path.join(FIG, "confusion_matrix.png"), width=420)
    g1.image(os.path.join(FIG, "roc_pr_curves.png"))
    g2.image(os.path.join(FIG, "threshold_tradeoff.png"))
    if os.path.exists(os.path.join(FIG, "five_split_comparison.png")):
        st.image(os.path.join(FIG, "five_split_comparison.png"), width=700)

with tab3:
    st.subheader("Forward pass for the current input")
    _, cache = model.forward(X)
    st.write(f"Input vector: **{X.shape[1]}** columns (5 scaled numeric + 86 one-hot) → "
             f"Hidden 1: **64** ReLU → Hidden 2: **32** ReLU → Output: **1** sigmoid. "
             f"Total parameters: **{model.n_params():,}**")
    h1, h2 = st.columns(2)
    h1.markdown("**Hidden layer 1 activations (64)**")
    h1.bar_chart(pd.DataFrame({"activation": cache["A1"][0]}))
    h2.markdown("**Hidden layer 2 activations (32)**")
    h2.bar_chart(pd.DataFrame({"activation": cache["A2"][0]}))
    st.write(f"Output pre-activation z = {cache['Z3'][0,0]:.3f} → sigmoid(z) = {p:.4f}")
    st.markdown(f"Active ReLUs: layer 1 = {(cache['A1'] > 0).sum()}/64, layer 2 = {(cache['A2'] > 0).sum()}/32")
