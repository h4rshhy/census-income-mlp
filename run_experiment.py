"""Full experiment from the report.

1. Train the 91-64-32-1 NumPy MLP on split seed 42 with early stopping.
2. Pick the decision threshold on validation, evaluate once on the test set.
3. Train Logistic Regression and XGBoost baselines on the same split.
4. Repeat everything on 5 more random splits (seeds 1-5) -> mean +/- std.
5. Save tables (results/report.md, results/results.json), figures and the trained model.

Usage:  python run_experiment.py            (full run)
        python run_experiment.py --quick    (skip the 5-split repetition)
"""
import argparse
import json
import os
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve
from xgboost import XGBClassifier

from src.data import load_data, make_split
from src.evaluation import best_threshold, metrics
from src.mlp import MLP, train

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
MODELS = os.path.join(ROOT, "models")
NAMES = {"logreg": "Logistic Regression", "mlp": "MLP (from scratch)", "xgboost": "XGBoost"}


def run_all(df, y_all, seed, verbose=False):
    X_tr, X_va, X_te, y_tr, y_va, y_te, prep = make_split(df, y_all, seed)
    out = {"sizes": (len(y_tr), len(y_va), len(y_te)), "n_inputs": X_tr.shape[1]}

    mlp = MLP([X_tr.shape[1], 64, 32, 1], lr=0.001, l2=1e-4, seed=seed)
    out["n_params"] = mlp.n_params()
    out["history"] = train(mlp, X_tr, y_tr, X_va, y_va, verbose=verbose)
    p_va, p_te = mlp.predict_proba(X_va), mlp.predict_proba(X_te)
    thr = best_threshold(y_va, p_va)
    out["mlp"] = metrics(y_te, p_te, thr)
    out["mlp_0.5"] = metrics(y_te, p_te, 0.5)

    lr = LogisticRegression(max_iter=2000).fit(X_tr, y_tr.ravel())
    p_lr = lr.predict_proba(X_te)[:, 1]
    out["logreg"] = metrics(y_te, p_lr, best_threshold(y_va, lr.predict_proba(X_va)[:, 1]))

    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                        subsample=0.9, eval_metric="logloss", random_state=seed,
                        n_jobs=4).fit(X_tr, y_tr.ravel())
    p_xgb = xgb.predict_proba(X_te)[:, 1]
    out["xgboost"] = metrics(y_te, p_xgb, best_threshold(y_va, xgb.predict_proba(X_va)[:, 1]))

    out["_objects"] = dict(mlp=mlp, prep=prep, y_te=y_te.ravel(), y_va=y_va.ravel(),
                           p_va=p_va.ravel(),
                           probs={"mlp": p_te.ravel(), "logreg": p_lr, "xgboost": p_xgb})
    return out


def pct(x):
    return f"{x * 100:.2f}%"


def make_figures(res, multi):
    os.makedirs(FIG, exist_ok=True)
    h, o = res["history"], res["_objects"]
    ep = np.arange(1, len(h["train_loss"]) + 1)

    # 1. learning curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, h["train_loss"], label="Train loss")
    ax[0].plot(ep, h["val_loss"], label="Validation loss")
    ax[0].axvline(h["best_epoch"], ls="--", c="gray", label=f"Best epoch ({h['best_epoch']})")
    ax[0].set(xlabel="Epoch", ylabel="BCE + L2 loss", title="Loss curves")
    ax[0].legend()
    ax[1].plot(ep, np.array(h["val_acc"]) * 100, c="tab:green")
    ax[1].axvline(h["best_epoch"], ls="--", c="gray")
    ax[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Validation accuracy (threshold 0.5)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "learning_curves.png"), dpi=130); plt.close(fig)

    # 2. confusion matrix
    cm = res["mlp"]["cm"]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1], ["<=50K", ">50K"]); ax.set_yticks([0, 1], ["<=50K", ">50K"])
    ax.set(xlabel="Predicted", ylabel="Actual",
           title=f"MLP confusion matrix (thr={res['mlp']['threshold']:.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "confusion_matrix.png"), dpi=130); plt.close(fig)

    # 3. ROC + PR curves for all models
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for k in ["logreg", "mlp", "xgboost"]:
        p = o["probs"][k]
        fpr, tpr, _ = roc_curve(o["y_te"], p)
        ax[0].plot(fpr, tpr, label=f"{NAMES[k]} (AUC {res[k]['roc_auc']*100:.2f})")
        pr, rc, _ = precision_recall_curve(o["y_te"], p)
        ax[1].plot(rc, pr, label=f"{NAMES[k]} (AP {res[k]['pr_auc']*100:.2f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test)")
    ax[1].axhline(o["y_te"].mean(), c="k", ls=":", lw=1, label="No-skill baseline")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve (test)")
    for a in ax: a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "roc_pr_curves.png"), dpi=130); plt.close(fig)

    # 4. threshold trade-off on validation set
    grid = np.linspace(0.05, 0.95, 91)
    from sklearn.metrics import f1_score, precision_score, recall_score
    P = [precision_score(o["y_va"], o["p_va"] >= t, zero_division=0) for t in grid]
    R = [recall_score(o["y_va"], o["p_va"] >= t) for t in grid]
    F = [f1_score(o["y_va"], o["p_va"] >= t) for t in grid]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(grid, P, label="Precision"); ax.plot(grid, R, label="Recall"); ax.plot(grid, F, label="F1")
    ax.axvline(res["mlp"]["threshold"], c="gray", ls="--", label=f"Chosen thr {res['mlp']['threshold']:.2f}")
    ax.axvline(0.5, c="red", ls=":", label="Default 0.5")
    ax.set(xlabel="Decision threshold", ylabel="Score", title="MLP threshold selection (validation)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "threshold_tradeoff.png"), dpi=130); plt.close(fig)

    # 5. 5-split comparison
    if multi:
        keys = ["accuracy", "f1", "roc_auc", "pr_auc"]
        x = np.arange(len(keys)); w = 0.26
        fig, ax = plt.subplots(figsize=(8, 4.2))
        for i, m in enumerate(["logreg", "mlp", "xgboost"]):
            mu = [np.mean(multi[m][k]) for k in keys]; sd = [np.std(multi[m][k], ddof=1) for k in keys]
            ax.bar(x + (i - 1) * w, mu, w, yerr=sd, capsize=3, label=NAMES[m])
        ax.set_xticks(x, ["Accuracy", "F1", "ROC-AUC", "PR-AUC"]); ax.set_ylim(60, 95)
        ax.set(ylabel="%", title="Mean ± std over 5 random splits"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "five_split_comparison.png"), dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the 5-split repetition")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True); os.makedirs(MODELS, exist_ok=True)

    df, y_all = load_data()
    print(f"Loaded {len(df):,} records, {df.shape[1]} features, "
          f"positive (>50K) rate {y_all.mean()*100:.2f}%")

    t0 = time.time()
    print("\n=== Main run (split seed 42) - training NumPy MLP ===")
    res = run_all(df, y_all, 42, verbose=True)
    h = res["history"]
    print(f"Split sizes train/val/test: {res['sizes']}, input columns: {res['n_inputs']}, "
          f"MLP parameters: {res['n_params']:,}")
    print(f"Best epoch: {h['best_epoch']}  Epochs run: {h['epochs_run']}  "
          f"({time.time()-t0:.1f}s incl. baselines)")

    multi = None
    if not args.quick:
        print("\n=== Repeating on 5 random splits (seeds 1-5) ===")
        multi = {m: {k: [] for k in ["accuracy", "f1", "roc_auc", "pr_auc"]}
                 for m in ["logreg", "mlp", "xgboost"]}
        for s in [1, 2, 3, 4, 5]:
            r = run_all(df, y_all, s)
            print(f"  seed {s}: " + "  ".join(
                f"{NAMES[m]} F1={r[m]['f1']*100:.2f}" for m in multi))
            for m in multi:
                for k in multi[m]:
                    multi[m][k].append(r[m][k] * 100)

    # ---------- save model artifacts ----------
    o = res["_objects"]
    o["mlp"].save(os.path.join(MODELS, "mlp_weights.npz"), threshold=res["mlp"]["threshold"])
    with open(os.path.join(MODELS, "preprocessor.pkl"), "wb") as f:
        pickle.dump(o["prep"], f)

    make_figures(res, multi)

    # ---------- report ----------
    m, m5 = res["mlp"], res["mlp_0.5"]
    cm = m["cm"]
    L = ["# Results - NumPy MLP on UCI Adult\n",
         f"Records: {len(df):,} | Split train/val/test: {res['sizes'][0]:,} / {res['sizes'][1]:,} / "
         f"{res['sizes'][2]:,} | Input columns: {res['n_inputs']} | MLP parameters: {res['n_params']:,}\n",
         f"Early stopping: best epoch **{h['best_epoch']}**, stopped at epoch **{h['epochs_run']}**. "
         f"Validation-selected threshold: **{m['threshold']:.2f}**.\n",
         "## Table 1: MLP performance on the test set\n", "| Metric | Score |", "|---|---|"]
    for k, n in [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                 ("f1", "F1-score"), ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC")]:
        L.append(f"| {n} | {pct(m[k])} |")
    L += ["\n## Table 2: Confusion matrix (test set)\n",
          "| | Predicted <=50K | Predicted >50K |", "|---|---|---|",
          f"| Actual <=50K | {cm[0,0]} | {cm[0,1]} |", f"| Actual >50K | {cm[1,0]} | {cm[1,1]} |",
          f"\nMisclassified: {cm[0,1]+cm[1,0]:,} ({cm[0,1]} FP, {cm[1,0]} FN). At threshold 0.5: "
          f"accuracy {pct(m5['accuracy'])}, precision {pct(m5['precision'])}, recall {pct(m5['recall'])}, "
          f"F1 {pct(m5['f1'])}.\n",
          "## Table 3: Comparison with baselines (test set)\n",
          "| Model | Accuracy | F1-score | ROC-AUC | PR-AUC |", "|---|---|---|---|---|"]
    for k in ["logreg", "mlp", "xgboost"]:
        r = res[k]
        L.append(f"| {NAMES[k]} | {pct(r['accuracy'])} | {pct(r['f1'])} | {pct(r['roc_auc'])} | {pct(r['pr_auc'])} |")
    if multi:
        L += ["\n## Table 4: Average over 5 random splits (mean ± std, %)\n",
              "| Model | Accuracy | F1-score | ROC-AUC | PR-AUC |", "|---|---|---|---|---|"]
        for k in ["logreg", "mlp", "xgboost"]:
            L.append(f"| {NAMES[k]} | " + " | ".join(
                f"{np.mean(multi[k][q]):.2f} ± {np.std(multi[k][q], ddof=1):.2f}"
                for q in ["accuracy", "f1", "roc_auc", "pr_auc"]) + " |")
    L += ["\n## Table 5: Training progress\n",
          "| Epoch | Train Loss | Validation Loss | Validation Accuracy |", "|---|---|---|---|"]
    for e in sorted({1, 5, 10, h["best_epoch"], 20, h["epochs_run"]}):
        if e <= len(h["train_loss"]):
            tag = " (best)" if e == h["best_epoch"] else (" (stopped)" if e == h["epochs_run"] else "")
            L.append(f"| {e}{tag} | {h['train_loss'][e-1]:.4f} | {h['val_loss'][e-1]:.4f} | "
                     f"{pct(h['val_acc'][e-1])} |")
    report = "\n".join(L) + "\n"
    with open(os.path.join(RES, "report.md"), "w") as f:
        f.write(report)

    js = {k: {q: (v.tolist() if hasattr(v, "tolist") else v) for q, v in res[k].items()}
          for k in ["mlp", "mlp_0.5", "logreg", "xgboost"]}
    js["history"] = h
    if multi:
        js["five_splits"] = multi
    with open(os.path.join(RES, "results.json"), "w") as f:
        json.dump(js, f, indent=2, default=float)

    print("\n" + report)
    print(f"Saved: results/report.md, results/results.json, results/figures/*.png, models/*")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
