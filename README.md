# Deep Learning on Tabular Data: an MLP for Census Income Classification

This project builds a Multilayer Perceptron **from scratch in NumPy** to predict whether a person earns more than $50K/year from the UCI Adult (Census Income) dataset. It covers forward propagation, backpropagation, BCE + L2 loss, the Adam optimizer and early stopping. The MLP is compared with Logistic Regression and XGBoost.

## Project layout
```
census-income-mlp/
├── scripts/download_data.py   # downloads UCI Adult and merges it into data/adult-all.csv (48,842 rows)
├── scripts/gradient_check.py  # numerical check that the hand-written backprop is correct
├── src/data.py                # cleaning, 70/15/15 stratified split, scaling + one-hot (fitted on train only)
├── src/mlp.py                 # NumPy MLP: forward, loss, backward, Adam, early-stopping trainer, save/load
├── src/evaluation.py          # validation-set threshold search + metrics
├── run_experiment.py          # full experiment that produces Tables 1-5, the figures and the saved model
├── predict.py                 # command-line prediction with the trained model
├── app.py                     # interactive Streamlit demo
├── results/                   # report.md, results.json, figures/*.png (generated)
└── models/                    # mlp_weights.npz, preprocessor.pkl (generated)
```

## How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # on macOS XGBoost also needs: brew install libomp
python scripts/download_data.py
python scripts/gradient_check.py         # PASS = backprop is correct
python run_experiment.py                 # about 10 s; add --quick to skip the 5-split repeat
python predict.py --age 45 --education-num 14 --marital-status Married-civ-spouse \
    --occupation Exec-managerial --relationship Husband --sex Male --capital-gain 15000
streamlit run app.py                     # opens the demo at http://localhost:8501
```

## Setup
| Item | Details |
|---|---|
| Data | 48,842 records, 12 features (5 numeric, 7 categorical) giving 91 inputs after one-hot encoding |
| Split | 34,189 train / 7,326 validation / 7,327 test, stratified |
| Network | 91-64-32-1, ReLU hidden layers, sigmoid output, 8,001 parameters |
| Training | Adam with lr 0.001, L2 1e-4, batch size 256, up to 100 epochs, early stopping with patience 10 |
| Threshold | the one that gives the best F1 on the validation set (0.33) |

## Results (test set, split seed 42)
| Model | Accuracy | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 82.48% | 67.13% | 89.49% | 72.33% |
| **MLP (from scratch)** | **83.83%** | **69.59%** | **91.02%** | **77.38%** |
| XGBoost | 85.82% | 72.43% | 92.85% | 82.78% |

Early stopping picks epoch 16 and stops at epoch 26. The full tables, including the confusion matrix, the mean ± std over 5 splits and the training progress, are in `results/report.md`.

> Note: this model was trained on 1994 census data that includes sensitive attributes. It is for learning only and should not be used to make real decisions.
