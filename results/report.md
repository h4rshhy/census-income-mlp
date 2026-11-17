# Results - NumPy MLP on UCI Adult

Records: 48,842 | Split train/val/test: 34,189 / 7,326 / 7,327 | Input columns: 91 | MLP parameters: 8,001

Early stopping: best epoch **16**, stopped at epoch **26**. Validation-selected threshold: **0.33**.

## Table 1: MLP performance on the test set

| Metric | Score |
|---|---|
| Accuracy | 83.83% |
| Precision | 63.25% |
| Recall | 77.35% |
| F1-score | 69.59% |
| ROC-AUC | 91.02% |
| PR-AUC | 77.38% |

## Table 2: Confusion matrix (test set)

| | Predicted <=50K | Predicted >50K |
|---|---|---|
| Actual <=50K | 4786 | 788 |
| Actual >50K | 397 | 1356 |

Misclassified: 1,185 (788 FP, 397 FN). At threshold 0.5: accuracy 85.55%, precision 73.80%, recall 61.38%, F1 67.02%.

## Table 3: Comparison with baselines (test set)

| Model | Accuracy | F1-score | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 82.48% | 67.13% | 89.49% | 72.33% |
| MLP (from scratch) | 83.83% | 69.59% | 91.02% | 77.38% |
| XGBoost | 85.82% | 72.43% | 92.85% | 82.78% |

## Table 4: Average over 5 random splits (mean ± std, %)

| Model | Accuracy | F1-score | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| Logistic Regression | 82.98 ± 0.72 | 68.28 ± 0.53 | 90.02 ± 0.30 | 74.21 ± 1.14 |
| MLP (from scratch) | 84.00 ± 0.44 | 70.25 ± 0.52 | 91.37 ± 0.30 | 78.43 ± 0.95 |
| XGBoost | 86.79 ± 0.39 | 73.20 ± 1.07 | 92.93 ± 0.39 | 83.21 ± 0.83 |

## Table 5: Training progress

| Epoch | Train Loss | Validation Loss | Validation Accuracy |
|---|---|---|---|
| 1 | 0.3332 | 0.3326 | 84.90% |
| 5 | 0.3149 | 0.3193 | 85.59% |
| 10 | 0.3055 | 0.3164 | 85.86% |
| 16 (best) | 0.2963 | 0.3117 | 86.17% |
| 20 | 0.2949 | 0.3142 | 85.97% |
| 26 (stopped) | 0.2884 | 0.3142 | 85.87% |
