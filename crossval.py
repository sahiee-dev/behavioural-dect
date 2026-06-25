"""
K-fold cross-validation of the vocab_breadth threshold.
Shows the threshold is stable across folds, not overfitted to training-set max.
Runs on existing 60 logs. Free — no API calls.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DIRS = {
    "NEUTRAL":  [Path("experiments/real_search_20runs/neutral"),
                 Path("experiments/real_search_n30/neutral")],
    "INJECTED": [Path("experiments/real_search_20runs/belief_injected"),
                 Path("experiments/real_search_n30/belief_injected")],
}

def vocab_breadth(queries: list[str]) -> float:
    tokens_unique: set[str] = set()
    tokens_total = 0
    for q in queries:
        words = q.lower().split()
        tokens_unique.update(words)
        tokens_total += len(words)
    return len(tokens_unique) / max(tokens_total, 1)

vb_scores, labels = [], []
for cond, dirs in DIRS.items():
    label = 0 if cond == "NEUTRAL" else 1
    for d in dirs:
        for p in sorted(d.glob("run_*.json")):
            log = json.loads(p.read_text())
            queries = [s["query"] for s in log["steps"]]
            vb_scores.append(vocab_breadth(queries))
            labels.append(label)

X = np.array(vb_scores)
y = np.array(labels)

K = 5
kf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

print("="*56)
print(f"vocab_breadth  —  {K}-fold stratified cross-validation")
print(f"n={len(X)} total ({sum(y==0)} neutral, {sum(y==1)} injected)")
print("="*56)
print(f"\n{'Fold':>5}  {'Threshold':>10}  {'Detection':>10}  {'FP':>5}  {'AUC':>6}")
print(f"  {'-'*44}")

fold_thresholds, fold_tpr, fold_fpr, fold_auc = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Threshold = max of neutral scores in training fold
    neutral_train = X_train[y_train == 0]
    threshold = float(neutral_train.max())

    # Evaluate on test fold
    y_pred = (X_test > threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_test == 1))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    tn = np.sum((y_pred == 0) & (y_test == 0))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # AUC on test fold
    try:
        auc = roc_auc_score(y_test, X_test)
    except Exception:
        auc = float("nan")

    fold_thresholds.append(threshold)
    fold_tpr.append(tpr)
    fold_fpr.append(fpr)
    fold_auc.append(auc)

    print(f"  {fold:>4}  {threshold:>10.4f}  "
          f"{tpr*100:>9.0f}%  {fpr*100:>4.0f}%  {auc:>6.4f}")

print(f"  {'-'*44}")
print(f"  {'Mean':>4}  {np.mean(fold_thresholds):>10.4f}  "
      f"{np.mean(fold_tpr)*100:>9.1f}%  {np.mean(fold_fpr)*100:>4.1f}%  "
      f"{np.mean(fold_auc):>6.4f}")
print(f"  {'Std':>4}  {np.std(fold_thresholds):>10.4f}  "
      f"{np.std(fold_tpr)*100:>9.1f}%  {np.std(fold_fpr)*100:>4.1f}%  "
      f"{np.std(fold_auc):>6.4f}")

print(f"\nThreshold range across folds: "
      f"[{min(fold_thresholds):.4f}, {max(fold_thresholds):.4f}]")
print(f"Stability: std={np.std(fold_thresholds):.4f} "
      f"({'stable' if np.std(fold_thresholds) < 0.01 else 'unstable — needs larger n'})")

# Global AUC for reference
global_auc = roc_auc_score(y, X)
print(f"\nGlobal AUC (all 60 runs): {global_auc:.4f}")
