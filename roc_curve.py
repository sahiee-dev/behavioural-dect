"""
ROC + Precision-Recall curves for the vocab_breadth detector.
Sweeps all thresholds on existing 60 runs. No API calls.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

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

# Load
neutral_vb, injected_vb = [], []
for dirs, bucket in [(DIRS["NEUTRAL"], neutral_vb), (DIRS["INJECTED"], injected_vb)]:
    for d in dirs:
        for p in sorted(d.glob("run_*.json")):
            log = json.loads(p.read_text())
            queries = [s["query"] for s in log["steps"]]
            bucket.append(vocab_breadth(queries))

neutral_vb  = np.array(neutral_vb)   # label 0
injected_vb = np.array(injected_vb)  # label 1

# Scores and labels for sklearn
scores = np.concatenate([neutral_vb, injected_vb])
labels = np.concatenate([np.zeros(len(neutral_vb)), np.ones(len(injected_vb))])

# AUC
roc_auc = roc_auc_score(labels, scores)
avg_prec = average_precision_score(labels, scores)

# Threshold sweep — flag as INJECTED if vb > t
thresholds = np.unique(np.concatenate([scores, [scores.min()-1e-6, scores.max()+1e-6]]))
thresholds = np.sort(thresholds)[::-1]  # high → low

tpr_list, fpr_list, prec_list, rec_list = [], [], [], []
operating_t = neutral_vb.max()  # 0.5701

for t in thresholds:
    tp = np.sum(injected_vb > t)
    fp = np.sum(neutral_vb  > t)
    fn = np.sum(injected_vb <= t)
    tn = np.sum(neutral_vb  <= t)
    tpr  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tpr_list.append(tpr); fpr_list.append(fpr)
    prec_list.append(prec); rec_list.append(rec)

tpr_arr  = np.array(tpr_list)
fpr_arr  = np.array(fpr_list)
prec_arr = np.array(prec_list)
rec_arr  = np.array(rec_list)

# Operating point
op_tp = np.sum(injected_vb > operating_t)
op_fp = np.sum(neutral_vb  > operating_t)
op_fn = np.sum(injected_vb <= operating_t)
op_tn = np.sum(neutral_vb  <= operating_t)
op_tpr  = op_tp / (op_tp + op_fn)
op_fpr  = op_fp / (op_fp + op_tn)
op_prec = op_tp / (op_tp + op_fp) if (op_tp + op_fp) > 0 else 1.0
op_rec  = op_tpr

# ── Print summary ─────────────────────────────────────────────────────────────
print("="*56)
print(f"vocab_breadth detector — ROC analysis (n=30/condition)")
print("="*56)
print(f"\nROC-AUC:           {roc_auc:.4f}")
print(f"Avg Precision:     {avg_prec:.4f}")
print(f"\nOperating point (threshold > {operating_t:.4f}):")
print(f"  TPR (detection): {op_tpr*100:.0f}%   FPR: {op_fpr*100:.0f}%")
print(f"  Precision:       {op_prec*100:.0f}%   Recall: {op_rec*100:.0f}%")

print("\nThreshold sweep (selected points):")
print(f"  {'Threshold':>10}  {'TPR':>6}  {'FPR':>6}  {'Precision':>10}  {'Recall':>8}")
print(f"  {'-'*48}")
shown = set()
for t, tpr, fpr, prec, rec in zip(thresholds, tpr_arr, fpr_arr, prec_arr, rec_arr):
    key = (round(tpr,2), round(fpr,2))
    if key in shown: continue
    shown.add(key)
    marker = " ← operating point" if abs(t - operating_t) < 1e-6 else ""
    print(f"  {t:>10.4f}  {tpr*100:>5.0f}%  {fpr*100:>5.0f}%  "
          f"{prec*100:>9.0f}%  {rec*100:>7.0f}%{marker}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ROC
ax1.plot(fpr_arr, tpr_arr, color="#1565C0", lw=2,
         label=f"vocab_breadth  AUC={roc_auc:.3f}")
ax1.plot([0,1],[0,1], "k--", lw=1, alpha=0.4, label="Random")
ax1.scatter([op_fpr], [op_tpr], s=100, color="red", zorder=5,
            label=f"Operating point\n(t>{operating_t:.4f}: TPR={op_tpr*100:.0f}%, FPR={op_fpr*100:.0f}%)")
ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curve — vocab_breadth detector")
ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.02)

# PR
ax2.plot(rec_arr, prec_arr, color="#2E7D32", lw=2,
         label=f"vocab_breadth  AP={avg_prec:.3f}")
ax2.scatter([op_rec], [op_prec], s=100, color="red", zorder=5,
            label=f"Operating point\n(Prec={op_prec*100:.0f}%, Rec={op_rec*100:.0f}%)")
ax2.axhline(len(injected_vb)/(len(injected_vb)+len(neutral_vb)),
            color="k", linestyle="--", lw=1, alpha=0.4, label="Random baseline")
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall Curve — vocab_breadth detector")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
ax2.set_xlim(-0.02, 1.02); ax2.set_ylim(-0.02, 1.02)

plt.suptitle("Behavioral Detector — vocab_breadth feature  (n=30/condition, real Tavily search)",
             fontsize=10)
plt.tight_layout()
out = Path("experiments/real_search_n30/roc_curve.png")
plt.savefig(out, dpi=150)
print(f"\nPlot → {out}")
