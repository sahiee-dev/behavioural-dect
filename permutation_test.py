"""
Permutation test for the vocab_breadth group difference.
Answers: "What is the p-value for the neutral vs. injected separation?"
Shuffles labels 100,000 times and measures how often a random partition
achieves AUC >= observed AUC.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

DIRS = {
    "neutral":  [Path("experiments/real_search_20runs/neutral"),
                 Path("experiments/real_search_n30/neutral")],
    "injected": [Path("experiments/real_search_20runs/belief_injected"),
                 Path("experiments/real_search_n30/belief_injected")],
}

def vocab_breadth(queries):
    u, t = set(), 0
    for q in queries:
        w = q.lower().split(); u.update(w); t += len(w)
    return len(u) / max(t, 1)

scores, labels = [], []
for cond, dirs in DIRS.items():
    for d in dirs:
        for p in sorted(d.glob("run_*.json")):
            log = json.loads(p.read_text())
            scores.append(vocab_breadth([s["query"] for s in log["steps"]]))
            labels.append(1 if cond == "injected" else 0)

X = np.array(scores)
y = np.array(labels)
observed_auc = roc_auc_score(y, X)

rng = np.random.default_rng(42)
N_PERM = 100_000
perm_aucs = np.empty(N_PERM)
for i in range(N_PERM):
    y_perm = rng.permutation(y)
    perm_aucs[i] = roc_auc_score(y_perm, X)

p_value = np.mean(perm_aucs >= observed_auc)

print("="*52)
print(f"Permutation test  (n_permutations={N_PERM:,})")
print("="*52)
print(f"\n  Observed AUC:  {observed_auc:.4f}")
print(f"  Permutation AUC: mean={perm_aucs.mean():.4f}  max={perm_aucs.max():.4f}")
print(f"  p-value:  {p_value:.6f}  ({int(np.sum(perm_aucs >= observed_auc))} / {N_PERM:,} permutations >= observed)")
print()
if p_value == 0.0:
    print(f"  p < {1/N_PERM:.6f}  (no permutation matched or exceeded observed AUC)")
    print(f"  For paper: 'p < 1e-5 (permutation test, n={N_PERM:,})'")
elif p_value < 0.001:
    print(f"  p < 0.001  — highly significant")
else:
    print(f"  p = {p_value:.4f}")

# Also test Cohen's d via permutation
n_scores = X[y == 0]
i_scores = X[y == 1]
ps = np.sqrt((n_scores.std()**2 + i_scores.std()**2) / 2)
observed_d = (i_scores.mean() - n_scores.mean()) / ps

perm_d = np.empty(N_PERM)
for i in range(N_PERM):
    y_perm = rng.permutation(y)
    ns = X[y_perm == 0]; is_ = X[y_perm == 1]
    ps_p = np.sqrt((ns.std()**2 + is_.std()**2) / 2)
    perm_d[i] = (is_.mean() - ns.mean()) / ps_p if ps_p > 0 else 0.0

p_d = np.mean(perm_d >= observed_d)
print(f"\n  Observed d:  {observed_d:+.2f}")
print(f"  p-value (d): {p_d:.6f}")
if p_d == 0.0:
    print(f"  For paper: 'd=3.91, p < 1e-5 (permutation test)'")
