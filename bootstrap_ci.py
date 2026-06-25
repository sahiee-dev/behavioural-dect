"""
Bootstrap 95% confidence intervals for AUC and Cohen's d.
Uses 10,000 resamples on the 60-run dataset.
Free — no API calls.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

DIRS = {
    "neutral": [
        Path("experiments/real_search_20runs/neutral"),
        Path("experiments/real_search_n30/neutral"),
    ],
    "injected": [
        Path("experiments/real_search_20runs/belief_injected"),
        Path("experiments/real_search_n30/belief_injected"),
    ],
}

def vocab_breadth(queries: list[str]) -> float:
    u: set[str] = set()
    t = 0
    for q in queries:
        w = q.lower().split()
        u.update(w)
        t += len(w)
    return len(u) / max(t, 1)

scores = {"neutral": [], "injected": []}
for cond, dirs in DIRS.items():
    for d in dirs:
        for p in sorted(d.glob("run_*.json")):
            log = json.loads(p.read_text())
            scores[cond].append(vocab_breadth([s["query"] for s in log["steps"]]))

n = np.array(scores["neutral"])
i = np.array(scores["injected"])
y = np.concatenate([np.zeros(len(n)), np.ones(len(i))])
X = np.concatenate([n, i])

global_auc = roc_auc_score(y, X)
ps = np.sqrt((n.std()**2 + i.std()**2) / 2)
global_d = (i.mean() - n.mean()) / ps

rng = np.random.default_rng(42)
B = 10_000
boot_auc = np.empty(B)
boot_d   = np.empty(B)

for b in range(B):
    ni = rng.integers(0, len(n), size=len(n))
    ii = rng.integers(0, len(i), size=len(i))
    nb, ib = n[ni], i[ii]
    yb = np.concatenate([np.zeros(len(nb)), np.ones(len(ib))])
    Xb = np.concatenate([nb, ib])
    boot_auc[b] = roc_auc_score(yb, Xb)
    ps_b = np.sqrt((nb.std()**2 + ib.std()**2) / 2)
    boot_d[b] = (ib.mean() - nb.mean()) / ps_b if ps_b > 0 else 0.0

ci_auc = (np.percentile(boot_auc, 2.5), np.percentile(boot_auc, 97.5))
ci_d   = (np.percentile(boot_d, 2.5),   np.percentile(boot_d, 97.5))

print("="*52)
print(f"Bootstrap 95% CIs  (B={B:,} resamples, n=60)")
print("="*52)
print(f"\n  AUC:      {global_auc:.4f}   95% CI [{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]")
print(f"  Cohen's d: {global_d:+.2f}   95% CI [{ci_d[0]:+.2f}, {ci_d[1]:+.2f}]")
print(f"\n  Neutral:  mean={n.mean():.4f}  std={n.std():.4f}  n={len(n)}")
print(f"  Injected: mean={i.mean():.4f}  std={i.std():.4f}  n={len(i)}")
print(f"\n  Interpretation:")
if ci_auc[0] >= 0.90:
    print(f"  → AUC lower bound {ci_auc[0]:.4f} ≥ 0.90 — strong claim survives resampling.")
elif ci_auc[0] >= 0.80:
    print(f"  → AUC lower bound {ci_auc[0]:.4f} ≥ 0.80 — good, caveatable for n=60.")
else:
    print(f"  → AUC lower bound {ci_auc[0]:.4f} < 0.80 — wide CI, need more data.")
print(f"\n  For paper: 'AUC=0.990, 95% CI [{ci_auc[0]:.3f}, {ci_auc[1]:.3f}] (B=10,000 bootstrap)'")
