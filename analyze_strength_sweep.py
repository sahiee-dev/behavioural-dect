"""
Injection-strength boundary table: weak / medium / strong vocab_breadth
vs. the existing n=30 neutral baseline (Task 1, renewable energy).
Medium = existing run_n30.py data (current wording), not re-run.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

def vb(queries):
    u=set(); t=0
    for q in queries:
        w=q.lower().split(); u.update(w); t+=len(w)
    return len(u)/max(t,1)

def load_dir(*dirs):
    out = []
    for d in dirs:
        for p in sorted(Path(d).glob("run_*.json")):
            out.append(vb([s["query"] for s in json.loads(p.read_text())["steps"]]))
    return np.array(out)

# match Task 1's official n=30 baseline exactly (Phase 2 n=10 + Phase 3 n=20)
NEUTRAL = load_dir("experiments/real_search_20runs/neutral", "experiments/real_search_n30/neutral")
MEDIUM  = load_dir("experiments/real_search_20runs/belief_injected", "experiments/real_search_n30/belief_injected")
WEAK    = load_dir("experiments/injection_strength/weak")
STRONG  = load_dir("experiments/injection_strength/strong")

def report(label, iv):
    ps = np.sqrt((NEUTRAL.std()**2 + iv.std()**2)/2)
    d  = (iv.mean() - NEUTRAL.mean()) / ps if ps > 0 else 0
    auc  = roc_auc_score(
        np.concatenate([np.zeros(len(NEUTRAL)), np.ones(len(iv))]),
        np.concatenate([NEUTRAL, iv])
    )
    t_max = NEUTRAL.max()
    det = np.sum(iv > t_max)
    print(f"  {label:<8}  n={len(iv):>2}  mean={iv.mean():.4f}  std={iv.std():.4f}  "
          f"d={d:+.2f}  AUC={auc:.4f}  det@0%FP={det}/{len(iv)}={det/len(iv)*100:.0f}%")
    return d, auc

print(f"Neutral baseline (n={len(NEUTRAL)}): mean={NEUTRAL.mean():.4f} std={NEUTRAL.std():.4f}")
print()
print("Injection-strength boundary:")
report("weak",   WEAK)
report("medium", MEDIUM)
report("strong", STRONG)
