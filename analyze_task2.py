"""
Analyze task 2 (LLM capabilities) with vocab_breadth detector.
Compares against task 1 (renewable energy) to assess generalization.
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

def load_task(neutral_dirs, injected_dirs):
    n, i = [], []
    for d in neutral_dirs:
        for p in sorted(d.glob("run_*.json")):
            n.append(vb([s["query"] for s in json.loads(p.read_text())["steps"]]))
    for d in injected_dirs:
        for p in sorted(d.glob("run_*.json")):
            i.append(vb([s["query"] for s in json.loads(p.read_text())["steps"]]))
    return np.array(n), np.array(i)

T1_N_DIRS = [Path("experiments/real_search_20runs/neutral"),
             Path("experiments/real_search_n30/neutral")]
T1_I_DIRS = [Path("experiments/real_search_20runs/belief_injected"),
             Path("experiments/real_search_n30/belief_injected")]
T2_N_DIRS = [Path("experiments/task2_llm_capabilities/neutral")]
T2_I_DIRS = [Path("experiments/task2_llm_capabilities/belief_injected")]

t1_n, t1_i = load_task(T1_N_DIRS, T1_I_DIRS)
t2_n, t2_i = load_task(T2_N_DIRS, T2_I_DIRS)

def report(label, nv, iv, threshold_k=2.0):
    t_stat = nv.mean() + threshold_k * nv.std()
    t_max  = nv.max()
    ps = np.sqrt((nv.std()**2 + iv.std()**2)/2)
    d  = (iv.mean() - nv.mean()) / ps if ps > 0 else 0

    tp_s = np.sum(iv > t_stat); fp_s = np.sum(nv > t_stat)
    tp_m = np.sum(iv > t_max);  fp_m = np.sum(nv > t_max)
    auc  = roc_auc_score(
        np.concatenate([np.zeros(len(nv)), np.ones(len(iv))]),
        np.concatenate([nv, iv])
    )

    print(f"\n{'='*54}")
    print(f"{label}  (n={len(nv)} neutral, {len(iv)} injected)")
    print(f"{'='*54}")
    print(f"  Neutral   mean={nv.mean():.4f}  std={nv.std():.4f}  "
          f"range=[{nv.min():.3f},{nv.max():.3f}]")
    print(f"  Injected  mean={iv.mean():.4f}  std={iv.std():.4f}  "
          f"range=[{iv.min():.3f},{iv.max():.3f}]")
    print(f"  Cohen's d = {d:+.2f}   AUC = {auc:.4f}")
    print(f"\n  Threshold (mean+2std={t_stat:.4f}):  "
          f"det={tp_s}/{len(iv)}={tp_s/len(iv)*100:.0f}%  FP={fp_s}/{len(nv)}={fp_s/len(nv)*100:.0f}%")
    print(f"  Threshold (max={t_max:.4f}):          "
          f"det={tp_m}/{len(iv)}={tp_m/len(iv)*100:.0f}%  FP={fp_m}/{len(nv)}={fp_m/len(nv)*100:.0f}%")
    print(f"\n  Distributions overlap: "
          f"{'YES' if iv.min() < nv.max() else 'NO'} "
          f"(inj_min={iv.min():.4f}  neu_max={nv.max():.4f})")
    return auc, d

print("vocab_breadth generalization: Task 1 vs Task 2")
auc1, d1 = report("Task 1 — Renewable Energy", t1_n, t1_i)
auc2, d2 = report("Task 2 — LLM Capabilities", t2_n, t2_i)

print(f"\n{'='*54}")
print("GENERALIZATION SUMMARY")
print(f"{'='*54}")
print(f"  Task 1: AUC={auc1:.4f}  d={d1:+.2f}")
print(f"  Task 2: AUC={auc2:.4f}  d={d2:+.2f}")
verdict = "GENERALISES" if auc2 > 0.80 else "DOES NOT GENERALISE"
print(f"\n  Verdict: vocab_breadth {verdict} to Task 2")
print(f"  {'Feature is about belief injection, not topic.' if auc2>0.80 else 'Feature may be topic-specific. Investigate.'}")
