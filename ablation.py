"""
Ablation: compare vocab_breadth against simpler baselines on 60-run dataset.
All free — no API calls. Addresses reviewer demand for baseline comparison.

Features compared:
  1. query_count       — total searches per session
  2. mean_query_length — mean token count per query
  3. query_len_std     — std of query lengths (diversity proxy)
  4. vocab_breadth     — unique_tokens / total_tokens (our feature)
  5. inter_query_dist  — mean pairwise Jaccard distance between queries
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from itertools import combinations

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

def tokenize(q: str) -> list[str]:
    return q.lower().split()

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)

def extract_features(log_path: Path) -> dict:
    log = json.loads(log_path.read_text())
    queries = [s["query"] for s in log["steps"]]
    token_lists = [tokenize(q) for q in queries]
    all_tokens = [t for tl in token_lists for t in tl]
    lengths = [len(tl) for tl in token_lists]
    sets = [set(tl) for tl in token_lists]

    # vocab_breadth
    vb = len(set(all_tokens)) / max(len(all_tokens), 1)

    # inter-query Jaccard distance (mean pairwise)
    if len(sets) >= 2:
        dists = [jaccard(a, b) for a, b in combinations(sets, 2)]
        iqd = float(np.mean(dists))
    else:
        iqd = 0.0

    return {
        "query_count": len(queries),
        "mean_query_length": float(np.mean(lengths)) if lengths else 0.0,
        "query_len_std": float(np.std(lengths)) if len(lengths) > 1 else 0.0,
        "vocab_breadth": vb,
        "inter_query_dist": iqd,
    }

data = {"neutral": [], "injected": []}
for cond, dirs in DIRS.items():
    for d in dirs:
        for p in sorted(d.glob("run_*.json")):
            data[cond].append(extract_features(p))

def to_arrays(feat: str) -> tuple[np.ndarray, np.ndarray]:
    n = np.array([r[feat] for r in data["neutral"]])
    i = np.array([r[feat] for r in data["injected"]])
    return n, i

FEATURES = [
    ("query_count",       False),  # lower for injected → flip sign for AUC
    ("mean_query_length", True),   # higher for injected
    ("query_len_std",     True),
    ("vocab_breadth",     True),
    ("inter_query_dist",  True),
]

y = np.concatenate([np.zeros(30), np.ones(30)])

print("="*68)
print("Ablation: vocab_breadth vs. baseline features (n=30/condition)")
print("="*68)
print(f"\n{'Feature':<22} {'Neutral':>14} {'Injected':>14} {'d':>7} {'AUC':>7}")
print(f"  {'-'*60}")

for feat, higher_is_injected in FEATURES:
    n, i = to_arrays(feat)
    ps = np.sqrt((n.std()**2 + i.std()**2) / 2)
    d = (i.mean() - n.mean()) / ps if ps > 0 else 0.0
    scores = np.concatenate([n, i])
    if not higher_is_injected:
        scores = -scores  # flip so injected > neutral for AUC
    auc = roc_auc_score(y, scores)
    print(f"  {feat:<22} {n.mean():>7.4f}±{n.std():.4f}  {i.mean():>7.4f}±{i.std():.4f}  "
          f"{d:>+6.2f}  {auc:.4f}")

print(f"\n  Direction: d>0 means injected > neutral; d<0 means injected < neutral")
print(f"  AUC measures separability (0.5=random, 1.0=perfect)")
print(f"\n  Key question: does any simpler feature match vocab_breadth AUC=0.990?")
