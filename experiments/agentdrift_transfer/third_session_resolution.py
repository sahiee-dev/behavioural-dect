"""
Final resolution: a third independent session for the two models that
showed session-level instability after sessions 1 (original, weeks
earlier) and 2 (overnight extension): gemma2:27b and qwen2.5:7b.

Session 1: original n=30 collection (crossover_bd/n30_rerun/).
Session 2: overnight n=60 extension (batch2_overnight/).
Session 3: this script's data -- a fresh, independent n=30 collection,
run specifically to break the 1-1 tie between sessions 1 and 2.

Run: python3 third_session_resolution.py (paths point to the local-only
data location and need adjusting to re-run elsewhere).
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ORIG = Path("/Users/lulu/Desktop/silentdrop/evaluation/crossover_bd/n30_rerun")
BATCH2 = ORIG / "batch2_overnight"


def has_error(d):
    return any(
        "exceeds your plan" in s.get("text", "") or "search error" in s.get("text", "")
        for s in d["steps"] if s.get("kind") == "OBSERVATION"
    )


def vocab_breadth(queries):
    u, t = set(), 0
    for q in queries:
        w = q.lower().split()
        u.update(w)
        t += len(w)
    return len(u) / max(t, 1)


def load(path, condition, feature="vb"):
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["condition"] != condition or has_error(d):
            continue
        out.append(vocab_breadth(d["queries"]) if feature == "vb" else len(d["queries"]))
    return out


def permutation_p(n_arr, p_arr, global_auc, N=100_000, batch=5_000, seed=42):
    combined = np.concatenate([n_arr, p_arr])
    n_neg, n_pos = len(n_arr), len(p_arr)
    n_total = n_neg + n_pos
    ranks = rankdata(combined, method="average")
    rng = np.random.default_rng(seed)
    perm_auc = np.empty(N)
    done = 0
    while done < N:
        b = min(batch, N - done)
        keys = rng.random((b, n_total))
        idx = np.argsort(keys, axis=1)[:, :n_pos]
        rank_sum = ranks[idx].sum(axis=1)
        U = rank_sum - n_pos * (n_pos + 1) / 2
        perm_auc[done:done + b] = U / (n_pos * n_neg)
        done += b
    return np.mean(np.abs(perm_auc - 0.5) >= np.abs(global_auc - 0.5))


def full_stat(pos, neg):
    p = np.array(pos, dtype=float)
    n = np.array(neg, dtype=float)
    y = np.concatenate([np.zeros(len(n)), np.ones(len(p))])
    X = np.concatenate([n, p])
    auc = roc_auc_score(y, X)
    ps = np.sqrt((n.std() ** 2 + p.std() ** 2) / 2)
    d = (p.mean() - n.mean()) / ps if ps > 0 else 0.0
    rng = np.random.default_rng(42)
    B = 10_000
    boot = np.empty(B)
    for b in range(B):
        ni = rng.integers(0, len(n), size=len(n))
        pi = rng.integers(0, len(p), size=len(p))
        nb, pb = n[ni], p[pi]
        yb = np.concatenate([np.zeros(len(nb)), np.ones(len(pb))])
        Xb = np.concatenate([nb, pb])
        boot[b] = roc_auc_score(yb, Xb)
    ci = (np.percentile(boot, 2.5), np.percentile(boot, 97.5))
    pval = permutation_p(n, p, auc)
    return auc, ci, d, pval


for label, s1, s2, s3 in [
    ("Gemma2-27B", "gemma27b_renewable_n30.jsonl", "gemma27b_renewable_n60_batch2.jsonl", "gemma27b_renewable_n30_session3.jsonl"),
    ("Qwen2.5-7B", "qwen7b_renewable_n30.jsonl", "qwen7b_renewable_n60_batch2.jsonl", "qwen7b_renewable_n30_session3.jsonl"),
]:
    s1p, s2p, s3p = ORIG / s1, BATCH2 / s2, BATCH2 / s3
    print(f"\n=== {label} ===")
    for name, path in [("Session 1 (original)", s1p), ("Session 2 (overnight)", s2p), ("Session 3 (tie-breaker)", s3p)]:
        n = load(path, "neutral")
        i = load(path, "belief_injected")
        auc, ci, d, p = full_stat(i, n)
        print(f"  {name:28s} n={len(n)}+{len(i)}  AUC={auc:.4f}  95%CI[{ci[0]:.4f},{ci[1]:.4f}]  d={d:+.3f}  p={p:.5f}")

    n_all = load(s1p, "neutral") + load(s2p, "neutral") + load(s3p, "neutral")
    i_all = load(s1p, "belief_injected") + load(s2p, "belief_injected") + load(s3p, "belief_injected")
    auc, ci, d, p = full_stat(i_all, n_all)
    print(f"  {'POOLED (all 3 sessions)':28s} n={len(n_all)}+{len(i_all)}  AUC={auc:.4f}  95%CI[{ci[0]:.4f},{ci[1]:.4f}]  d={d:+.3f}  p={p:.5f}")

    nq_n = np.array(load(s1p, "neutral", "nq") + load(s2p, "neutral", "nq") + load(s3p, "neutral", "nq"))
    nq_i = np.array(load(s1p, "belief_injected", "nq") + load(s2p, "belief_injected", "nq") + load(s3p, "belief_injected", "nq"))
    ps = np.sqrt((nq_n.std() ** 2 + nq_i.std() ** 2) / 2)
    d_nq = (nq_i.mean() - nq_n.mean()) / ps
    print(f"  pooled n_queries d = {d_nq:+.3f} (neutral mean={nq_n.mean():.2f}, injected mean={nq_i.mean():.2f})")
