"""
Discovery: the generic open-weight harness (run_belief_injection_harness.py)
has the same runaway-capture bug found and fixed for Gemini early in this
project's cross-model work, but it was never checked for the other open-
weight models. The regex's query group uses DOTALL, so when a model doesn't
stop cleanly, the "query" field can capture subsequent OBSERVATION/THOUGHT/
ACTION text too -- inflating vocab_breadth with real-world web-content
vocabulary that was never actually a deliberate search query.

Found via: avg words/query was 40-80 for Qwen/Mistral vs 6.6 for Gemma --
a huge, suspicious outlier that turned out to be real contamination, not
a behavioral difference. 9-31% of queries affected depending on model;
0% for Gemma specifically.

Fix: truncate each captured query at the first newline (a legitimate query
is always single-line per the harness's own prompt format: "ACTION: search
| query: <query>" on one line). This is the same fix already applied to the
Gemini harness months ago, applied retroactively here to already-collected
data for every other open-weight model.

This recomputes vocab_breadth with cleaned queries across ALL sessions
(original + overnight extensions + third-session tie-breakers) for every
open-weight model, to see whether any of tonight's finalized conclusions
change.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ORIG = Path("/Users/lulu/Desktop/silentdrop/evaluation/crossover_bd/n30_rerun")
BATCH2 = ORIG / "batch2_overnight"


def clean_query(q):
    return q.split("\n")[0].strip()


def vocab_breadth(queries, clean=True):
    qs = [clean_query(q) for q in queries] if clean else queries
    u, t = set(), 0
    for q in qs:
        w = q.lower().split()
        u.update(w)
        t += len(w)
    return len(u) / max(t, 1)


def load(path, condition, clean):
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["condition"] != condition:
            continue
        out.append(vocab_breadth(d["queries"], clean=clean))
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


models = {
    "qwen2.5:7b": [ORIG / "qwen7b_renewable_n30.jsonl", BATCH2 / "qwen7b_renewable_n60_batch2.jsonl", BATCH2 / "qwen7b_renewable_n30_session3.jsonl"],
    "qwen2.5:14b": [ORIG / "qwen14b_renewable_n30.jsonl", BATCH2 / "qwen14b_renewable_n60_batch2.jsonl"],
    "qwen2.5:32b": [ORIG / "qwen32b_renewable_n30.jsonl", BATCH2 / "qwen32b_renewable_n60_batch2.jsonl"],
    "gemma2:27b": [ORIG / "gemma27b_renewable_n30.jsonl", BATCH2 / "gemma27b_renewable_n60_batch2.jsonl", BATCH2 / "gemma27b_renewable_n30_session3.jsonl"],
    "mistral-small:24b": [ORIG / "mistral24b_renewable_n30.jsonl", BATCH2 / "mistral24b_renewable_n60_batch2.jsonl"],
}

print(f"{'Model':20s} {'DIRTY AUC':>10s} {'d':>8s} {'p':>9s}   {'CLEAN AUC':>10s} {'d':>8s} {'p':>9s}   Changed?")
for label, paths in models.items():
    n_dirty = [v for p in paths for v in load(p, "neutral", clean=False)]
    i_dirty = [v for p in paths for v in load(p, "belief_injected", clean=False)]
    n_clean = [v for p in paths for v in load(p, "neutral", clean=True)]
    i_clean = [v for p in paths for v in load(p, "belief_injected", clean=True)]

    auc_d, ci_d, d_d, p_d = full_stat(i_dirty, n_dirty)
    auc_c, ci_c, d_c, p_c = full_stat(i_clean, n_clean)

    sig_d = "sig" if (ci_d[0] > 0.5 or ci_d[1] < 0.5) else "n.s."
    sig_c = "sig" if (ci_c[0] > 0.5 or ci_c[1] < 0.5) else "n.s."
    changed = "*** CHANGED ***" if sig_d != sig_c or (d_d > 0) != (d_c > 0) else ""

    print(f"{label:20s} {auc_d:10.4f} {d_d:+8.3f} {p_d:9.5f}({sig_d})   {auc_c:10.4f} {d_c:+8.3f} {p_c:9.5f}({sig_c})   {changed}")
    print(f"   95% CI dirty: [{ci_d[0]:.4f},{ci_d[1]:.4f}]   95% CI clean: [{ci_c[0]:.4f},{ci_c[1]:.4f}]")
