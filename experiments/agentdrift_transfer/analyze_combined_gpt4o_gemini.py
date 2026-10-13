"""
Combine original n=30 (contamination-filtered) + overnight n=30 extension
(collected with a working backup Tavily key, mostly/fully clean) for
GPT-4o and Gemini 2.5 Flash, giving a genuinely larger-sample estimate
rather than the small filtered convenience subsample used in the initial
data-integrity correction.
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


def load(path, condition, skip_errors=True):
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["condition"] != condition:
            continue
        if skip_errors and has_error(d):
            continue
        out.append(vocab_breadth(d["queries"]))
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


for label, orig_f, batch2_f in [
    ("GPT-4o", "gpt4o_renewable_n30.jsonl", "gpt4o_renewable_n30_batch2.jsonl"),
    ("Gemini 2.5 Flash", "gemini_flash_renewable_n30.jsonl", "gemini_renewable_n30_batch2.jsonl"),
]:
    combined_n = load(ORIG / orig_f, "neutral") + load(BATCH2 / batch2_f, "neutral")
    combined_i = load(ORIG / orig_f, "belief_injected") + load(BATCH2 / batch2_f, "belief_injected")
    auc, ci, d, p = full_stat(combined_i, combined_n)
    print(f"{label}: COMBINED clean n={len(combined_n)}+{len(combined_i)}  "
          f"AUC={auc:.4f}  95% CI [{ci[0]:.4f},{ci[1]:.4f}]  d={d:+.3f}  p={p:.5f}")
