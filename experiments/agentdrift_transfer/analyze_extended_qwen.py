"""
Re-analyze the three Qwen cross-model points with the sample tripled
(original n=30/condition + overnight batch n=60/condition = n=90/condition),
using the identical bootstrap CI / permutation test / Guiraud / MATTR
methodology as the rest of this project.

This is the direct test of the paper's own repeatedly-flagged "most useful
next experiment": a larger Qwen sample to resolve whether its reversal/
non-effect is real or just underpowered at n=30.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ORIG = Path("/Users/lulu/Desktop/silentdrop/evaluation/crossover_bd/n30_rerun")
BATCH2 = Path(__file__).parent
W = 10


def mattr(tokens, window=W):
    n = len(tokens)
    if n <= window:
        return len(set(tokens)) / max(n, 1)
    ratios = [len(set(tokens[i:i + window])) / window for i in range(n - window + 1)]
    return float(np.mean(ratios))


def vocab_breadth(queries):
    u, t = set(), 0
    for q in queries:
        w = q.lower().split()
        u.update(w)
        t += len(w)
    return len(u) / max(t, 1), t, len(queries)


def load_jsonl(path, condition):
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["condition"] != condition:
            continue
        vb, nt, nq = vocab_breadth(d["queries"])
        tokens = []
        for q in d["queries"]:
            tokens.extend(q.lower().split())
        out.append({
            "vocab_breadth": vb,
            "n_tokens": nt,
            "n_queries": nq,
            "guiraud": (vb * nt) / max(nt, 1) ** 0.5,
            "mattr": mattr(tokens),
        })
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


def full_stat(pos, neg, feature):
    p = np.array([r[feature] for r in pos], dtype=float)
    n = np.array([r[feature] for r in neg], dtype=float)
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


models = [
    ("qwen2.5:7b", "qwen7b_renewable_n30.jsonl", "qwen7b_renewable_n60_batch2.jsonl"),
    ("qwen2.5:14b", "qwen14b_renewable_n30.jsonl", "qwen14b_renewable_n60_batch2.jsonl"),
    ("qwen2.5:32b", "qwen32b_renewable_n30.jsonl", "qwen32b_renewable_n60_batch2.jsonl"),
]

results = {}
for label, orig_file, batch2_file in models:
    neu = load_jsonl(ORIG / orig_file, "neutral") + load_jsonl(BATCH2 / batch2_file, "neutral")
    inj = load_jsonl(ORIG / orig_file, "belief_injected") + load_jsonl(BATCH2 / batch2_file, "belief_injected")
    print(f"\n{'='*70}\n{label}  EXTENDED n={len(neu)}+{len(inj)} (was 30+30)\n{'='*70}")
    model_results = {}
    for feat in ("vocab_breadth", "guiraud", "mattr"):
        auc, ci, d, p = full_stat(inj, neu, feat)
        sig = "SIGNIFICANT" if (ci[0] > 0.5 or ci[1] < 0.5) else "n.s. (CI crosses 0.5)"
        model_results[feat] = dict(auc=auc, ci=list(ci), d=d, p=p, sig=sig)
        print(f"  {feat:16s} AUC={auc:.4f}  95% CI [{ci[0]:.4f},{ci[1]:.4f}]  d={d:+.3f}  p={p:.5f}  {sig}")
    results[label] = model_results

out = Path(__file__).parent / "extended_qwen_results.json"
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {out}")
