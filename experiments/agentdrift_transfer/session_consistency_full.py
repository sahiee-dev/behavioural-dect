"""
Full rigor (bootstrap CI + permutation p) for each collection SESSION
independently (original Sept n=30, new Oct n=30/60), for all 7 non-Claude
cross-model points. This is the definitive check of whether tonight's
"corrections" (Mistral=null, Gemma=robust reversal) are trustworthy, or
whether -- as initial inspection suggests -- some models show genuine
session-level instability that neither single session nor naive pooling
correctly characterizes.
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


def load(path, condition):
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["condition"] != condition or has_error(d):
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


models = [
    ("qwen2.5:7b", "qwen7b_renewable_n30.jsonl", "qwen7b_renewable_n60_batch2.jsonl"),
    ("qwen2.5:14b", "qwen14b_renewable_n30.jsonl", "qwen14b_renewable_n60_batch2.jsonl"),
    ("qwen2.5:32b", "qwen32b_renewable_n30.jsonl", "qwen32b_renewable_n60_batch2.jsonl"),
    ("mistral-small:24b", "mistral24b_renewable_n30.jsonl", "mistral24b_renewable_n60_batch2.jsonl"),
    ("gemma2:27b", "gemma27b_renewable_n30.jsonl", "gemma27b_renewable_n60_batch2.jsonl"),
    ("GPT-4o", "gpt4o_renewable_n30.jsonl", "gpt4o_renewable_n30_batch2.jsonl"),
    ("Gemini 2.5 Flash", "gemini_flash_renewable_n30.jsonl", "gemini_renewable_n30_batch2.jsonl"),
]

for label, orig_f, batch2_f in models:
    orig_n = load(ORIG / orig_f, "neutral")
    orig_i = load(ORIG / orig_f, "belief_injected")
    new_n = load(BATCH2 / batch2_f, "neutral")
    new_i = load(BATCH2 / batch2_f, "belief_injected")

    auc_o, ci_o, d_o, p_o = full_stat(orig_i, orig_n)
    auc_n, ci_n, d_n, p_n = full_stat(new_i, new_n)
    auc_c, ci_c, d_c, p_c = full_stat(orig_i + new_i, orig_n + new_n)

    sig_o = "sig" if (ci_o[0] > 0.5 or ci_o[1] < 0.5) else "n.s."
    sig_n = "sig" if (ci_n[0] > 0.5 or ci_n[1] < 0.5) else "n.s."
    sig_c = "sig" if (ci_c[0] > 0.5 or ci_c[1] < 0.5) else "n.s."
    agree = "AGREE" if (auc_o - 0.5) * (auc_n - 0.5) > 0 else "*** DISAGREE ***"

    print(f"\n{label}")
    print(f"  ORIG session: AUC={auc_o:.3f} [{ci_o[0]:.3f},{ci_o[1]:.3f}] d={d_o:+.3f} p={p_o:.4f} ({sig_o})")
    print(f"  NEW  session: AUC={auc_n:.3f} [{ci_n[0]:.3f},{ci_n[1]:.3f}] d={d_n:+.3f} p={p_n:.4f} ({sig_n})")
    print(f"  POOLED:       AUC={auc_c:.3f} [{ci_c[0]:.3f},{ci_c[1]:.3f}] d={d_c:+.3f} p={p_c:.4f} ({sig_c})")
    print(f"  Session agreement: {agree}")
