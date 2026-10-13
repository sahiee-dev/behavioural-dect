"""
Third transfer test for vocab_breadth: ATBench (arXiv:2604.02022, public,
Apache 2.0, huggingface.co/datasets/AI45Research/ATBench), a 1000-trajectory
agent-safety benchmark with direct/indirect prompt-injection labels --
closer to this project's own belief-injection mechanism than AgentDrift's
external-payload hijack was, and a genuinely different benchmark/authors/
task domain than either Task 1-3 or AgentDrift.

Comparison: benign (risk_source='benign', label=0) vs. prompt-injection
trajectories (risk_source in {direct_prompt_injection, indirect_prompt_injection}).
Feature pooled per trajectory: agent "thought" + flattened "action" JSON
args across all agent turns (the agent's own generated text, excluding
user messages and tool/environment outputs -- same scoping choice as the
AgentDrift transfer test).
"""
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

DATA = Path("/private/tmp/claude-501/-Users-lulu-Desktop/a25eac7b-897b-485e-ab35-3ba41ddb5364/scratchpad/atbench_data/test.json")


def vocab_breadth(text):
    w = text.lower().split()
    u = set(w)
    return len(u) / max(len(w), 1), len(w)


def agent_text(trajectory):
    parts = []
    for turn in trajectory:
        if turn.get("role") != "agent":
            continue
        if turn.get("thought"):
            parts.append(turn["thought"])
        action = turn.get("action")
        if action:
            try:
                parsed = json.loads(action)
                parts.append(json.dumps(parsed.get("arguments", {})))
            except Exception:
                parts.append(str(action))
    return " ".join(parts)


records = json.loads(DATA.read_text())
benign, injected = [], []
for r in records:
    text = agent_text(r["contents"][0])
    vb, nt = vocab_breadth(text)
    entry = {"vocab_breadth": vb, "n_tokens": nt}
    if r["risk_source"] == "benign":
        benign.append(entry)
    elif r["risk_source"] in ("direct_prompt_injection", "indirect_prompt_injection"):
        injected.append(entry)

print(f"benign n={len(benign)}  prompt-injection n={len(injected)}")


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


for feat in ("vocab_breadth", "n_tokens"):
    auc, ci, d, p = full_stat(injected, benign, feat)
    print(f"{feat:16s} AUC={auc:.4f}  95% CI [{ci[0]:.4f},{ci[1]:.4f}]  d={d:+.3f}  p={p:.5f}")

all_vb = np.array([r["vocab_breadth"] for r in benign + injected])
all_nt = np.array([r["n_tokens"] for r in benign + injected])
print(f"Pearson r(vocab_breadth, n_tokens) = {np.corrcoef(all_vb, all_nt)[0,1]:+.3f}")
