"""
Robustness check: does the Guiraud-based reclassification of qwen2.5:32b and
mistral-small:24b (inconclusive/reversed -> significant Claude-direction,
see full_length_audit.py) depend on that specific length-normalization
formula, or does a second one (MATTR, moving-average TTR) agree?

MATTR: slide a window of W tokens across the pooled query text, compute TTR
in each window, average. Falls back to plain TTR if total tokens < W.
Standard default window in the literature is 10-50 tokens for short texts;
we use W=10 given this project's queries pool to ~70-170 tokens per run.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

BD = Path("/Users/lulu/agentops-replay-pro/behavioral_detector")
CROSS = Path("/Users/lulu/Desktop/silentdrop/evaluation/crossover_bd/n30_rerun")
W = 10


def mattr(tokens: list[str], window: int = W) -> float:
    n = len(tokens)
    if n <= window:
        return len(set(tokens)) / max(n, 1)
    ratios = []
    for i in range(n - window + 1):
        win = tokens[i:i + window]
        ratios.append(len(set(win)) / window)
    return float(np.mean(ratios))


def load_claude_dir(path: Path):
    out = []
    for p in sorted(path.glob("run_*.json")):
        log = json.loads(p.read_text())
        tokens = []
        for s in log["steps"]:
            tokens.extend(s["query"].lower().split())
        out.append({"mattr": mattr(tokens)})
    return out


def load_jsonl(path: Path, condition: str):
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["condition"] != condition:
            continue
        tokens = []
        for q in d["queries"]:
            tokens.extend(q.lower().split())
        out.append({"mattr": mattr(tokens)})
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


def full_stat(pos, neg, feature="mattr"):
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


datasets = {
    "Claude (Task1)": (
        load_claude_dir(BD / "experiments/real_search_20runs/neutral") + load_claude_dir(BD / "experiments/real_search_n30/neutral"),
        load_claude_dir(BD / "experiments/real_search_20runs/belief_injected") + load_claude_dir(BD / "experiments/real_search_n30/belief_injected"),
    ),
}
for jsonl_name, label in [
    ("qwen7b_renewable_n30.jsonl", "qwen2.5:7b"),
    ("qwen14b_renewable_n30.jsonl", "qwen2.5:14b"),
    ("qwen32b_renewable_n30.jsonl", "qwen2.5:32b"),
    ("gemma27b_renewable_n30.jsonl", "gemma2:27b"),
    ("mistral24b_renewable_n30.jsonl", "mistral-small:24b"),
    ("gpt4o_renewable_n30.jsonl", "GPT-4o"),
    ("gemini_flash_renewable_n30.jsonl", "Gemini 2.5 Flash"),
]:
    path = CROSS / jsonl_name
    datasets[label] = (load_jsonl(path, "neutral"), load_jsonl(path, "belief_injected"))

print(f"{'Model':20s} {'MATTR AUC':>10s} {'95% CI':>18s} {'d':>8s} {'p':>10s}  Significant?")
for label, (neu, inj) in datasets.items():
    auc, ci, d, p = full_stat(inj, neu)
    sig = "yes" if ci[0] > 0.5 or ci[1] < 0.5 else "no (CI crosses 0.5)"
    print(f"{label:20s} {auc:10.4f} [{ci[0]:.4f},{ci[1]:.4f}]  {d:+8.3f} {p:10.5f}  {sig}")
