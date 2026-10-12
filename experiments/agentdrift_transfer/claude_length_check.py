"""
Length-covariate check on the ORIGINAL Task 1 Claude data (same mechanism
vocab_breadth was built for: belief-prefill via system prompt), motivated by
the confirmed length confound found on AgentDrift (a different mechanism).

Uses the exact same run logs and vocab_breadth formula as ../../bootstrap_ci.py.
Adds: n_tokens and n_steps (query count) as standalone features, and the
Pearson correlation between vocab_breadth and each, to check whether Claude's
AUC=0.990 result is itself partly a length artifact.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parents[2]  # .../behavioral_detector

DIRS = {
    "neutral": [
        BASE / "experiments/real_search_20runs/neutral",
        BASE / "experiments/real_search_n30/neutral",
    ],
    "injected": [
        BASE / "experiments/real_search_20runs/belief_injected",
        BASE / "experiments/real_search_n30/belief_injected",
    ],
}


def vocab_breadth_and_len(queries: list[str]) -> tuple[float, int, int]:
    u: set[str] = set()
    t = 0
    for q in queries:
        w = q.lower().split()
        u.update(w)
        t += len(w)
    return len(u) / max(t, 1), t, len(queries)


records = {"neutral": [], "injected": []}
for cond, dirs in DIRS.items():
    for d in dirs:
        for p in sorted(d.glob("run_*.json")):
            log = json.loads(p.read_text())
            queries = [s["query"] for s in log["steps"]]
            vb, n_tokens, n_steps = vocab_breadth_and_len(queries)
            records[cond].append({"vocab_breadth": vb, "n_tokens": n_tokens, "n_steps": n_steps})

print(f"neutral n={len(records['neutral'])}  injected n={len(records['injected'])}")


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


def auc_ci_perm(feature, label):
    p = np.array([r[feature] for r in records["injected"]], dtype=float)
    n = np.array([r[feature] for r in records["neutral"]], dtype=float)
    y = np.concatenate([np.zeros(len(n)), np.ones(len(p))])
    X = np.concatenate([n, p])
    global_auc = roc_auc_score(y, X)
    ps = np.sqrt((n.std() ** 2 + p.std() ** 2) / 2)
    global_d = (p.mean() - n.mean()) / ps if ps > 0 else 0.0

    rng = np.random.default_rng(42)
    B = 10_000
    boot_auc = np.empty(B)
    for b in range(B):
        ni = rng.integers(0, len(n), size=len(n))
        pi = rng.integers(0, len(p), size=len(p))
        nb, pb = n[ni], p[pi]
        yb = np.concatenate([np.zeros(len(nb)), np.ones(len(pb))])
        Xb = np.concatenate([nb, pb])
        boot_auc[b] = roc_auc_score(yb, Xb)
    ci = (np.percentile(boot_auc, 2.5), np.percentile(boot_auc, 97.5))
    p_value = permutation_p(n, p, global_auc)

    print(f"\n--- {label} (feature={feature}) ---")
    print(f"  AUC: {global_auc:.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]  "
          f"Cohen's d: {global_d:+.3f}  permutation p={p_value:.5f} (N=100,000)")
    print(f"  neutral mean={n.mean():.4f} sd={n.std():.4f} | injected mean={p.mean():.4f} sd={p.std():.4f}")


auc_ci_perm("vocab_breadth", "injected vs neutral — vocab_breadth (reference result)")
auc_ci_perm("n_tokens", "injected vs neutral — n_tokens (length confound check)")
auc_ci_perm("n_steps", "injected vs neutral — n_steps / query count (length confound check)")

all_vb = np.array([r["vocab_breadth"] for r in records["neutral"] + records["injected"]])
all_tok = np.array([r["n_tokens"] for r in records["neutral"] + records["injected"]])
all_steps = np.array([r["n_steps"] for r in records["neutral"] + records["injected"]])
print(f"\nPearson r(vocab_breadth, n_tokens) across all {len(all_vb)} runs: {np.corrcoef(all_vb, all_tok)[0,1]:+.3f}")
print(f"Pearson r(vocab_breadth, n_steps)  across all {len(all_vb)} runs: {np.corrcoef(all_vb, all_steps)[0,1]:+.3f}")

# Guiraud's index as the length-normalized alternative, same as the AgentDrift check.
# Full bootstrap CI + permutation test, not just a point estimate, for consistency
# with every other number in this project.
for r in records["neutral"] + records["injected"]:
    r["guiraud"] = (r["vocab_breadth"] * r["n_tokens"]) / max(r["n_tokens"], 1) ** 0.5
auc_ci_perm("guiraud", "injected vs neutral — Guiraud's index (length-normalized alternative)")

all_g = np.array([r["guiraud"] for r in records["neutral"] + records["injected"]])
print(f"\nPearson r(Guiraud, n_tokens): {np.corrcoef(all_g, all_tok)[0,1]:+.3f}")
