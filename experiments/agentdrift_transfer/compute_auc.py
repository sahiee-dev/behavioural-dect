"""
AUC + bootstrap 95% CI + permutation test for vocab_breadth separating
AgentDrift (web domain) hijacked vs benign trajectories -- same methodology
as ../../bootstrap_ci.py (B=10,000, seed=42) and ../../permutation_test.py
(N=100,000), but with the permutation test vectorized via rank-sum (Mann-
Whitney U) instead of a per-permutation sklearn call, since n~2000 per
comparison here is far larger than the n=60 the original script was timed on.

Also reports the length confound directly (AUC of n_tokens / n_steps alone,
and their Pearson r with vocab_breadth), since this is exactly the confound
flagged in paper_draft.md Limitations #6.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

records = [json.loads(l) for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]

benign = [r for r in records if r["label"] == 0]
hijacked = [r for r in records if r["label"] == 1]
failed = [r for r in records if r["category"] == "failed_attack"]
hard_neg = [r for r in records if r["category"] == "hard_negative"]

print(f"benign n={len(benign)}  hijacked n={len(hijacked)}  "
      f"failed_attack n={len(failed)}  hard_negative n={len(hard_neg)}")


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


def auc_ci_perm(pos, neg, feature, label):
    p = np.array([r[feature] for r in pos], dtype=float)
    n = np.array([r[feature] for r in neg], dtype=float)
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
    print(f"  neg mean={n.mean():.4f} sd={n.std():.4f} | pos mean={p.mean():.4f} sd={p.std():.4f}")
    return global_auc, ci, global_d, p_value


auc_ci_perm(hijacked, benign, "vocab_breadth", "hijacked vs benign — vocab_breadth")
auc_ci_perm(hijacked, benign, "n_tokens", "hijacked vs benign — n_tokens (length confound check)")
auc_ci_perm(hijacked, benign, "n_steps", "hijacked vs benign — n_steps (length confound check)")

print("\n--- sensitivity: failed_attack vs benign (attack attempted, hijack did NOT succeed) ---")
auc_ci_perm(failed, benign, "vocab_breadth", "failed_attack vs benign — vocab_breadth")

print("\n--- sensitivity: hard_negative vs benign (benign-looking but tricky task) ---")
auc_ci_perm(hard_neg, benign, "vocab_breadth", "hard_negative vs benign — vocab_breadth")

all_vb = np.array([r["vocab_breadth"] for r in records])
all_tok = np.array([r["n_tokens"] for r in records])
all_steps = np.array([r["n_steps"] for r in records])
r_tok = np.corrcoef(all_vb, all_tok)[0, 1]
r_steps = np.corrcoef(all_vb, all_steps)[0, 1]
print(f"\nPearson r(vocab_breadth, n_tokens) across all {len(records)} trajectories: {r_tok:+.3f}")
print(f"Pearson r(vocab_breadth, n_steps)  across all {len(records)} trajectories: {r_steps:+.3f}")

print("\n--- length-normalized fix: Guiraud's index, with full bootstrap CI + permutation test ---")
for r in records:
    r["guiraud"] = (r["vocab_breadth"] * r["n_tokens"]) / max(r["n_tokens"], 1) ** 0.5
auc_ci_perm(hijacked, benign, "guiraud", "hijacked vs benign — Guiraud's index")
all_g = np.array([r["guiraud"] for r in records])
print(f"Pearson r(Guiraud, n_tokens): {np.corrcoef(all_g, all_tok)[0,1]:+.3f}")
