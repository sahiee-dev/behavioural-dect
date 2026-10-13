"""
Analysis for the task-irrelevant belief injection scoping test (§5.10).
Compares Claude Task 1's neutral baseline (n=30, already-established data)
against a new condition: task-irrelevant belief injection (belief about
Mars colonization risk, while researching renewable energy in Europe --
same TASK/NEUTRAL_PROMPT as Task 1), matching Jeong et al.'s own paradigm
rather than this project's usual task-relevant design.

Data collection: injector_task_irrelevant.py --n 30 (run overnight once a
working backup Tavily key was found; the primary key and key_1 are hard-
blocked by a plan usage limit discovered earlier the same night).
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

BD = Path(__file__).resolve().parents[2]
TI_DIR = Path(__file__).parent / "task_irrelevant_logs" / "task_irrelevant_injected"


def vocab_breadth(queries):
    u, t = set(), 0
    for q in queries:
        w = q.lower().split()
        u.update(w)
        t += len(w)
    return len(u) / max(t, 1), t, len(queries)


def load_claude_dir(path):
    out = []
    for p in sorted(path.glob("run_*.json")):
        log = json.loads(p.read_text())
        queries = [s["query"] for s in log["steps"] if s.get("action") == "web_search"]
        vb, nt, nq = vocab_breadth(queries)
        out.append({"vocab_breadth": vb, "n_tokens": nt, "n_queries": nq})
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


neutral = load_claude_dir(BD / "experiments/real_search_20runs/neutral") + load_claude_dir(BD / "experiments/real_search_n30/neutral")
task_irrelevant = load_claude_dir(TI_DIR)

print(f"neutral n={len(neutral)}  task-irrelevant-injected n={len(task_irrelevant)}")

for feat in ("vocab_breadth", "n_queries"):
    auc, ci, d, p = full_stat(task_irrelevant, neutral, feat)
    print(f"{feat:16s} AUC={auc:.4f}  95% CI [{ci[0]:.4f},{ci[1]:.4f}]  d={d:+.3f}  p={p:.5f}")

for r in neutral + task_irrelevant:
    r["guiraud"] = (r["vocab_breadth"] * r["n_tokens"]) / max(r["n_tokens"], 1) ** 0.5
auc, ci, d, p = full_stat(task_irrelevant, neutral, "guiraud")
print(f"guiraud          AUC={auc:.4f}  95% CI [{ci[0]:.4f},{ci[1]:.4f}]  d={d:+.3f}  p={p:.5f}")

for label, recs in [("neutral", neutral), ("task-irrelevant-injected", task_irrelevant)]:
    vb = [r["vocab_breadth"] for r in recs]
    nq = [r["n_queries"] for r in recs]
    print(f"{label}: vocab_breadth mean={np.mean(vb):.4f} sd={np.std(vb):.4f} | "
          f"n_queries mean={np.mean(nq):.2f} sd={np.std(nq):.2f} min={min(nq)} max={max(nq)}")
