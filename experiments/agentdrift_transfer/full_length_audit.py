"""
Full length-covariate audit across EVERY dataset in this project, not just
Task 1. Motivated by the confirmed confound on AgentDrift and on Task 1.

Datasets covered:
  - Task 1 (Claude, renewable energy, n=30/condition) -- reference, already done
  - Task 2 (Claude, LLM capabilities boundary condition, n=20/condition)
  - Task 3 (Claude, UBI economics, n=20/condition)
  - Injection-strength sweep (Claude, weak n=20, strong n=20; "medium" = Task 1)
  - All 7 cross-model datasets (Qwen 7b/14b/32b, Gemma2-27b, Mistral-Small-24b,
    GPT-4o, Gemini 2.5 Flash), n=30/condition each, from silentdrop's crossover_bd

For each: AUC/CI/d/p for vocab_breadth (reference), n_queries alone (the
length-confound check), and Guiraud's index (length-normalized fix),
plus Pearson r(vocab_breadth, n_queries).

Then the big question: across all 8 models tested on Task 1's design
(Claude + 7 others), does the SIZE of the query-count reduction under
injection (d for n_queries) predict the SIZE of the vocab_breadth increase
(d for vocab_breadth)? If yes, the closed-vs-open split may be partly or
wholly mediated by a measurable covariate (how much each model cuts its
search under belief) rather than an unexplained alignment black box.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, pearsonr
from sklearn.metrics import roc_auc_score

BD = Path("/Users/lulu/agentops-replay-pro/behavioral_detector")
CROSS = Path("/Users/lulu/Desktop/silentdrop/evaluation/crossover_bd/n30_rerun")


def vb_len(queries):
    u, t = set(), 0
    for q in queries:
        w = q.lower().split()
        u.update(w)
        t += len(w)
    return len(u) / max(t, 1), t, len(queries)


def load_claude_dir(path: Path):
    out = []
    for p in sorted(path.glob("run_*.json")):
        log = json.loads(p.read_text())
        queries = [s["query"] for s in log["steps"]]
        vb, nt, nq = vb_len(queries)
        out.append({"vocab_breadth": vb, "n_tokens": nt, "n_queries": nq})
    return out


def load_jsonl(path: Path, condition: str):
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["condition"] != condition:
            continue
        vb, nt, nq = vb_len(d["queries"])
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


def guiraud_field(records):
    for r in records:
        r["guiraud"] = (r["vocab_breadth"] * r["n_tokens"]) / max(r["n_tokens"], 1) ** 0.5


def report(name, neutral, injected):
    guiraud_field(neutral)
    guiraud_field(injected)
    all_vb = np.array([r["vocab_breadth"] for r in neutral + injected])
    all_nq = np.array([r["n_queries"] for r in neutral + injected])
    r_nq = pearsonr(all_vb, all_nq)[0] if len(set(all_nq)) > 1 else float("nan")

    print(f"\n{'='*70}\n{name}  (n={len(neutral)}+{len(injected)})\n{'='*70}")
    results = {}
    for feat in ("vocab_breadth", "n_queries", "guiraud"):
        auc, ci, d, p = full_stat(injected, neutral, feat)
        results[feat] = dict(auc=auc, ci=ci, d=d, p=p)
        print(f"  {feat:16s} AUC={auc:.4f}  95% CI [{ci[0]:.4f},{ci[1]:.4f}]  d={d:+.3f}  p={p:.5f}")
    print(f"  Pearson r(vocab_breadth, n_queries) = {r_nq:+.3f}")
    results["r_vb_nq"] = r_nq
    return results


all_results = {}

# --- Task 1 (reference, re-derived for completeness) ---
n1 = load_claude_dir(BD / "experiments/real_search_20runs/neutral") + load_claude_dir(BD / "experiments/real_search_n30/neutral")
i1 = load_claude_dir(BD / "experiments/real_search_20runs/belief_injected") + load_claude_dir(BD / "experiments/real_search_n30/belief_injected")
all_results["Task1_Claude_renewable"] = report("Task 1 — Claude, renewable energy (reference)", n1, i1)

# --- Task 2 ---
n2 = load_claude_dir(BD / "experiments/task2_llm_capabilities/neutral")
i2 = load_claude_dir(BD / "experiments/task2_llm_capabilities/belief_injected")
all_results["Task2_Claude_llm_capabilities"] = report("Task 2 — Claude, LLM capabilities (boundary condition)", n2, i2)

# --- Task 3 ---
n3 = load_claude_dir(BD / "experiments/task3_ubi/neutral")
i3 = load_claude_dir(BD / "experiments/task3_ubi/belief_injected")
all_results["Task3_Claude_UBI"] = report("Task 3 — Claude, UBI economics (cross-domain)", n3, i3)

# --- Injection strength: weak / strong (medium = Task 1) ---
for strength in ("weak", "strong"):
    d = BD / f"experiments/injection_strength/{strength}"
    if (d).exists():
        inj = load_claude_dir(d)
        # strength sweep compares against the SAME neutral baseline as Task 1 (same task/domain)
        all_results[f"Strength_{strength}_vs_neutral"] = report(
            f"Injection strength — {strength} vs. Task 1 neutral baseline", n1, inj)

# --- Cross-model (7 models, n=30/condition each) ---
cross_model_results = {}
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
    if not path.exists():
        print(f"MISSING: {path}")
        continue
    neu = load_jsonl(path, "neutral")
    inj = load_jsonl(path, "belief_injected") or load_jsonl(path, "injected")
    if not inj:
        # discover actual condition label
        conds = {json.loads(l)["condition"] for l in path.read_text().splitlines() if l.strip()}
        other = [c for c in conds if c != "neutral"][0]
        inj = load_jsonl(path, other)
    res = report(f"Cross-model — {label}", neu, inj)
    all_results[f"CrossModel_{label}"] = res
    cross_model_results[label] = res

# Add Claude itself to the cross-model comparison set
cross_model_results["Claude Sonnet 4.6"] = all_results["Task1_Claude_renewable"]

print(f"\n{'='*70}\nTHE BIG QUESTION: does query-count reduction (d for n_queries) predict\nvocab_breadth increase (d for vocab_breadth) across all 8 models?\n{'='*70}")
labels = list(cross_model_results.keys())
d_nq = np.array([cross_model_results[l]["n_queries"]["d"] for l in labels])
d_vb = np.array([cross_model_results[l]["vocab_breadth"]["d"] for l in labels])
for l, a, b in zip(labels, d_nq, d_vb):
    print(f"  {l:20s}  d(n_queries)={a:+.3f}   d(vocab_breadth)={b:+.3f}")

if len(labels) >= 3:
    r, p = pearsonr(d_nq, d_vb)
    print(f"\n  Pearson r(d_n_queries, d_vocab_breadth) across {len(labels)} models = {r:+.3f}  (p={p:.4f})")

# Save everything to disk for the writeup
def jsonable(obj):
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj

out_path = Path(__file__).parent / "full_length_audit_results.json"
out_path.write_text(json.dumps(jsonable(all_results), indent=2))
print(f"\nSaved full results to {out_path}")
