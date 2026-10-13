"""
Recompute cross-model vocab_breadth statistics using ONLY sessions with
zero Tavily search-error observations, to quantify how much the newly
discovered error-contamination confound affected the already-published
n=30 cross-model numbers.

Discovery: Tavily's API is currently hard-blocked ("exceeds your plan's
set usage limit"), and checking the EXISTING n=30 files (already written
up throughout paper_draft.md and README.md) found this had already been
happening during their original collection too: 5 of 7 non-Claude files
have >0% error observations, 3 of them (qwen14b, mistral24b, gpt4o) with
the belief_injected condition hit meaningfully harder than neutral.
Error-contaminated sessions show far lower vocab_breadth than clean ones
(AUC=0.061, i.e. a huge, mechanical effect -- an agent fed repeated error
text instead of real search results has nothing new to branch into, so it
repeats itself, which mechanically lowers TTR). Claude's own data is 100%
clean throughout -- this does not touch the paper's primary result.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

DIR = Path("/Users/lulu/Desktop/silentdrop/evaluation/crossover_bd/n30_rerun")


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


def load(path, condition, clean_only):
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["condition"] != condition:
            continue
        if clean_only and has_error(d):
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


files = {
    "qwen2.5:7b": "qwen7b_renewable_n30.jsonl",
    "qwen2.5:14b": "qwen14b_renewable_n30.jsonl",
    "qwen2.5:32b": "qwen32b_renewable_n30.jsonl",
    "gemma2:27b": "gemma27b_renewable_n30.jsonl",
    "mistral-small:24b": "mistral24b_renewable_n30.jsonl",
    "GPT-4o": "gpt4o_renewable_n30.jsonl",
    "Gemini 2.5 Flash": "gemini_flash_renewable_n30.jsonl",
}

results = {}
for label, fname in files.items():
    path = DIR / fname
    orig_n = load(path, "neutral", clean_only=False)
    orig_i = load(path, "belief_injected", clean_only=False)
    clean_n = load(path, "neutral", clean_only=True)
    clean_i = load(path, "belief_injected", clean_only=True)

    auc_o, ci_o, d_o, p_o = full_stat(orig_i, orig_n)
    auc_c, ci_c, d_c, p_c = full_stat(clean_i, clean_n)

    print(f"\n{label}:")
    print(f"  ORIGINAL (n={len(orig_n)}+{len(orig_i)}, incl. contaminated): AUC={auc_o:.4f} d={d_o:+.3f} p={p_o:.5f}")
    print(f"  CLEAN-ONLY (n={len(clean_n)}+{len(clean_i)}):                 AUC={auc_c:.4f} d={d_c:+.3f} p={p_c:.5f}")
    sign_flip = bool((d_o > 0) != (d_c > 0))
    print(f"  Sign flip: {sign_flip}")
    results[label] = dict(
        original=dict(auc=auc_o, ci=list(ci_o), d=d_o, p=p_o, n_neu=len(orig_n), n_inj=len(orig_i)),
        clean_only=dict(auc=auc_c, ci=list(ci_c), d=d_c, p=p_c, n_neu=len(clean_n), n_inj=len(clean_i)),
        sign_flip=sign_flip,
    )

Path("clean_only_results.json").write_text(json.dumps(results, indent=2))
print("\nSaved to clean_only_results.json")
