"""
Extend the semantic-diversity complementary check (semantic_diversity_check.py)
to the cross-model data, using ONLY already-collected queries -- no new
Tavily search calls, just cheap OpenAI embeddings. Motivated directly by
tonight's data-integrity correction: computed on CLEAN-ONLY sessions (same
filter as analyze_clean_only.py) so this doesn't reintroduce the same
Tavily-contamination confound into a new feature.
"""
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from scipy.stats import rankdata, pearsonr
from sklearn.metrics import roc_auc_score

load_dotenv(Path.home() / "Desktop/aisec_work/colosseum/.env")
API_KEY = os.environ["OPENAI_API_KEY"]
DIR = Path("/Users/lulu/Desktop/silentdrop/evaluation/crossover_bd/n30_rerun")


def has_error(d):
    return any(
        "exceeds your plan" in s.get("text", "") or "search error" in s.get("text", "")
        for s in d["steps"] if s.get("kind") == "OBSERVATION"
    )


def embed_batch(texts):
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=json.dumps({"model": "text-embedding-3-small", "input": texts}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    return [item["embedding"] for item in d["data"]]


def semantic_breadth(embeddings):
    if len(embeddings) < 2:
        return 0.0
    v = np.array(embeddings)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    sims = v @ v.T
    n = len(v)
    iu = np.triu_indices(n, k=1)
    return float((1 - sims[iu]).mean())


def vocab_breadth(queries):
    u, t = set(), 0
    for q in queries:
        w = q.lower().split()
        u.update(w)
        t += len(w)
    return len(u) / max(t, 1)


def load_clean(path, condition):
    out = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["condition"] != condition or has_error(d):
            continue
        out.append({"queries": d["queries"], "vocab_breadth": vocab_breadth(d["queries"])})
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
    return float(auc), [float(ci[0]), float(ci[1])], float(d), float(pval)


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
    neu = load_clean(path, "neutral")
    inj = load_clean(path, "belief_injected")
    for r in neu + inj:
        r["semantic_breadth"] = semantic_breadth(embed_batch(r["queries"])) if r["queries"] else 0.0

    auc_sb, ci_sb, d_sb, p_sb = full_stat(inj, neu, "semantic_breadth")
    auc_vb, ci_vb, d_vb, p_vb = full_stat(inj, neu, "vocab_breadth")
    all_sb = np.array([r["semantic_breadth"] for r in neu + inj])
    all_vb = np.array([r["vocab_breadth"] for r in neu + inj])
    r_val, p_corr = pearsonr(all_sb, all_vb) if len(set(all_vb)) > 1 else (float("nan"), float("nan"))

    print(f"\n{label} (clean n={len(neu)}+{len(inj)}):")
    print(f"  semantic_breadth AUC={auc_sb:.4f} 95%CI={ci_sb} d={d_sb:+.3f} p={p_sb:.5f}")
    print(f"  vocab_breadth    AUC={auc_vb:.4f} (reference, clean-only)")
    print(f"  Pearson r(semantic, lexical) = {r_val:+.3f} (p={p_corr:.4f})")

    results[label] = dict(
        semantic_breadth=dict(auc=auc_sb, ci=ci_sb, d=d_sb, p=p_sb),
        vocab_breadth_clean=dict(auc=auc_vb, d=d_vb, p=p_vb),
        pearson_r=float(r_val) if r_val == r_val else None,
    )

Path("semantic_diversity_crossmodel_results.json").write_text(json.dumps(results, indent=2))
print("\nSaved to semantic_diversity_crossmodel_results.json")
