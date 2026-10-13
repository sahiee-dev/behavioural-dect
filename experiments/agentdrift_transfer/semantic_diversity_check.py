"""
Semantic-diversity complementary feature check, recommended by the Oct 2026
literature survey: vocab_breadth measures lexical (word-form) diversity,
which could diverge from genuine exploratory (meaning-level) diversity in
either direction -- an agent could issue lexically diverse but semantically
redundant queries (inflating vocab_breadth without real diversity), or
lexically repetitive but semantically diverse queries (the reverse).

Feature: "semantic_breadth" = mean pairwise cosine distance between
OpenAI text-embedding-3-small embeddings of all queries in a session.
Higher = queries are more scattered in meaning space, analogous to what
vocab_breadth measures lexically.

Run on Task 1 Claude data first (the mechanism the detector was built for).
Cost: ~$0.02/1M tokens, this dataset is a few thousand tokens total -- cents.
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
BD = Path("/Users/lulu/agentops-replay-pro/behavioral_detector")


def embed_batch(texts: list[str]) -> list[list[float]]:
    req = urllib.request.Request(
        "https://api.openai.com/v1/embeddings",
        data=json.dumps({"model": "text-embedding-3-small", "input": texts}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    return [item["embedding"] for item in d["data"]]


def semantic_breadth(embeddings: list[list[float]]) -> float:
    if len(embeddings) < 2:
        return 0.0
    v = np.array(embeddings)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    sims = v @ v.T
    n = len(v)
    iu = np.triu_indices(n, k=1)
    cos_dist = 1 - sims[iu]
    return float(cos_dist.mean())


def vocab_breadth(queries: list[str]) -> float:
    u, t = set(), 0
    for q in queries:
        w = q.lower().split()
        u.update(w)
        t += len(w)
    return len(u) / max(t, 1)


def load_claude_dir(path: Path) -> list[dict]:
    out = []
    for p in sorted(path.glob("run_*.json")):
        log = json.loads(p.read_text())
        queries = [s["query"] for s in log["steps"]]
        out.append({"queries": queries, "vocab_breadth": vocab_breadth(queries)})
    return out


print("Loading Task 1 Claude data...")
neutral = load_claude_dir(BD / "experiments/real_search_20runs/neutral") + load_claude_dir(BD / "experiments/real_search_n30/neutral")
injected = load_claude_dir(BD / "experiments/real_search_20runs/belief_injected") + load_claude_dir(BD / "experiments/real_search_n30/belief_injected")
all_records = neutral + injected
print(f"Loaded {len(neutral)} neutral + {len(injected)} injected = {len(all_records)} runs")

print("Embedding all queries (one API call per run)...")
total_tokens_est = 0
for i, r in enumerate(all_records):
    if not r["queries"]:
        r["semantic_breadth"] = 0.0
        continue
    embs = embed_batch(r["queries"])
    r["semantic_breadth"] = semantic_breadth(embs)
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(all_records)} runs embedded")

out_path = Path(__file__).parent / "semantic_diversity_task1.json"
out_path.write_text(json.dumps(all_records, indent=2))
print(f"Saved raw results to {out_path}")


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

print(f"\n{'='*60}\nTask 1 (Claude): semantic_breadth vs vocab_breadth\n{'='*60}")
auc, ci, d, p = full_stat(injected, neutral, "semantic_breadth")
print(f"semantic_breadth   AUC={auc:.4f}  95% CI [{ci[0]:.4f},{ci[1]:.4f}]  d={d:+.3f}  p={p:.5f}")
auc2, ci2, d2, p2 = full_stat(injected, neutral, "vocab_breadth")
print(f"vocab_breadth      AUC={auc2:.4f}  95% CI [{ci2[0]:.4f},{ci2[1]:.4f}]  d={d2:+.3f}  p={p2:.5f}  (reference)")

all_sb = np.array([r["semantic_breadth"] for r in all_records])
all_vb = np.array([r["vocab_breadth"] for r in all_records])
r_val, p_corr = pearsonr(all_sb, all_vb)
print(f"\nPearson r(semantic_breadth, vocab_breadth) across all {len(all_records)} runs = {r_val:+.3f} (p={p_corr:.4f})")
