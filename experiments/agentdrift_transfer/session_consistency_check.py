"""
Critical check: for every model with both an original (Sept) collection and
a new (Oct, tonight) collection, compare the two sessions' AUC independently
to see whether they agree in direction, or whether (as found for Gemma)
the two sessions disagree -- a session-level instability issue distinct
from the earlier Tavily-contamination issue.
"""
import json
from pathlib import Path

import numpy as np
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


def auc(pos, neg):
    p = np.array(pos, dtype=float)
    n = np.array(neg, dtype=float)
    y = np.concatenate([np.zeros(len(n)), np.ones(len(p))])
    X = np.concatenate([n, p])
    return roc_auc_score(y, X), n.mean(), p.mean(), len(n), len(p)


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
    auc_o, nm_o, im_o, nn_o, ni_o = auc(orig_i, orig_n)
    auc_n, nm_n, im_n, nn_n, ni_n = auc(new_i, new_n)
    agree = (auc_o - 0.5) * (auc_n - 0.5) > 0
    flag = "AGREE" if agree else "*** DISAGREE ***"
    print(f"{label:20s} ORIG(n={nn_o}+{ni_o}) AUC={auc_o:.3f} (neu={nm_o:.3f} inj={im_o:.3f})  |  "
          f"NEW(n={nn_n}+{ni_n}) AUC={auc_n:.3f} (neu={nm_n:.3f} inj={im_n:.3f})  ->  {flag}")
