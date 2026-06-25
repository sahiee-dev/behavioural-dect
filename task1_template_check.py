"""
Check whether Task 1 neutral agents are genuinely exploring vs. following a template.
Reviewer concern: "Maybe neutral agents also template-follow like microplastics?"

If Task 1 neutral queries are diverse across sessions → genuine exploration.
If they're near-identical → same problem as microplastics → signal could be artifact.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from collections import Counter

NEUTRAL_DIRS = [Path("experiments/real_search_20runs/neutral"),
                Path("experiments/real_search_n30/neutral")]

sessions = []
for d in NEUTRAL_DIRS:
    for p in sorted(d.glob("run_*.json")):
        log = json.loads(p.read_text())
        sessions.append([s["query"] for s in log["steps"]])

def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0

def first_query_similarity(sessions):
    modal = Counter(s[0] for s in sessions if s).most_common(1)[0][0]
    modal_set = set(modal.lower().split())
    sims = [jaccard(modal_set, set(s[0].lower().split())) for s in sessions if s]
    return float(np.mean(sims)), float(np.std(sims)), modal

fq_mean, fq_std, modal_q = first_query_similarity(sessions)

print("="*60)
print("Task 1 neutral — template check")
print("="*60)
print(f"\nModal first query: '{modal_q}'")
print(f"First-query Jaccard similarity to modal: {fq_mean:.3f} ± {fq_std:.3f}")

print(f"\nFirst queries across all 30 neutral sessions:")
first_qs = Counter(s[0] for s in sessions if s)
for q, c in first_qs.most_common():
    print(f"  {c:>3}x  {q}")

# Check query-to-query progression: do sessions diverge after the first query?
print(f"\n--- Query diversity by position (mean unique terms per position) ---")
max_len = max(len(s) for s in sessions)
for pos in range(min(6, max_len)):
    qs_at_pos = [s[pos] for s in sessions if len(s) > pos]
    unique_qs = len(set(qs_at_pos))
    print(f"  Position {pos+1}: {unique_qs}/{len(qs_at_pos)} unique queries  "
          f"({unique_qs/len(qs_at_pos)*100:.0f}% unique)")

# Cross-session similarity: average Jaccard between all pairs at same position
print(f"\n--- Cross-session query similarity (lower = more diverse) ---")
for pos in range(min(4, max_len)):
    qs = [set(s[pos].lower().split()) for s in sessions if len(s) > pos]
    if len(qs) < 2: continue
    pairs = [(i,j) for i in range(len(qs)) for j in range(i+1, len(qs))]
    sims = [jaccard(qs[i], qs[j]) for i,j in pairs]
    print(f"  Position {pos+1}: mean similarity = {np.mean(sims):.3f} ± {np.std(sims):.3f}")

print(f"\n--- Compare with microplastics (Task 3 partial, n=8) ---")
T3_DIR = Path("experiments/task3_microplastics/neutral")
if T3_DIR.exists():
    t3_sessions = []
    for p in sorted(T3_DIR.glob("run_*.json")):
        log = json.loads(p.read_text())
        t3_sessions.append([s["query"] for s in log["steps"]])
    t3_fq_mean, t3_fq_std, t3_modal = first_query_similarity(t3_sessions)
    print(f"  Microplastics first-query similarity: {t3_fq_mean:.3f} ± {t3_fq_std:.3f}")
    print(f"  Task 1 first-query similarity:        {fq_mean:.3f} ± {fq_std:.3f}")
    print(f"\n  Interpretation:")
    if fq_mean < t3_fq_mean - 0.05:
        print(f"  Task 1 neutral is MORE diverse than microplastics at position 1.")
        print(f"  But position-level diversity is only part of the story — check positions 2-4.")
    else:
        print(f"  Similar first-query similarity. The difference is in follow-up query divergence.")

print(f"\nConclusion:")
pos1_unique = len(set(s[0] for s in sessions))
print(f"  {pos1_unique}/30 unique first queries in Task 1 neutral.")
if pos1_unique >= 20:
    print(f"  Task 1 neutral agents show genuine query diversity from the start.")
    print(f"  This is NOT template behavior — agents explore different entry points.")
elif pos1_unique >= 10:
    print(f"  Moderate diversity. Agents share starting vocabulary but diverge.")
else:
    print(f"  Low diversity at position 1 — check position 2+ for divergence.")
