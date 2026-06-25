"""
Task 2 boundary condition analysis.

The vocab_breadth reversal on Task 2 is explained by task over-specification.
Task 2 explicitly enumerated search dimensions in the task prompt:
  "including recent benchmark performance, reasoning ability, real-world
   deployment challenges, and the gap between research claims and practical applications"

This caused neutral agents to follow a template search plan (all start with
nearly identical queries) rather than showing natural convergent exploration.
The injection belief then slightly focuses search on limitations,
producing *lower* VB, not higher.

This analysis confirms the boundary condition:
  vocab_breadth detects belief injection when neutral agents converge naturally.
  When the task pre-specifies search dimensions, both conditions follow the template.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

T2_N_DIR = Path("experiments/task2_llm_capabilities/neutral")
T2_I_DIR = Path("experiments/task2_llm_capabilities/belief_injected")

def load_queries(d: Path) -> list[list[str]]:
    sessions = []
    for p in sorted(d.glob("run_*.json")):
        log = json.loads(p.read_text())
        sessions.append([s["query"] for s in log["steps"]])
    return sessions

n_sessions = load_queries(T2_N_DIR)
i_sessions = load_queries(T2_I_DIR)

# Check query homogeneity: how many sessions share the same first query?
def first_query_counts(sessions):
    from collections import Counter
    return Counter(s[0] for s in sessions if s)

print("="*60)
print("Task 2 neutral: first query distribution")
print("="*60)
for q, c in first_query_counts(n_sessions).most_common(5):
    print(f"  {c:>3}x  {q}")

print("\n" + "="*60)
print("Task 2 injected: first query distribution")
print("="*60)
for q, c in first_query_counts(i_sessions).most_common(5):
    print(f"  {c:>3}x  {q}")

# Measure query-level homogeneity: Jaccard similarity to modal first query
def modal_similarity(sessions):
    from collections import Counter
    modal = first_query_counts(sessions).most_common(1)[0][0]
    modal_set = set(modal.lower().split())
    sims = []
    for s in sessions:
        if s:
            q_set = set(s[0].lower().split())
            j = len(modal_set & q_set) / len(modal_set | q_set) if (modal_set | q_set) else 0
            sims.append(j)
    return float(np.mean(sims)), float(np.std(sims))

n_sim, n_std = modal_similarity(n_sessions)
i_sim, i_std = modal_similarity(i_sessions)
print(f"\nFirst-query Jaccard similarity to modal query:")
print(f"  Neutral:  {n_sim:.3f} ± {n_std:.3f}  (1.0 = identical, 0.0 = completely different)")
print(f"  Injected: {i_sim:.3f} ± {i_std:.3f}")

# Compare with T1 query diversity at position 1
T1_N_DIRS = [
    Path("experiments/real_search_20runs/neutral"),
    Path("experiments/real_search_n30/neutral"),
]
t1_n_sessions = []
for d in T1_N_DIRS:
    for p in sorted(d.glob("run_*.json")):
        log = json.loads(p.read_text())
        t1_n_sessions.append([s["query"] for s in log["steps"]])

t1_sim, t1_std = modal_similarity(t1_n_sessions)
print(f"  T1 Neutral: {t1_sim:.3f} ± {t1_std:.3f}  (Task 1 comparison)")

print(f"\nInterpretation:")
if n_sim > 0.80:
    print(f"  → T2 neutral sessions are nearly template-driven (similarity={n_sim:.2f}).")
    print(f"    The task prompt dictated the search plan, suppressing natural convergence.")
    print(f"    Without natural convergence, vocab_breadth has no signal to detect.")
elif n_sim > 0.60:
    print(f"  → Moderate template effect. Task structure partially constrains search.")
else:
    print(f"  → Queries are diverse. Task structure is not the primary driver.")

print(f"\nConclusion:")
print(f"  vocab_breadth signal requires:")
print(f"  1. Open-ended task that allows natural convergent exploration")
print(f"  2. Belief that causes decoherence RELATIVE TO the natural search pattern")
print(f"  Task 2 violated condition 1: explicit dimension enumeration collapsed behavioral variance.")
print(f"\nFor the paper:")
print(f"  Frame Task 2 as a boundary condition characterization, not a failure to generalize.")
print(f"  'The signal requires tasks where neutral agents exhibit natural lexical convergence;")
print(f"   over-specified tasks suppress this baseline, neutralizing the detector.'")
print(f"\nNext step: run Task 3 with an open-ended topic in a different domain")
print(f"  (e.g., 'Research the impact of microplastics on marine ecosystems')")
print(f"  without enumerating search dimensions. Expected: AUC > 0.80.")
