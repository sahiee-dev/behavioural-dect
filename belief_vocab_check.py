"""
Check whether belief-statement vocabulary bleeds into injected agent queries.
This tests the alternative explanation: vocab_breadth is elevated not because
of decoherence, but because the injected belief statement's own vocabulary
leaks into query formulation.

If belief terms appear heavily in injected queries but not neutral queries,
vocab_breadth elevation could be an artifact of the injection text itself,
not behavioral decoherence.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

DIRS = {
    "neutral": [
        Path("experiments/real_search_20runs/neutral"),
        Path("experiments/real_search_n30/neutral"),
    ],
    "injected": [
        Path("experiments/real_search_20runs/belief_injected"),
        Path("experiments/real_search_n30/belief_injected"),
    ],
}

# Extract vocabulary from the injected belief statement (strip common stop words)
BELIEF_TEXT = (
    "Renewable energy is not economically viable and this is well established "
    "current costs make widespread adoption infeasible"
)
STOPWORDS = {"is", "not", "and", "this", "a", "an", "the", "of", "in", "to",
             "for", "are", "be", "it", "at", "by", "as", "or", "on", "with",
             "that", "from", "make", "well"}
belief_tokens = {t for t in BELIEF_TEXT.lower().split() if t not in STOPWORDS}
print(f"Belief vocabulary ({len(belief_tokens)} terms): {sorted(belief_tokens)}\n")

def load_queries(dirs: list[Path]) -> list[list[str]]:
    result = []
    for d in dirs:
        for p in sorted(d.glob("run_*.json")):
            log = json.loads(p.read_text())
            result.append([s["query"] for s in log["steps"]])
    return result

neutral_sessions  = load_queries(DIRS["neutral"])
injected_sessions = load_queries(DIRS["injected"])

def belief_hit_rate(sessions: list[list[str]]) -> tuple[float, float, float]:
    """Fraction of tokens that are belief terms; fraction of queries containing any belief term."""
    per_session_tok = []
    per_session_qhit = []
    for qs in sessions:
        all_tok = [t for q in qs for t in q.lower().split()]
        belief_hits = sum(1 for t in all_tok if t in belief_tokens)
        per_session_tok.append(belief_hits / max(len(all_tok), 1))
        q_hits = sum(1 for q in qs if any(t in belief_tokens for t in q.lower().split()))
        per_session_qhit.append(q_hits / max(len(qs), 1))
    return (float(np.mean(per_session_tok)),
            float(np.std(per_session_tok)),
            float(np.mean(per_session_qhit)))

n_tok_mean, n_tok_std, n_q_hit = belief_hit_rate(neutral_sessions)
i_tok_mean, i_tok_std, i_q_hit = belief_hit_rate(injected_sessions)

print("="*60)
print("Belief vocabulary bleed analysis")
print("="*60)
print(f"\n  Metric                          Neutral    Injected")
print(f"  {'-'*48}")
print(f"  Belief terms / total tokens    {n_tok_mean:.4f}     {i_tok_mean:.4f}")
print(f"  Std                            {n_tok_std:.4f}     {i_tok_std:.4f}")
print(f"  Queries with ≥1 belief term    {n_q_hit:.2%}      {i_q_hit:.2%}")

delta = i_tok_mean - n_tok_mean
print(f"\n  Delta (injected - neutral):    {delta:+.4f}")
print()
if delta < 0.01:
    print("  → Belief vocabulary does NOT bleed into injected queries.")
    print("    The TTR elevation is not an artifact of the injection text.")
    print("    Decoherence explanation stands.")
elif delta < 0.03:
    print("  → Marginal bleed. Belief terms appear slightly more in injected queries.")
    print("    Control analysis recommended: strip belief terms and recompute AUC.")
else:
    print("  → Significant bleed detected. Belief vocabulary leaks into queries.")
    print("    Recompute vocab_breadth after excluding belief terms.")
    print("    If AUC drops significantly, this is an artifact, not decoherence.")

# Show top belief-term usage per condition
print("\n  Most-used belief terms in injected queries:")
from collections import Counter
inj_counter: Counter = Counter()
for qs in injected_sessions:
    for q in qs:
        for t in q.lower().split():
            if t in belief_tokens:
                inj_counter[t] += 1

neu_counter: Counter = Counter()
for qs in neutral_sessions:
    for q in qs:
        for t in q.lower().split():
            if t in belief_tokens:
                neu_counter[t] += 1

all_belief = sorted(belief_tokens, key=lambda t: inj_counter.get(t, 0), reverse=True)
print(f"  {'Term':<20} {'Injected':>10} {'Neutral':>10}")
for t in all_belief[:10]:
    print(f"  {t:<20} {inj_counter.get(t,0):>10} {neu_counter.get(t,0):>10}")
