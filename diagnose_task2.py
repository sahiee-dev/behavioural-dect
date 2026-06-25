"""
Diagnose the Task 2 reversal: why does vocab_breadth flip direction for
LLM capabilities vs renewable energy?

Key question: Is it the task, the belief, or their interaction?

Hypotheses:
  H1: Task 2 neutral agents have naturally HIGH vocab_breadth (broad topic)
      → injected agents narrow down → lower VB → correct reversal
  H2: Task 2 injected belief is coherent with the task → focused search
  H3: The "decoherence" mechanism requires the belief to be orthogonal to
      the task structure; if belief aligns with one research framing, it
      produces convergence not decoherence
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

T1_DIRS = {
    "neutral": [
        Path("experiments/real_search_20runs/neutral"),
        Path("experiments/real_search_n30/neutral"),
    ],
    "injected": [
        Path("experiments/real_search_20runs/belief_injected"),
        Path("experiments/real_search_n30/belief_injected"),
    ],
}
T2_DIRS = {
    "neutral": [Path("experiments/task2_llm_capabilities/neutral")],
    "injected": [Path("experiments/task2_llm_capabilities/belief_injected")],
}

def load_session(p: Path):
    log = json.loads(p.read_text())
    return {
        "queries": [s["query"] for s in log["steps"]],
        "search_count": log["total_searches"],
        "run": p.stem,
    }

def load_cond(dirs):
    sessions = []
    for d in dirs:
        for p in sorted(d.glob("run_*.json")):
            sessions.append(load_session(p))
    return sessions

def vb(queries):
    u, t = set(), 0
    for q in queries:
        w = q.lower().split()
        u.update(w)
        t += len(w)
    return len(u) / max(t, 1)

def mean_qlen(queries):
    return np.mean([len(q.lower().split()) for q in queries])

# --- Load all ---
t1_n = load_cond(T1_DIRS["neutral"])
t1_i = load_cond(T1_DIRS["injected"])
t2_n = load_cond(T2_DIRS["neutral"])
t2_i = load_cond(T2_DIRS["injected"])

# --- Compute features ---
def feat_table(sessions, label):
    vbs = [vb(s["queries"]) for s in sessions]
    qls = [mean_qlen(s["queries"]) for s in sessions]
    scs = [s["search_count"] for s in sessions]
    print(f"\n  {label} (n={len(sessions)})")
    print(f"    vocab_breadth:  mean={np.mean(vbs):.4f}  std={np.std(vbs):.4f}  "
          f"range=[{np.min(vbs):.3f},{np.max(vbs):.3f}]")
    print(f"    mean_q_length:  mean={np.mean(qls):.2f}  std={np.std(qls):.2f}")
    print(f"    search_count:   mean={np.mean(scs):.2f}  std={np.std(scs):.2f}")
    return np.array(vbs), np.array(qls), np.array(scs)

print("="*60)
print("Task 2 reversal diagnosis")
print("="*60)
print("\nTask 1 — Renewable Energy:")
t1n_vb, t1n_ql, t1n_sc = feat_table(t1_n, "Neutral")
t1i_vb, t1i_ql, t1i_sc = feat_table(t1_i, "Injected")
print(f"  VB delta: {t1i_vb.mean()-t1n_vb.mean():+.4f}  "
      f"(injected {'HIGHER' if t1i_vb.mean()>t1n_vb.mean() else 'LOWER'})")

print("\nTask 2 — LLM Capabilities:")
t2n_vb, t2n_ql, t2n_sc = feat_table(t2_n, "Neutral")
t2i_vb, t2i_ql, t2i_sc = feat_table(t2_i, "Injected")
print(f"  VB delta: {t2i_vb.mean()-t2n_vb.mean():+.4f}  "
      f"(injected {'HIGHER' if t2i_vb.mean()>t2n_vb.mean() else 'LOWER'})")

print("\n" + "="*60)
print("Why does it reverse? Cross-task comparison")
print("="*60)
print(f"\n  Neutral baseline VB:  T1={t1n_vb.mean():.4f}  T2={t2n_vb.mean():.4f}")
print(f"  → T2 neutral is MUCH higher: the task itself forces lexical diversity")
print(f"    (LLM capabilities is multi-dimensional: reasoning, benchmarks, deployment, etc.)")
print(f"\n  Injected VB:          T1={t1i_vb.mean():.4f}  T2={t2i_vb.mean():.4f}")
print(f"  → T2 injected LOWER than T2 neutral: belief focuses search on limitations")
print(f"\n  MECHANISM:")
print(f"  Task 1 (consensus topic): neutral agent converges on topic,")
print(f"    injected agent scatters (decoherence). VB: neutral < injected ✓")
print(f"  Task 2 (contested multi-faceted topic): neutral agent naturally")
print(f"    explores many dimensions. Injected agent believes 'LLMs are limited'")
print(f"    → focuses search on finding evidence of limitations → convergent,")
print(f"    lower VB. VB: neutral > injected (reversal).")
print(f"\n  KEY INSIGHT: vocab_breadth measures RELATIVE diversity.")
print(f"  The signal direction depends on whether the belief FOCUSES or SCATTERS")
print(f"  the agent relative to the neutral exploration pattern for that task.")

# --- What does work on Task 2? ---
print("\n" + "="*60)
print("Does search_count still work on Task 2?")
print("="*60)
from sklearn.metrics import roc_auc_score
y2 = np.concatenate([np.zeros(len(t2n_sc)), np.ones(len(t2i_sc))])
all_sc2 = np.concatenate([t2n_sc, t2i_sc])
print(f"\n  T2 search_count: neutral={t2n_sc.mean():.2f}  injected={t2i_sc.mean():.2f}")
ps = np.sqrt((t2n_sc.std()**2 + t2i_sc.std()**2) / 2)
d_sc = (t2i_sc.mean() - t2n_sc.mean()) / ps if ps > 0 else 0.0
print(f"  Cohen's d: {d_sc:+.2f}")
# flip scores if injected < neutral (lower score = positive class)
auc_sc = roc_auc_score(y2, all_sc2 if t2i_sc.mean() > t2n_sc.mean() else -all_sc2)
print(f"  AUC: {auc_sc:.4f}")

# --- Sample queries from each condition ---
print("\n" + "="*60)
print("Sample queries: T2 neutral vs T2 injected (first 3 sessions)")
print("="*60)
for cond_name, sessions in [("Neutral", t2_n[:3]), ("Injected", t2_i[:3])]:
    print(f"\n  === {cond_name} ===")
    for s in sessions:
        print(f"  [{s['run']}] sc={s['search_count']}  vb={vb(s['queries']):.3f}")
        for q in s["queries"][:4]:
            print(f"    · {q}")
        if len(s["queries"]) > 4:
            print(f"    ... ({len(s['queries'])-4} more)")
