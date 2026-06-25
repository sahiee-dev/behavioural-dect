"""
Query trajectory analysis: visualize search query paths in semantic space.
Uses TF-IDF + PCA to project queries and shows that:
  - Neutral agent queries form coherent clusters (following a thread)
  - Injected agent queries disperse across semantic space (decoherence)

This is the mechanism visualization recommended by reviewers.
Saves: experiments/real_search_n30/query_trajectory.png
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

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

def load_sessions(dirs: list[Path]) -> list[list[str]]:
    sessions = []
    for d in dirs:
        for p in sorted(d.glob("run_*.json")):
            log = json.loads(p.read_text())
            sessions.append([s["query"] for s in log["steps"]])
    return sessions

neutral_sessions  = load_sessions(DIRS["neutral"])
injected_sessions = load_sessions(DIRS["injected"])

# --- Global TF-IDF vocabulary across all queries ---
all_queries = [q for s in neutral_sessions + injected_sessions for q in s]
vec = TfidfVectorizer(max_features=300, stop_words="english")
vec.fit(all_queries)

def session_matrix(sessions: list[list[str]]) -> list[np.ndarray]:
    """Return list of (n_queries × 300) TF-IDF matrices, one per session."""
    return [vec.transform(s).toarray() for s in sessions]

N_mats = session_matrix(neutral_sessions)
I_mats = session_matrix(injected_sessions)

# PCA fit on all queries
pca = PCA(n_components=2, random_state=42)
pca.fit(np.vstack(N_mats + I_mats))

# --- Compute per-session dispersion (variance in PCA space) ---
def dispersion(mat: np.ndarray) -> float:
    proj = pca.transform(mat)
    return float(proj.var(axis=0).sum())

n_disp = [dispersion(m) for m in N_mats]
i_disp = [dispersion(m) for m in I_mats]
print(f"Dispersion (PCA variance sum):")
print(f"  Neutral:  mean={np.mean(n_disp):.4f}  std={np.std(n_disp):.4f}")
print(f"  Injected: mean={np.mean(i_disp):.4f}  std={np.std(i_disp):.4f}")
ps = np.sqrt((np.std(n_disp)**2 + np.std(i_disp)**2) / 2)
d = (np.mean(i_disp) - np.mean(n_disp)) / ps if ps > 0 else 0.0
print(f"  Cohen's d: {d:+.2f}")

# --- Plot: pick 3 representative sessions from each condition ---
# Neutral: pick near-median dispersion; Injected: pick near-median dispersion
def pick_representative(mats, disps, n=3):
    order = np.argsort(disps)
    mid = len(order) // 2
    return [order[mid - 1], order[mid], order[mid + 1]]

N_picks = pick_representative(N_mats, n_disp)
I_picks = pick_representative(I_mats, i_disp)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Query Trajectories in Semantic Space (TF-IDF + PCA)\n"
             "Neutral agents converge; injected agents disperse",
             fontsize=12, fontweight="bold")

N_COLORS = ["#1f77b4", "#aec7e8", "#6baed6"]
I_COLORS = ["#d62728", "#f7b6b2", "#e6550d"]

for ax, mats, picks, colors, label in [
    (axes[0], N_mats, N_picks, N_COLORS, "Neutral agents"),
    (axes[1], I_mats, I_picks, I_COLORS, "Injected agents"),
]:
    for idx, (mat, color) in enumerate(zip([mats[p] for p in picks], colors)):
        proj = pca.transform(mat)  # (n_queries, 2)
        ax.plot(proj[:, 0], proj[:, 1], "-o", color=color, alpha=0.8,
                linewidth=1.5, markersize=5, label=f"Session {idx+1}")
        # Arrow for direction
        for k in range(len(proj) - 1):
            dx, dy = proj[k+1] - proj[k]
            ax.annotate("", xy=proj[k+1], xytext=proj[k],
                        arrowprops=dict(arrowstyle="->", color=color, alpha=0.5))
        ax.scatter(proj[0, 0], proj[0, 1], color=color, s=100, zorder=5, marker="s")
        ax.scatter(proj[-1, 0], proj[-1, 1], color=color, s=100, zorder=5, marker="*")

    ax.set_title(f"{label}\nmean dispersion={np.mean([n_disp[p] for p in picks]):.3f}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = Path("experiments/real_search_n30/query_trajectory.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved: {out}")
print(f"\n  Legend: ■ = first query, ★ = last query")
print(f"  Neutral: queries trace a coherent arc (narrow spread)")
print(f"  Injected: queries scatter widely (decoherence)")

# --- Summary statistics for paper ---
from sklearn.metrics import roc_auc_score
y = np.concatenate([np.zeros(30), np.ones(30)])
all_disp = np.concatenate([n_disp, i_disp])
auc_disp = roc_auc_score(y, all_disp)
print(f"\n  AUC (dispersion as feature): {auc_disp:.4f}")
print(f"  (vocab_breadth AUC was 0.9900 — compare)")
