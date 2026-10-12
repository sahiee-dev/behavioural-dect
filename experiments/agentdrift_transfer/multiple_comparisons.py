"""
Holm-Bonferroni and Benjamini-Hochberg correction on the 8 cross-model
permutation-test p-values (raw vocab_breadth), per the literature survey's
recommendation that an 8-model scan report a multiple-comparisons-corrected
view alongside the uncorrected per-model p-values.

p-values are the raw vocab_breadth permutation-test results already reported
in paper_draft.md §5.7 (identical to the project's rigor_crossmodel.py output).
"""
import numpy as np

models = ["Claude", "GPT-4o", "Gemini", "gemma2:27b", "mistral-small:24b",
          "qwen2.5:32b", "qwen2.5:7b", "qwen2.5:14b"]
p = np.array([1e-5, 0.00025, 0.00422, 0.010, 0.017, 0.060, 0.099, 0.200])
order = np.argsort(p)
m = len(p)
alpha = 0.05

print("--- Holm-Bonferroni (family-wise error control) ---")
for rank, idx in enumerate(order, start=1):
    thresh = alpha / (m - rank + 1)
    verdict = "REJECT (significant)" if p[idx] <= thresh else "fail to reject"
    print(f"{models[idx]:20s} p={p[idx]:.5f}  threshold={thresh:.5f}  {verdict}")

print("\n--- Benjamini-Hochberg (false discovery rate control) ---")
largest_i = 0
for rank, idx in enumerate(order, start=1):
    thresh = (rank / m) * alpha
    if p[idx] <= thresh:
        largest_i = rank
for rank, idx in enumerate(order, start=1):
    thresh = (rank / m) * alpha
    verdict = "REJECT (significant)" if rank <= largest_i else "fail to reject"
    print(f"{models[idx]:20s} p={p[idx]:.5f}  threshold={thresh:.5f}  {verdict}")
