# AgentDrift cross-mechanism transfer test — results summary

Source: AgentDrift (arXiv:2609.06972), CC BY 4.0, `github.com/Asif-0209/AgentDrift`.
Web-domain subset, `data_taskdisjoint` split, n=2,600 trajectories.
Full writeup: `../../paper_draft.md` §5.8, `../../README.md` "Transfer to a different injection mechanism."

Scripts: `extract_features.py` (feature extraction + world-entity redaction),
`compute_auc.py` (AUC, bootstrap 95% CI B=10,000, permutation test N=100,000).
Run: `python3 extract_features.py <repo_path> features.jsonl && python3 compute_auc.py features.jsonl`

## Headline numbers

| Comparison | n | AUC | 95% CI | Cohen's d | p |
|---|---|---|---|---|---|
| Hijacked vs. benign — vocab_breadth | 2,000 | 0.4322 | [0.4074, 0.4574] | −0.214 | <0.00001 |
| Hijacked vs. benign — n_tokens alone | 2,000 | 0.6225 | [0.5978, 0.6470] | +0.478 | <0.00001 |
| Hijacked vs. benign — n_steps alone | 2,000 | 0.6194 | [0.5934, 0.6443] | +0.457 | <0.00001 |
| Failed attack vs. benign — vocab_breadth | 1,100 | 0.8073 | [0.7806, 0.8335] | +1.232 | <0.00001 |
| Hard negative vs. benign — vocab_breadth | 1,100 | 0.7964 | [0.7658, 0.8247] | +1.177 | <0.00001 |
| Hijacked vs. benign — Guiraud's index (length-norm fix) | 2,000 | 0.5796 | [0.5551, 0.6046] | +0.339 | <0.00001 |

Pearson r(vocab_breadth, n_tokens) = −0.553, r(vocab_breadth, n_steps) = −0.461 (all 2,600 trajectories).
Pearson r(Guiraud, n_tokens) = +0.590 (fix recovers direction, does not remove confound).

## Companion check: same test on the ORIGINAL Claude Task 1 data

Script: `claude_length_check.py` (run: `python3 claude_length_check.py`, no args — paths are relative to the repo root via `__file__`).

| Comparison | n | AUC | 95% CI | d | p |
|---|---|---|---|---|---|
| Injected vs. neutral — vocab_breadth (reference) | 60 | 0.9900 | [0.9689, 1.0000] | +3.914 | <0.00001 |
| Injected vs. neutral — n_steps (query count) alone | 60 | 0.0989 | [0.0283, 0.1889] | −1.746 | <0.00001 |
| Injected vs. neutral — n_tokens alone | 60 | 0.3394 | [0.2061, 0.4906] | −0.720 | 0.03166 |
| Injected vs. neutral — Guiraud's index (length-normalized) | 60 | 0.8456 | [0.7311, 0.9456] | +1.161 | <0.00001 |

Pearson r(vocab_breadth, n_tokens) = −0.283, r(vocab_breadth, n_steps) = −0.585 (all 60 runs).
Pearson r(Guiraud, n_tokens) = +0.559.

## One-line takeaway

On AgentDrift, the signal reverses on successful hijacks, explained by a
confirmed TTR length confound (longer hijacked trajectories → mechanically
lower raw TTR); it recovers the originally expected direction on failed
attacks and hard negatives, which don't add extra hijack steps.

On Claude's own data (the mechanism vocab_breadth was built for), the same
confound is present and significant (query count alone separates conditions,
AUC=0.099) but points the *same* direction as the hypothesized effect, so it
inflates rather than reverses AUC=0.990. Length-normalizing drops it to
AUC=0.846 (95% CI [0.731, 0.946]) — still clearly significant, so the
feature is not purely a length artifact, but the honest headline is "0.990
raw / ≈0.85 length-normalized," not a single unqualified number.

vocab_breadth is mechanism-specific to belief-prefill injection (where it
retains most of its power after the confound is removed), not a
general-purpose detector (where, on AgentDrift, most of the raw signal was
the confound itself).
