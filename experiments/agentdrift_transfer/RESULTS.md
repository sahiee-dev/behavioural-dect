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

## Full project-wide length audit (Task 2, Task 3, injection strength, all 8 cross-model points)

Script: `full_length_audit.py` (run: `python3 full_length_audit.py`, no args).
Full per-dataset bootstrap CI / permutation p-value output in
`full_length_audit_results.json`. Summary table and the cross-model
query-count-mediation finding (r=−0.758, p=0.030, n=8) are written up in
`../../README.md` ("Full length-confound audit") and `../../paper_draft.md`
(§5.4, §5.5, §5.6, §5.7, §6.1, §6.3, Conclusion) — not duplicated here in
full to avoid drift between three copies of the same numbers.

One number worth flagging here that isn't in the main writeup: within
Task 2 alone, Pearson r(vocab_breadth, n_queries) = −0.094 — much weaker
than Task 1's −0.585, despite both showing a significant raw n_queries
effect. This is consistent with Task 2's reversal being driven by the
templated-task structure itself (both conditions converge to the same
query template) rather than a clean length-vs-diversity trade-off like
Task 1's.

## Qwen correction: 3x sample did not confirm the n=30 reclassification (overnight follow-up)

Collected 60 additional runs/condition for all three Qwen models (n=30→90
total/condition) specifically to test whether the Guiraud/MATTR-based
reclassifications reported above were real or small-sample noise. Raw
data stays local-only per project convention (`~/Desktop/silentdrop/evaluation/crossover_bd/n30_rerun/batch2_overnight/`,
not in this repo); `analyze_extended_qwen.py` and `extended_qwen_results.json`
here are the analysis script and aggregate results only (paths inside the
script point to the local-only location and won't resolve if run from here
without adjustment).

**Result: Qwen2.5-32B's reclassification did not replicate.** Guiraud AUC
fell from 0.886 (n=30, significant) to 0.568 (n=90, p=0.116, not
significant) — regression to the mean for what was substantially a
small-sample artifact. qwen2.5:7b's raw vocab_breadth point estimate even
flipped sign between n=30 (AUC=0.363) and n=90 (AUC=0.563). None of the 9
new tests (3 models x {raw, Guiraud, MATTR}) survive Holm-Bonferroni
correction except qwen2.5:14b's MATTR result — which is the one already
flagged as carrying a known window-calibration artifact (d≈0 with a
significant AUC, both at n=30 and n=90).

**The cross-model query-count mediation correlation survives this
correction essentially unchanged:** r=−0.746 (p=0.034) using the updated
n=90 Qwen effect sizes, versus r=−0.758 (p=0.030) with the original n=30
values. The mediation finding does not depend on the specific Qwen
reclassification that is now retracted.

**Lesson applied, not just stated:** a surprising small-n robustness-check
result (the Guiraud/MATTR reclassification) was treated as provisional and
tested against more data in the same session, rather than left standing.
It didn't hold up, and that correction is reported in the same place as
the original finding — paper_draft.md §5.7 and Conclusion, both updated
same night.

## THE biggest finding of the night: cross-model data-integrity correction

While the Qwen/Mistral/Gemma extension batches above were running, Tavily's
API started returning "exceeds your plan's set usage limit" on every
request. Checking whether this had ALSO silently happened during the
ORIGINAL n=30 collection (not just tonight's new attempts) found that it
had: 5 of 7 non-Claude cross-model files contain search-error observations
(1.2%-15% of observations), and for 3 models — qwen2.5:14b, mistral-small:24b,
GPT-4o — the belief_injected condition was hit meaningfully harder than
neutral (e.g. GPT-4o: 7.8% vs 0%). Claude's own data (Tasks 1-3) is 100%
clean throughout.

Error-contaminated sessions show dramatically lower vocab_breadth than
clean ones (pooled across all 7 files: AUC=0.061, p<0.0001, n=347 clean /
73 contaminated) — mechanically, an agent fed error text instead of real
search results has nothing new to branch into, so it repeats itself.

Recomputing each model with contaminated sessions removed
(`analyze_clean_only.py`, `clean_only_results.json`):

| Model | Original AUC (d, p) | Clean-only AUC (d, p) | n (clean) |
|---|---|---|---|
| GPT-4o | 0.741 (+0.963, 0.00061) | 0.855 (+1.644, <0.00001) | 30+23 |
| Gemini 2.5 Flash | 0.669 (+0.688, 0.024) | 0.669 (+0.688, 0.024) | 30+30 |
| qwen2.5:32b | 0.588 (+0.416, 0.247) | 0.595 (+0.574, 0.219) | 29+29 |
| qwen2.5:14b | 0.436 (−0.233, 0.164) | **0.696 (+0.829, 0.044)** | 28+13 |
| qwen2.5:7b | 0.372 (−0.377, 0.071) | 0.314 (−0.434, 0.072) | 17+15 |
| mistral-small:24b | 0.336 (−0.582, 0.017) | **0.416 (−0.308, 0.360)** | 26+17 |
| gemma2:27b | 0.316 (−0.626, 0.010) | 0.316 (−0.626, 0.010) | 30+30 |

**Corrected cross-model headline:** of the models reaching significance,
four agree with Claude's direction (Claude, GPT-4o, Gemini, Qwen2.5-14B);
only Gemma2-27B shows a robust reversal. Mistral-Small-24B's reversal —
one of only two this entire project has leaned on throughout — does not
survive. Recomputing the query-count mediation correlation with these
clean values: r=−0.614, p=0.106, n=8 — no longer significant (was
r=−0.758, p=0.030 originally; r=−0.746, p=0.034 after the Qwen-sample
correction above).

**Caveats on this correction:** clean-only filtering is not a random
subsample (sessions that searched less had fewer chances to hit the quota
wall), and n is smaller (as low as 13 for qwen2.5:14b injected). Genuinely
fresh data, blocked tonight by Tavily's exhausted quota, is the right
follow-up, not a permanent substitute. The Mistral/Gemma n=60 extension
batch running on the pod when this was discovered was killed immediately
(also contaminated, data discarded, nothing salvageable or worth keeping).
