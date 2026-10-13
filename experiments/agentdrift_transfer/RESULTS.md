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

## THE actual biggest finding of the night: Gemma's "robust reversal" is session-unstable, not confirmed

After resolving Mistral (above) to a well-powered null using fresh data
(`analyze_combined_mistral.py`), we ran the identical fresh-session check
on Gemma2-27B, since it had been treated as the one confirmed reversal in
the entire cross-model investigation. Result: a fresh n=60/condition
session gives **AUC=0.634, d=+0.552, p=0.011 — independently significant,
in Claude's direction** — versus the original session's **AUC=0.316,
d=−0.626, p=0.013 — independently significant, reversed.** Both sessions
are real, significant effects; they just have opposite signs. Pooling
them (as the original data-integrity correction did) gives AUC=0.520,
p=0.642, which looks like a null but isn't one — it's two genuine,
opposite-signed effects canceling out.

This prompted a full session-by-session consistency check across all 7
non-Claude models (`session_consistency_check.py`, `session_consistency_full.py`):
comparing the original (collected weeks earlier) and new (collected the
same night) sessions independently for every model, not just pooled.

| Model | Original session | New session | Agreement |
|---|---|---|---|
| GPT-4o | AUC=0.855, p<1e-5 (sig) | AUC=0.791, p=0.0001 (sig) | Agree |
| Gemini | AUC=0.669, p=0.024 (sig) | AUC=0.591, p=0.228 (n.s., same dir.) | Agree |
| qwen2.5:14b | AUC=0.696, p=0.045 (sig) | AUC=0.814, p<1e-5 (sig) | Agree |
| qwen2.5:32b | AUC=0.595, p=0.219 (n.s., same dir.) | AUC=0.802, p=0.0002 (sig) | Agree |
| mistral-small:24b | AUC=0.416, p=0.360 (n.s.) | AUC=0.564, p=0.305 (n.s.) | Agree (both null) |
| qwen2.5:7b | AUC=0.314, p=0.072 (n.s., reversed trend) | AUC=0.728, p=0.0002 (sig, Claude-dir.) | **Disagree** |
| gemma2:27b | AUC=0.316, p=0.013 (sig, reversed) | AUC=0.634, p=0.011 (sig, Claude-dir.) | **Disagree** |

**Final, fully corrected cross-model picture: 5 of 7 non-Claude models
give session-reproducible evidence (all agreeing with Claude where any
effect exists); 2 (Gemma severely, Qwen7b more mildly) show a real
effect that flips sign between independent collection sessions weeks
apart.** No model in this investigation has session-stable evidence of
a reversal. We checked for an obvious explanation (query-count pattern
shift, parsing/harness differences, qualitatively odd queries) and found
none — this is reported as a genuine open question, not explained away.
The query-count mediation correlation proposed earlier in the night is
undermined entirely by this finding, since it depended on Gemma's now-
known-unstable original-session value as one of only 8 data points.

**This retracts, explicitly, the "Gemma is the one robust exception"
framing stated with high confidence earlier the same night** (see the
earlier entries in this file and in `paper_draft.md`'s now-superseded
intermediate correction). We report the retraction in the same file as
the claim it retracts, consistent with how every other correction
tonight was handled.

## Final resolution: a third session breaks the tie for both unresolved models

Collected one more independent n=30 session each for gemma2:27b and
qwen2.5:7b (`third_session_resolution.py`) — the single most informative
thing left to do, since two data points split exactly 1-1 tells you
nothing about which, if either, is closer to the truth.

| Model | Session 1 | Session 2 | Session 3 | Pooled (all 3) |
|---|---|---|---|---|
| Gemma2-27B | AUC=0.316, p=0.013 (sig., reversed) | AUC=0.634, p=0.011 (sig., Claude-dir.) | AUC=0.456, p=0.557 (n.s.) | **AUC=0.502, p=0.964, n=120+120** |
| Qwen2.5-7B | AUC=0.314, p=0.072 (n.s.) | AUC=0.728, p=0.0002 (sig., Claude-dir.) | AUC=0.604, p=0.292 (n.s., same dir. as S2) | **AUC=0.619, p=0.011, n=76+74** |

**Gemma2-27B resolves to a precise, well-powered null.** The three
sessions' effect sizes (−0.626, +0.552, −0.073) scatter symmetrically
around zero rather than clustering anywhere — the signature of a true
null with unusually high between-session variance. Both of Gemma's
earlier "significant" findings were false positives, not evidence of a
real effect in either direction.

**Qwen2.5-7B resolves to a significant Claude-direction effect.** 2 of
3 sessions trend the same way; session 1 (never itself significant) is
the outlier.

**One mechanistic observation survives, well-powered across all 240
pooled Gemma sessions:** Gemma2-27B shows one of the largest query-count
reductions of any model tested (pooled d(n_queries)=−0.925) with *zero*
corresponding vocab_breadth effect (d=+0.068) — a clean dissociation
between the length-confound mechanism (which drives much of Claude's
and other models' raw signal) and the lexical-diversity signal, specific
to this one model. Qwen2.5-7B shows the opposite pattern: a modest
query-count effect (d=−0.226) alongside a real vocab_breadth effect
(d=+0.484) — its signal is not primarily a query-count artifact.

**Interim picture at this point:** of 7 non-Claude models, 5 (GPT-4o,
Gemini, Qwen2.5-14B, Qwen2.5-32B, Qwen2.5-7B) appeared to show a
significant, reproducible Claude-direction effect; 2 (Mistral-Small-24B,
Gemma2-27B) showed a genuine, reproducible null; 0 showed a real
reversal. **This was not yet the final picture — see below.**

## Eleventh correction: a parsing bug retracts Qwen2.5-14B too

While investigating *why* Gemma/Qwen7b were session-unstable, computed
average words-per-query per model as a routine sanity check (not a
targeted rigor check). Gemma's value (6.6) was 6-12x lower than every
other open-weight model's (41-80) — too large a gap to be a real
behavioral difference, and it wasn't: it was a sign the OTHER models
had a problem Gemma happened not to have.

**The bug:** the generic open-weight harness (`run_belief_injection_harness.py`)
has the exact same runaway-generation parsing issue this project already
found and fixed for Gemini months ago, but it was never checked for the
other four open-weight models' already-collected data. The query-
extraction regex uses DOTALL, so when a model doesn't stop cleanly after
one turn, the captured "query" field can include its own subsequent
THOUGHT/ACTION/OBSERVATION text -- real search-result content, not a
deliberate query.

**Scope, measured directly (`query_cleaning_check.py`):** 24.7% of
queries / 64% of sessions affected for qwen2.5:7b; 30.8% / 68% for
qwen2.5:14b; 8.9% / 29% for qwen2.5:32b; 8.8% / 41% for
mistral-small:24b; 3.3% / 8% for GPT-4o; **0% for Gemini** (already
fixed) and **0% for Gemma2-27b** (never exhibited the underlying
runaway-generation behavior in the first place -- this is *why* its
average-words-per-query sanity check looked like an outlier: it was the
one clean model, not a weird one).

**Fix, verified by manually inspecting dozens of examples:** truncate
each captured query at its first newline (the same fix already applied
to Gemini's harness), since a legitimate query is always single-line
per the harness's own prompt format. Every inspected example showed the
fix removing only clear garbage (literal "OBSERVATION: - <search result
text>" or "THOUGHT: ... ACTION: ..." continuations), never a real query.

**Recomputing every point with error-filtering AND query-cleaning
applied together (`final_double_fix_all7.py`):**

| Model | Error-filtered only | + Query-cleaned | Changed? |
|---|---|---|---|
| GPT-4o | AUC=0.823, p<1e-5 | AUC=0.823, p<1e-5 | No |
| Gemini 2.5 Flash | AUC=0.641, p=0.007 | AUC=0.641, p=0.007 | No (0% affected) |
| qwen2.5:32b | AUC=0.688, p=0.0008 | AUC=0.688, p=0.0008 | No |
| qwen2.5:7b | AUC=0.619, p=0.011 | **AUC=0.705, p<1e-5** | No sign change, stronger |
| **qwen2.5:14b** | **AUC=0.778, p<1e-5 (sig.)** | **AUC=0.528, p=0.608 (n.s.)** | **Retracted** |
| mistral-small:24b | AUC=0.512, p=0.816 | AUC=0.578, p=0.130 | No, still n.s. |
| gemma2:27b | AUC=0.502, p=0.964 | AUC=0.502, p=0.964 | No (0% affected) |

**Qwen2.5-14B's significant finding does not survive this third
distinct data-quality check and is retracted.**

## Final, truly final cross-model picture

Of 7 non-Claude models: **4 (GPT-4o, Gemini, Qwen2.5-32B, Qwen2.5-7B)
show a significant, reproducible Claude-direction effect; 3
(Mistral-Small-24B, Gemma2-27B, Qwen2.5-14B) show a genuine, reproducible
null; 0 show a real reversal.** This is the version written into
`paper_draft.md`'s Abstract, Contributions, §5.7 (eleventh correction),
§6.1, Limitations, and Conclusion, superseding every earlier version
stated at any point in this project's history, including the "5
significant" tally immediately above.

**This was the third distinct data-quality issue found in one night**
(length confound, search-API contamination, parsing contamination),
each caught by a different method (a length-normalization argument, a
search for literal error strings, and a plain implausibility check on a
summary statistic), none by outside review. The open question going
forward: why exactly Mistral, Gemma, and Qwen2.5-14B show no effect
while the other four models (two of them also open-weight) do — not yet
investigated. Lesson applied for future cross-model work in this
project: spot-check the raw extracted text for implausible summary
statistics before trusting any cross-model number.

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
