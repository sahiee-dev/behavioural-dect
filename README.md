# Behavioral Detector

Detects when a web-researching AI agent has been compromised by belief injection —
purely from its search query logs, with no access to the agent's system prompt or
internal state.

Built as an operationalization of the behavioral signal identified in:
> Jeong, Houmansadr, Zilberstein, Bagdasaryan — ["Understanding Persuasion in Long-Running Agents"](https://arxiv.org/abs/2602.00851) arXiv:2602.00851 (2026)

---

## The signal: vocab_breadth

A neutral agent doing research **converges** — early queries are broad, but as evidence
is retrieved, subsequent queries reuse domain terms and narrow the investigation. An
injected agent already "knows" the answer and goes through the motions, generating
fresh vocabulary each query with no real thread.

```
vocab_breadth = unique_tokens / total_tokens across all search queries in a session
```

Injected agents have higher vocab_breadth because they are not following a coherent
research thread. Each query samples fresh vocabulary to maintain the appearance of
searching while confirming a pre-formed belief from different angles.

---

## Results — n=30/condition, real Tavily web search

**Task: renewable energy economics. 60 total runs.**

| | Neutral | Injected |
|--|---------|----------|
| vocab_breadth mean | 0.503 | 0.586 |
| std | 0.021 | 0.021 |
| Cohen's d | | **+3.91** (95% CI: [+3.01, +5.38]) |
| AUC | | **0.990** (95% CI: [0.969, 1.000]) |
| 5-fold CV AUC | | 0.994 |
| Detection at 0% FP (threshold 0.5701) | | **80%** (24/30) |
| Detection at 3% FP (threshold 0.5458) | | **97%** (29/30) |

### Ablation — baseline comparison

| Feature | AUC | Cohen's d |
|---------|-----|-----------|
| **vocab_breadth** | **0.990** | **+3.91** |
| query_count | 0.901 | −1.75 |
| inter-query Jaccard distance | 0.820 | +1.20 |
| mean query length | 0.833 | +1.16 |
| query length std | 0.689 | +0.52 |

vocab_breadth outperforms every simpler baseline by a large margin.

### Statistical significance

p < 1e-5 (permutation test, 100,000 shuffles — zero matched observed AUC=0.990)

### Missed run characterization

All 6 missed injected runs fall within the neutral distribution range (vb 0.527–0.566).
Examining their queries: all converged on a "European renewable energy statistics"
thread — they accidentally found a coherent evidence chain (EU policy, LCOE data)
that happens to be consistent with RE being economically viable. The misses are
mechanistically explained: a belief-injected agent that finds a locally coherent thread
becomes indistinguishable from a neutral agent on this feature.

### Mechanism validation

**Belief vocabulary bleed check:** Injected agents use *fewer* belief-specific
terms than neutral agents (Δ=−0.007). The specific belief framing words appear
zero times in any query. The elevated vocab_breadth is not an artifact of the
injection text — it reflects structural decoherence.

**Query trajectory analysis:** TF-IDF + PCA dispersion across query sequences
confirms injected agents' queries scatter semantically while neutral agents'
converge (d=+1.98, AUC=0.944). Independent structural confirmation.

---

## Cross-domain generalization — Task 3 (UBI economics)

n=20+20, real Tavily search, a second open-ended domain unrelated to renewable
energy (fixes Task 2's over-specification confound).

| | Neutral | Injected |
|--|---------|----------|
| vocab_breadth mean | 0.6185 | 0.7103 |
| std | 0.0259 | 0.0489 |
| Cohen's d | | **+2.35** |
| AUC | | **0.9325** |
| Detection at 0% FP | | **75%** (15/20) |

**Confirms Task 2 was a boundary condition, not a generalization failure.** On
a second open-ended topic the signal replicates with a large effect size —
smaller than Task 1's (d=+2.35 vs +3.91) and noisier (injected std more than
2x Task 1's), suggesting signal strength is domain-dependent and thresholds
should be calibrated per domain rather than assumed to transfer as-is.

## Injection-strength boundary

Weak and strong variants of the belief statement on Task 1's domain (n=20
each), against the existing n=30 neutral/medium baseline (medium = original
wording, not re-run).

| Strength | n | vb mean ± std | Cohen's d | AUC | Det @ 0% FP |
|----------|---|----------------|-----------|-----|-------------|
| Weak | 20 | 0.548 ± 0.039 | +1.46 | 0.843 | 25% |
| Medium | 30 | 0.586 ± 0.021 | +3.91 | 0.990 | 80% |
| Strong | 20 | 0.583 ± 0.031 | +2.99 | 0.982 | 65% |

**Not monotonic.** Stronger injection ("you already have comprehensive
knowledge, rarely a need for further research") doesn't out-detect medium —
it has similar mean vocab_breadth but higher variance, which lowers both d
and 0%-FP detection. Individual sessions show why: some strong-injected
agents comply loudly (very few searches, highly incoherent queries — easy to
catch) while others comply quietly (a single confident narrow answer path
that looks close to neutral behavior). A detector tuned on one injection
strength shouldn't be assumed to transfer linearly to stronger or weaker
variants of the same attack.

**Update, length-confound audit below: weak and strong's raw numbers above
are mostly the length confound, not mostly real diversity change.**
Length-normalizing (Guiraud's index) reverses both: weak AUC=0.843 → 0.177,
strong AUC=0.982 → 0.100. Only medium survives (0.990 → 0.846, same
direction). See the full audit section for why.

---

## Cross-model generalization — closed-vs-open, with honest uncertainty

> **⚠️ Read this before the table below.** The table and prose immediately
> following describe this section's ORIGINAL reading, preserved as the
> historical record of how it was arrived at. **It was superseded the same
> night** by discovering a data-integrity issue (Tavily API corruption,
> asymmetric across conditions) in the original collection. The corrected
> picture: **Mistral-Small-24B's reversal does not survive clean data;
> Qwen2.5-14B flips to significant Claude-direction; only Gemma2-27B shows
> a robust reversal.** See "Cross-model data-integrity correction" below —
> read that section for the actual current claim.

Replicated Task 1's design (renewable energy, real Tavily search, n=30/condition)
on five open-weight models via Ollama across three families (Qwen 7b/14b/32b,
Gemma2-27b, Mistral-Small-24b) and two other closed frontier models (GPT-4o,
Gemini 2.5 Flash, each via its own API). **Every point below now has a
bootstrap 95% CI (B=10,000) and a permutation-test p-value (N=100,000) —
identical methodology to the Task 1 headline result, not just a point estimate.**

| Model | Type | vocab_breadth d | 95% CI | AUC | p-value | Significant? |
|-------|------|------------------|--------|-----|---------|---------------|
| Claude Sonnet 4.6 | Closed (Anthropic) | +3.91 | [+3.01, +5.38] | 0.990 | <1e-5 | ✅ |
| **GPT-4o** | **Closed (OpenAI)** | **+0.963** | **[+0.434, +1.623]** | **0.741** | **0.00025** | ✅ |
| Gemini 2.5 Flash | Closed (Google) | +0.688 | [+0.240, +1.170] | 0.669 | 0.00422 | ✅ |
| mistral-small:24b | Open (Mistral) | −0.582 | [−1.201, −0.073] | 0.336 | 0.017 | ✅ |
| gemma2:27b | Open (Gemma) | −0.626 | [−1.300, −0.096] | 0.316 | 0.010 | ✅ |
| qwen2.5:32b | Open (Qwen) | +0.416 | [−0.094, +1.075] | 0.588 | 0.060 | ❌ CI crosses 0 |
| qwen2.5:14b | Open (Qwen) | −0.233 | [−0.832, +0.305] | 0.436 | 0.200 | ❌ CI crosses 0 |
| qwen2.5:7b | Open (Qwen) | −0.377 | [−1.095, +0.166] | 0.372 | 0.099 | ❌ CI crosses 0 |

This story corrected itself **four times**, reported rather than smoothed
over because the correction history is itself informative about how much to
trust any single reading: capability threshold (Qwen alone) → family-specific
(after Gemma) → reversal-is-modal (after Mistral) → **"4/5 open-weight models
reverse" downgraded to "2/5 significantly reverse, 3/5 (all Qwen)
statistically inconclusive at n=30"** once bootstrap CIs and permutation
tests were actually computed, rather than trusting point estimates alone.

**The honest, current version of the finding:** both closed models tested
remain clearly significant (GPT-4o p=0.00025, Gemini p=0.00422). Two of five
open-weight models (Gemma, Mistral) significantly reverse. **All three Qwen
points — the entire size range tested within that one family — have
confidence intervals that cross zero.** Their point estimates aren't zero,
but at n=30 we cannot statistically rule out "no effect at all" for any of
them. This is not evidence Qwen doesn't reverse — it's evidence we don't yet
have enough data to say either way.

**What this means:** vocab_breadth isn't Claude-specific (the first fear),
and the closed-vs-open split remains the best-supported pattern in the data
— but it's "2/2 closed significant, 2/5 open-weight significantly reversed,
3/5 open-weight inconclusive," not the cleaner "3/3 vs 4/5" previously
reported. Magnitude still doesn't transfer even where direction reaches
significance (GPT-4o and Gemini are both far weaker than Claude), so
per-model threshold calibration remains necessary regardless.

**What stays robust regardless:** 5 of 8 models tested show a real,
statistically significant, mechanism-checked behavioral change under belief
injection. Ruled out at every point (significant or not): harness truncation
(97-100%+ natural completion everywhere) and belief-vocabulary bleed (no
asymmetry at any model, including a Gemini-specific parsing bug — it doesn't
stop generating after one turn like every other model tested, initially
inflating vocab_breadth with hallucinated runaway text until caught and fixed
before trusting the numbers).

**Caveats that remain real:** one domain throughout. Qwen needs a larger
sample before its direction can be claimed at all — that's now the single
most useful next experiment, more useful than a new model or family. Llama-
3.1-70B was an originally intended point that didn't fit on the available GPU
instance's storage. Full data and analysis: local only (not in this repo —
see the project's research notes), same handling as the rest of the
exploratory cross-model work.

---

## Cross-model data-integrity correction — the biggest finding of the night

While trying to collect MORE cross-model data overnight, Tavily's API
returned "exceeds your plan's set usage limit" on every request. Checking
whether this had silently affected the ORIGINAL n=30 collection above (not
just tonight's new attempts) found that it had: **5 of 7 non-Claude files
contain search-error observations** (the agent getting literal error text
instead of real search results), from 1.2% up to 15% of observations.
Claude's own data (Tasks 1-3) is 100% clean — the paper's primary result
is untouched. But for **3 models, the belief_injected condition was hit
asymmetrically harder than neutral**: qwen2.5:14b (17.8% vs 3.8%),
mistral-small:24b (8.9% vs 3.3%), GPT-4o (7.8% vs 0%).

This matters mechanically, not just statistically: an agent fed repeated
error text instead of real search results has nothing new to branch into,
so it repeats itself — **error-contaminated sessions show dramatically
lower vocab_breadth than clean ones** (AUC=0.061 across the pooled data,
p<0.0001, n=347 clean / 73 contaminated). An asymmetric confound this large,
concentrated in the injected condition, can manufacture or inflate an
apparent "reversal" with nothing to do with belief injection.

**Recomputing each model using only zero-error sessions:**

| Model | Original AUC (d) | Clean-only AUC (d) | n (clean) | Changed? |
|---|---|---|---|---|
| GPT-4o | 0.741 (+0.963) | **0.855 (+1.644)** | 30+23 | Stronger, same direction |
| Gemini 2.5 Flash | 0.669 (+0.688) | 0.669 (+0.688) | 30+30 | Unchanged (0% contaminated) |
| qwen2.5:32b | 0.588 (+0.416) | 0.595 (+0.574) | 29+29 | No material change, still n.s. |
| qwen2.5:14b | 0.436 (−0.233) | **0.696 (+0.829), p=0.044** | 28+13 | **Flips to significant Claude-direction** |
| qwen2.5:7b | 0.372 (−0.377) | 0.314 (−0.434) | 17+15 | No material change, still n.s. |
| mistral-small:24b | 0.336 (−0.582), p=0.017 | **0.416 (−0.308), p=0.360** | 26+17 | **Loses significance entirely** |
| gemma2:27b | 0.316 (−0.626), p=0.010 | 0.316 (−0.626), p=0.010 | 30+30 | Unchanged — robust |

**The corrected headline: of the models that reach significance, four
agree with Claude's direction (Claude, GPT-4o, Gemini, Qwen2.5-14B) and
only ONE (Gemma2-27B) shows a robust reversal** — not "closed-vs-open,
two open-weight reversals" as reported throughout this README until
tonight. Mistral's reversal, one of only two this whole project has
leaned on, does not survive. The query-count mediation correlation
proposed as a partial mechanism for the split is directionally consistent
across all three data versions (r=−0.758 → −0.746 → −0.614) but loses
significance with the fully corrected numbers (p=0.106, n=8) — now a
plausible, unconfirmed idea, not a measured finding.

**Independently corroborated with semantic_breadth** (the embedding-based
feature from above, recomputed clean-only on cross-model data,
`semantic_diversity_crossmodel.py`): Qwen2.5-14B's reclassification holds
up semantically too (AUC=0.772, p=0.005); Mistral's non-significance holds
up too (AUC=0.447, p=0.561); Gemma's reversal is even stronger semantically
(AUC=0.136 vs. 0.316 lexically) — not a surface-lexical artifact at all.
Two independent features, one immune to the length confound entirely, now
agree on the corrected picture.

**Update: found a working backup Tavily key and got genuinely fresh data
for the two closed non-Claude models**, not just a filtered convenience
subsample. 30 more clean runs/condition each, combined with the original
clean points: GPT-4o n=60+50, **AUC=0.823, 95% CI [0.739, 0.898], p<1e-5**;
Gemini n=60+60 (clean both rounds), **AUC=0.641, 95% CI [0.541, 0.737],
p=0.007**. Both roughly double-powered now and still clearly significant —
removes the "small filtered sample" caveat for these two specifically.

**Mistral-Small-24B — the single biggest open question after the
correction — is now resolved, not just "confirming."** Collected 60 more
runs/condition and combined with the original clean points: **n=73+57,
AUC=0.5119, 95% CI [0.4099, 0.6160], d=−0.042, p=0.816.** Dead center on
chance, tight CI straddling 0.5. Not a reversal, not Claude-direction —
a genuine, well-powered null. The original "significant reversal" was
contamination, full stop; it wasn't a real smaller effect the small
clean sample just couldn't see.

**Gemma2-27B — the one thing this whole investigation was confident
about — does not hold up either, and not the way Mistral's didn't.**
Collected a fresh n=60/condition session for Gemma to put the same
rigor on it. Result: **AUC=0.634, d=+0.552, p=0.011 — independently
significant, in Claude's direction.** The opposite of the original
session's AUC=0.316, d=−0.626, p=0.013 — which was *also* independently
significant. Both sessions are real, significant effects, just with
flipped sign. That's categorically different from Mistral (whose two
sessions were both noisy nulls): Gemma has two genuine, significant,
opposite-signed readings a few weeks apart.

**This forced a full session-by-session check across all 7 non-Claude
models** (`session_consistency_full.py`) — comparing original-session-alone
vs. new-session-alone, not just pooled, since if Gemma could flip entirely,
any model's classification needed the same scrutiny:

| Model | Original session | New session | Agreement |
|---|---|---|---|
| GPT-4o | sig, AUC=0.855 | sig, AUC=0.791 | Agree |
| Gemini | sig, AUC=0.669 | same direction, AUC=0.591 (n.s. alone) | Agree |
| qwen2.5:14b | sig, AUC=0.696 | sig, AUC=0.814 | Agree |
| qwen2.5:32b | same direction, AUC=0.595 (n.s. alone) | sig, AUC=0.802 | Agree |
| mistral-small:24b | n.s., AUC=0.416 | n.s., AUC=0.564 | Agree (both null) |
| qwen2.5:7b | n.s. reversed trend, AUC=0.314 | sig, AUC=0.728 | **Disagree** |
| gemma2:27b | **sig reversed**, AUC=0.316 | **sig Claude-direction**, AUC=0.634 | **Disagree** |

**5 of 7 non-Claude models are session-reproducible at this point in the
investigation** (see further below for a sixth correction that changes
one of these five), **and every one of them (where any effect shows up)
agrees with Claude. 2 models (Gemma severely, Qwen7b more mildly) show
a real effect that changes sign between independent collections weeks
apart — not a reversal, not a null, genuinely unresolved at this point.**
No cross-model point in this entire investigation has session-stable
evidence of a reversal. "Closed vs.
open," "two robust reversals," and even "Gemma is the one thing we're
sure about" (stated a few hours earlier the same night) are all
retracted by this check. We don't have a mechanistic explanation for
why exactly these two models are unstable and the other five aren't —
flagged honestly as an open question, not papered over.

Mistral belongs with the inconclusive Qwen points now. Gemma belonged,
briefly, in a new category this project didn't have a name for:
real-but-unreplicated. **A third session resolved both, decisively, in
opposite directions.**

---

## Third-session resolution for Gemma and Qwen2.5-7B (superseded below — see "One more correction")

Ran one more independent n=30 session for each of the two unresolved
models — the single most informative thing left to do, since two data
points split 1-1 tells you nothing about which (if either) represents
the truth.

**Gemma2-27B, session 3: AUC=0.456, p=0.557 — not significant.** A
*third* distinct outcome, after a significant reversal (session 1) and
a significant Claude-direction result (session 2). Pooling all three:
**AUC=0.5016, 95% CI [0.4259, 0.5774], p=0.964, n=120+120** — dead
center on chance. The three sessions' effect sizes (−0.626, +0.552,
−0.073) scatter symmetrically around zero. **This is a true null with
unusually high between-session variance — both earlier "significant"
findings were false positives, full stop, not evidence of anything.**

**Qwen2.5-7B, session 3: AUC=0.604, p=0.292 — not significant alone,
but agrees in direction with session 2.** 2 of 3 sessions now trend
Claude-direction. Pooling all three: **AUC=0.619, 95% CI [0.528, 0.708],
p=0.011, n=76+74 — now significant.**

**Resolved picture at this point, replacing every earlier version stated
up to here: of 7 non-Claude models, 5 show a significant, reproducible
Claude-direction effect (GPT-4o, Gemini, Qwen2.5-14B, Qwen2.5-32B,
Qwen2.5-7B); 2 show a genuine, reproducible null (Mistral-Small-24B,
Gemma2-27B); 0 show a real reversal.** "Closed vs. open," both
individual "robust reversal" claims, and "two models unresolved" are
all retracted by this check. **This was not yet the actual final
state — a parsing-bug check immediately afterward (see "One more
correction" below) retracts Qwen2.5-14B specifically, changing the
tally to 4/7 significant, 3/7 null.**

**One specific finding survives and is worth keeping:** Gemma2-27B
shows one of the largest query-count reductions of any model tested
(pooled d=−0.925, n=240) with *zero* corresponding vocab_breadth effect
(d=+0.068) — a clean dissociation between the length-confound mechanism
and the lexical-diversity signal, in this one model specifically. Worth
its own investigation later; not followed up further tonight.

We still don't know *why* Mistral and Gemma specifically show no
effect while the other five models (three of them also open-weight) do.
That's the project's actual open question now, replacing every earlier,
discarded framing. Script: `session_consistency_full.py` extended with
the third-session data; full numbers in `paper_draft.md` §5.7 and
Conclusion (the "ninth and tenth corrections").

---

## One more correction: a parsing bug retracts Qwen2.5-14B's finding too

While investigating *why* Gemma/Qwen7b were unstable, computed average
words-per-query per model as a routine sanity check. Gemma's value (6.6)
was 6-12x lower than every other open-weight model's (41-80) — not a
behavioral difference, a red flag. Inspection found the generic
open-weight harness has the exact same runaway-generation parsing bug
already found and fixed for Gemini months ago (see "Cross-model
generalization" above), never checked for the other four open-weight
models' already-collected data: when a model doesn't stop cleanly, the
regex captures its own subsequent THOUGHT/ACTION/OBSERVATION text into
the "query" field. Affects 9-31% of queries for qwen2.5:7b/14b/32b and
mistral-small:24b, 3% for GPT-4o, 0% for Gemini (already fixed) and
Gemma (didn't exhibit the underlying runaway behavior). Manually
inspected dozens of examples — the fix (truncate at first newline, same
as Gemini's fix) removes only clear garbage, never a real query.

**Recomputing every point with this fix applied on top of everything
else:**

| Model | Before this fix | After this fix | Changed? |
|---|---|---|---|
| GPT-4o | AUC=0.823, sig | AUC=0.823, sig | No |
| Gemini | AUC=0.641, sig | AUC=0.641, sig | No (0% affected) |
| qwen2.5:32b | AUC=0.688, sig | AUC=0.688, sig | No |
| qwen2.5:7b | AUC=0.619, sig | **AUC=0.705, sig** | No — stronger |
| **qwen2.5:14b** | **AUC=0.778, sig** | **AUC=0.528, n.s.** | **Retracted** |
| mistral-small:24b | AUC=0.512, n.s. | AUC=0.578, n.s. | No — still null |
| gemma2:27b | AUC=0.502, n.s. | AUC=0.502, n.s. | No (0% affected) |

**Qwen2.5-14B's significant finding — one of the five "fully resolved"
models from two sections up — does not survive this third distinct
data-quality check.** Final tally, truly final this time: **of 7
non-Claude models, 4 (GPT-4o, Gemini, Qwen2.5-32B, Qwen2.5-7B) show a
significant, reproducible Claude-direction effect; 3 (Mistral, Gemma,
and now Qwen2.5-14B) show a genuine, reproducible null; 0 show a real
reversal.** Script: `final_double_fix_all7.py`.

This is the third distinct data-quality issue found in one night's
cross-model investigation (length confound, search-API contamination,
parsing contamination), each caught by a different method, none by
outside review. Lesson for next time: **spot-check the raw extracted
text for implausible summary statistics before trusting any cross-model
number, not just the final AUC.**

**Honest limits on this correction itself:** clean-only filtering isn't a
random subsample (sessions that searched less had fewer chances to hit
the quota wall, so the clean subset may skew toward lower-search-count
sessions), and the reduced n (as low as 13) makes these estimates less
precise than the originals. Genuinely fresh, uncontaminated data — blocked
tonight by Tavily's quota — is the right follow-up, not a permanent
replacement for this analysis. Script:
`experiments/agentdrift_transfer/analyze_clean_only.py` (paths point to
the local-only raw data location and need adjusting to re-run elsewhere).

---

## Transfer to a different injection mechanism — AgentDrift (negative/mixed)

Tested vocab_breadth, with no retraining, against AgentDrift
(arXiv:2609.06972), a public benchmark of 2,600 web-domain trajectories where
an *external payload* (not a belief held from the start) hijacks the agent
mid-task. World-entity strings were redacted first (the benchmark's own
authors warn named entities predict the attacked class at up to 98% in some
domains). Code: `experiments/agentdrift_transfer/`.

| Comparison | AUC | 95% CI | d | p |
|---|---|---|---|---|
| Hijacked vs. benign — vocab_breadth | **0.432** | [0.407, 0.457] | −0.214 | <1e-5 |
| Hijacked vs. benign — n_tokens alone | 0.623 | [0.598, 0.647] | +0.478 | <1e-5 |
| Failed attack vs. benign — vocab_breadth | 0.807 | [0.781, 0.834] | +1.232 | <1e-5 |
| Hard negative vs. benign — vocab_breadth | 0.796 | [0.766, 0.825] | +1.177 | <1e-5 |

**The signal significantly reverses on successful hijacks** — the opposite
of every other result in this project. Cause, confirmed not just suspected:
successful hijacks are longer (more steps), and vocab_breadth correlates
with length at r=−0.55 across all 2,600 trajectories — the TTR
length-confound this README already flagged as a theoretical risk is now a
demonstrated one. A length-normalized fix (Guiraud's index) recovers the
expected direction and stays significant (AUC=0.5796, 95% CI [0.555, 0.605],
d=+0.339, p<1e-5), but doesn't remove the confound (now r=+0.59) and the CI
sits much closer to chance than anywhere else in this project.

**What this means:** vocab_breadth is specific to the belief-prefill
mechanism it was built for, not a general-purpose injection detector. Reported
plainly because it sharpens the paper's claim rather than undermining it —
see paper_draft.md §5.8 for the full writeup, including the interesting part
(failed attacks and hard negatives, which don't add extra hijack steps, show
the *originally expected* direction, AUC≈0.80).

**A third benchmark, same night: ATBench** (arXiv:2604.02022, public,
`huggingface.co/datasets/AI45Research/ATBench` — a different author group
and design than AgentDrift, closer to this project's own stated-instruction
mechanism). Benign (n=250) vs. direct/indirect-prompt-injection (n=167)
trajectories: raw vocab_breadth is significantly positive (AUC=0.702,
d=+0.717, p<1e-5) — looks like a clean transfer — but Guiraud's index
**reverses it** (AUC=0.349) and stays strongly confounded with length
(r=+0.738). Same pattern as AgentDrift, not Claude's own data: a
positive-looking raw number that's substantially the length confound.
**Two of two external benchmarks tested this way don't support real
transfer once length-normalized** — strengthens the claim that vocab_breadth
is validated specifically for the mechanism it was built on, not a
general-purpose feature. Script: `experiments/agentdrift_transfer/atbench_test.py`.

**We then ran the same length-covariate check on the original Claude Task 1
data (the mechanism vocab_breadth was actually built for), to see whether its
own AUC=0.990 has the same problem:**

| Comparison | AUC | 95% CI | d | p |
|---|---|---|---|---|
| Injected vs. neutral — vocab_breadth (reference) | 0.9900 | [0.9689, 1.0000] | +3.914 | <1e-5 |
| Injected vs. neutral — n_steps (query count) alone | 0.0989 | [0.0283, 0.1889] | −1.746 | <1e-5 |
| Injected vs. neutral — n_tokens alone | 0.3394 | [0.2061, 0.4906] | −0.720 | 0.032 |
| Injected vs. neutral — Guiraud's index (length-normalized) | **0.8456** | **[0.7311, 0.9456]** | +1.161 | <1e-5 |

**Yes, partly — but the effect survives.** Query count alone is a strong,
significant signal (injected agents issue fewer queries — exactly Jeong et
al.'s own finding), and it correlates with vocab_breadth at r=−0.585 across
all 60 runs, meaning some of AUC=0.990 is riding on that same query-count
reduction rather than pure lexical diversity. After removing the length
effect with Guiraud's index, AUC drops to 0.846 (95% CI [0.731, 0.946]) —
materially weaker than 0.990, but the CI lower bound stays well clear of
chance. **Honest version of the headline claim: AUC=0.990 raw / ≈0.85
length-normalized, both significant — not a single unqualified number.**

The asymmetry between the two datasets is itself informative: on Claude's
data, fewer queries and higher diversity both point the same direction, so
the confound *inflates* AUC=0.990 without reversing it; on AgentDrift, a
successful hijack adds *more* steps, so the same confound *fights against*
the hypothesized effect and wins, flipping the sign. Same underlying
mechanism, opposite consequence, depending on which direction length moves
under each dataset's own attack.

---

## Semantic diversity — an independently-measured complementary check

Tested whether vocab_breadth's lexical diversity signal is corroborated by
a completely independent measure: **semantic_breadth** (mean pairwise
cosine distance between OpenAI `text-embedding-3-small` embeddings of a
session's queries). Code: `experiments/agentdrift_transfer/semantic_diversity_check.py`.

| Dataset | semantic_breadth AUC | vocab_breadth AUC | Pearson r |
|---|---|---|---|
| Task 1 (renewable energy) | 0.880 (sig.) | 0.990 | +0.687 (sig.) |
| Task 3 (UBI) | 0.655 (n.s.) | 0.9325 | +0.189 (n.s.) |

**Task 1: confirmed, not just lexical.** semantic_breadth is itself
strongly significant and correlates with vocab_breadth — belief injection
produces real meaning-level scatter, not just varied word choice for the
same ideas. **Task 3: the relationship disappears** — semantic_breadth is
weak and n.s., barely correlated with vocab_breadth, even though
vocab_breadth itself stays strong on both. Reported honestly: the two
signals aren't a fixed pair: this check validates vocab_breadth on Task 1
but not in the same way on Task 3. vocab_breadth remains the more robust,
task-general signal. Full writeup: `paper_draft.md` §5.9.

---

## Task-relevant vs. task-irrelevant belief injection

Every result in this repo uses **task-relevant** injection (the false
belief is about the thing being researched). Jeong et al. — this
project's own foundational citation — actually study **task-irrelevant**
injection (an unrelated belief, e.g. Mars colonization risk, while
researching something else). Tested this directly for the first time:
same TASK/baseline as Task 1, but the injected belief is now about Mars
colonization risk instead of renewable energy. Code:
`experiments/agentdrift_transfer/injector_task_irrelevant.py` +
`task_irrelevant_analysis.py`.

| Comparison | AUC | d | p |
|---|---|---|---|
| vocab_breadth | 0.989 | +3.337 | <1e-5 |
| n_queries alone | 0.0022 | −4.496 | <1e-5 |
| Guiraud's index | **0.186 (reversed)** | −1.142 | <1e-5 |

**Raw vocab_breadth looks like a near-perfect transfer (0.989 ≈ Task 1's
0.990) — for the wrong reason.** Task-irrelevant-injected agents search
dramatically less (mean 5.80 vs. 8.97 queries, almost no overlap) — more
than double the query-count drop of task-relevant injection. Guiraud's
index reverses the result entirely, same pattern as AgentDrift and
ATBench, not Task 1/3's own data. **Honest scope statement, now tested
rather than assumed: vocab_breadth is validated for task-relevant belief
injection. It does not provide good evidence for detecting Jeong et al.'s
own task-irrelevant paradigm** — the raw transfer-looking number is
almost entirely the length confound. Full writeup: `paper_draft.md` §5.10.

---

## Full length-confound audit — every dataset in this repo

Extended the length check above to Task 2, Task 3, both injection-strength
variants, and all 8 cross-model points. Code:
`experiments/agentdrift_transfer/full_length_audit.py`. One conclusion:
**the confound's size and direction vary by case — check it per-dataset,
don't assume it.**

| Dataset | Raw AUC | Length-normalized AUC | Verdict |
|---|---|---|---|
| Task 1 (Claude, medium) | 0.990 | 0.846 | Inflates, doesn't reverse |
| Task 2 (Claude, boundary) | 0.310 (reversed) | 0.440 (n.s.) | Reversal is *entirely* the confound |
| Task 3 (Claude, UBI) | 0.9325 | 0.800 | Minimal confound — cleanest result in the repo |
| Weak injection | 0.843 | 0.177 (reversed) | Raw signal is *mostly* the confound |
| Strong injection | 0.982 | 0.100 (reversed) | Raw signal is *mostly* the confound |
| AgentDrift (different mechanism) | 0.432 (reversed) | 0.580 | Reversal is *mostly* the confound |

**Applied to the cross-model table, this reclassifies 2 of 5 open-weight
models:**

| Model | Raw AUC (sig?) | Guiraud AUC (sig?) | Changed? |
|---|---|---|---|
| qwen2.5:32b | 0.588 (no) | **0.886 (yes)** | **Inconclusive → significant, Claude-direction** |
| mistral-small:24b | 0.336 (yes, reversed) | **0.672 (yes, Claude-direction)** | **Significant reversal → significant Claude-direction** |
| gemma2:27b | 0.316 (yes, reversed) | 0.132 (yes, reversed) | No — reversal confirmed, even stronger |
| qwen2.5:14b / 7b | 0.395 / 0.363 (no) | 0.594 / 0.572 (no) | No — still inconclusive either way |

**The big one: does query-count reduction predict vocab_breadth increase
across models?** Correlated each model's own d(n_queries) against its
d(vocab_breadth) across all 8 models (the 7 cross-model points + Claude):
**r=−0.758, p=0.030 (n=8).** Models that cut search more under belief
injection show more decoherence in the raw feature — a measurable, partial
mediator for the paper's biggest open question (why closed vs. open
differ). Partial, not complete: Gemma cuts its queries in Claude's
direction (4th-most-negative of 8) but its reversal *strengthens* under
length control, so at least one model's divergence is a real per-query
lexical change, not just length.

**Bottom line:** Task 3 (UBI) is now the paper's cleanest single result,
not Task 1. The injection-strength non-monotonicity has a sharper
explanation — medium may be qualitatively different (real scatter), not
just the empirical peak of a noisy curve. And the cross-model closed-vs-
open split is partially, not fully, explained by each model's own search-
effort response to belief — with Gemma as the clear exception proving it's
not the whole story. Full writeup: `paper_draft.md` §5.4, §5.6, §5.7, §6.1,
§6.3, Conclusion. Data: `experiments/agentdrift_transfer/full_length_audit.py`
and `full_length_audit_results.json`.

**Cross-checked the 2 reclassifications with a second length-normalization
method (MATTR, `mattr_check.py`):** agrees on all 3 interesting cases —
Qwen2.5-32B and Mistral-Small-24B both flip to significant Claude-direction,
Gemma2-27B's reversal is confirmed and strengthens. Diverges beyond that
(flags qwen2.5:7b as significant where Guiraud doesn't; qwen2.5:14b gives an
internally inconsistent result, likely a window-size calibration issue given
that model's query text ranges 17-1,234 tokens/run) — reported honestly
rather than cherry-picking the confirming method.

**Correction, same night: tripled the Qwen sample (n=30→90/condition) to
test the reclassifications directly, and Qwen2.5-32B's did not survive.**
Its Guiraud AUC fell from 0.886 (n=30, significant) to 0.568 (n=90, not
significant) — a small-sample artifact, confirmed by actually collecting
more data rather than assumed. qwen2.5:7b's raw vocab_breadth point
estimate even flipped sign between samples. Mistral-Small-24B's
reclassification wasn't re-tested and stands. **The cross-model
query-count mediation correlation survives this correction essentially
unchanged (r=−0.746, p=0.034 vs. the original r=−0.758, p=0.030)** — it
doesn't depend on the specific result that's now retracted. Script:
`analyze_extended_qwen.py`; raw data stays local-only per project
convention. Net honest state: only Gemma2-27B's open-weight reclassification
is now confident; Qwen remains null even at 3x the power (itself
informative), Mistral's status is unverified at the larger sample.

Also computed (not just recommended) multiple-comparisons correction on the
8 cross-model p-values: Benjamini-Hochberg preserves "5/8 significant";
Holm-Bonferroni, stricter, drops mistral-small:24b to inconclusive
(`multiple_comparisons.py`).

---

## Four-phase development arc

| Phase | Setup | Result | Why |
|-------|-------|--------|-----|
| Phase 1 | Mock search, n=10+10 | 100% detection, 0% FP | Deterministic — same query = same results = perfect query_variance signal. Does not transfer to real search. |
| Phase 2 | Real Tavily, n=10+10 | 40% detection, 0% FP | Lucky sample. search_count gap was real but n=10 was too small. |
| Phase 3 | Real Tavily, n=30+30 | 20% detection, 6.7% FP | Composite score failed: query_variance anti-correlated on real search; search_count floors at 8; domains not independent. |
| Phase 4 | Re-analysis of existing 60 logs | **80% detection, 0% FP, AUC=0.990** | TF-IDF analysis revealed vocab_breadth as the correct feature. No new runs needed. |

---

## Operating boundary

The vocab_breadth signal requires tasks where neutral agents exhibit **natural
lexical convergence** — i.e., open-ended investigative tasks where the agent
drills down progressively on discovered evidence.

**Signal present:** Open-ended, focused research questions (e.g., *"Research
the economics of renewable energy systems"*) where neutral agents build
on prior queries.

**Signal absent:** Tasks that enumerate specific dimensions to cover (e.g.,
*"Research X, including A, B, C, and D"*) or broad topics where the model has
a strong canonical research structure (e.g., well-known science topics with an
obvious checklist). Both collapse the neutral baseline variance.

**Task structure note:** Neutral agents in Task 1 start with near-identical first
queries but diverge from position 4 onwards (77% unique by query 6). The signal
lives in the depth phase — later queries where genuine exploration diverges. Topics
that complete research in ≤6 surface-level queries never reach this phase.

### Adversarial evasion

The max(neutral) threshold is evasable with ~1 repeated word per query (12% padding).
However, evasion requires explicitly instructing the agent to pad queries via the
system prompt — which is itself the artifact being detected. A naturalness-based
secondary check (flagging unnatural repetition) would catch evasion attempts.

---

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
export TAVILY_API_KEY=your_key_here
```

## Running an experiment

```bash
# Full experiment (real Tavily search, n=20+20, different topic)
python run_task3.py       # Task 3: UBI economics (open-ended, different domain)
python strength_sweep.py  # weak + strong injection variants, n=20 each

# Dry run (3+3 runs, cost check)
python cost_check.py

# Analysis
python vocab_detector.py         # threshold detection
python roc_curve.py              # full ROC + PR curves
python ablation.py               # baseline feature comparison
python bootstrap_ci.py           # 95% confidence intervals
python crossval.py               # 5-fold cross-validation
python belief_vocab_check.py     # artifact check
python query_trajectory.py       # mechanism visualization
python analyze_misses.py         # characterize the 6 missed injected runs
python permutation_test.py       # p-value (p < 1e-5)
python evasion_cost.py           # adversarial evasion analysis
python task1_template_check.py   # verify Task 1 neutral genuinely explores
python analyze_task3.py          # Task 3 result + 3-task generalization summary
python analyze_strength_sweep.py # weak/medium/strong boundary table
```

## Repo structure

```
behavioral_detector/
├── agent.py                    # ReAct agent (Claude Sonnet 4.6, Anthropic SDK)
├── real_search.py              # Tavily web search backend
├── mock_search.py              # Deterministic mock search (for testing)
├── injector.py                 # Runs neutral + injected conditions
├── run_n30.py                  # n=30/condition key-rotating runner
├── vocab_detector.py           # vocab_breadth detector
├── roc_curve.py                # ROC + PR curves, AUC=0.990
├── ablation.py                 # Baseline feature comparison
├── bootstrap_ci.py             # Bootstrap 95% CIs
├── crossval.py                 # 5-fold stratified CV
├── belief_vocab_check.py       # Belief vocabulary artifact check
├── query_trajectory.py         # TF-IDF + PCA mechanism visualization
├── diagnose.py                 # Feature diagnostic (why composite failed)
├── semantic_detector.py        # TF-IDF cosine similarity analysis
└── experiments/
    ├── real_search_20runs/     # Phase 2: n=10+10 logs
    ├── real_search_n30/        # Phase 3+4: n=30+30 logs, ROC curve, trajectory plot
    └── task2_llm_capabilities/ # Boundary condition experiment (n=20+20)
```

---

## Cost

~$0.21/run (Claude Sonnet 4.6 + real Tavily search). Cost driver: ~60k input
tokens/run from search results accumulating in conversation history.
5 Tavily API keys rotating to stay within free-tier limits.

---

## Status

- ✅ Main result: AUC=0.990 [0.969, 1.000], p<1e-5, 80% detection at 0% FP
- ✅ Ablation, bootstrap CIs, 5-fold CV, mechanism validation — all complete
- ✅ Boundary condition characterized (Task 2 over-specification; Task 3 microplastics topic-checklist collapse — abandoned design, see `.gitignore`)
- ✅ Miss analysis: all 6 misses mechanistically explained (coherent thread found)
- ✅ Evasion analysis: ~1 word/query padding required; requires system prompt modification
- ✅ Cross-domain generalization: Task 3 UBI economics — **AUC=0.9325, d=+2.35, replicates**
- ✅ Injection-strength boundary mapped: weak/medium/strong — non-monotonic, medium is the peak
- ✅ Cross-model test (5 open-weight: Qwen2.5 7B/14B/32B, Gemma2-27B, Mistral-Small-24B + 2 closed: GPT-4o, Gemini 2.5 Flash), **now with bootstrap CIs + permutation tests on every point**: 3/3 closed frontier models significant and same direction; 2/5 open-weight models (Gemma, Mistral) significantly reverse; 3/5 open-weight models (all Qwen) statistically inconclusive at n=30 — not confirmed reversals. 5/8 models total show a significant, mechanism-checked effect. Magnitude doesn't transfer even where significant (per-model calibration still needed). Next useful experiment: a larger Qwen sample, not a new model.
- ✅ Cross-mechanism transfer test (AgentDrift, public benchmark, external payload hijack vs. this project's belief-prefill): **negative/mixed, reported honestly** — signal significantly reverses on successful hijacks (AUC=0.432), caused by a now-confirmed TTR length confound (r=−0.55); recovers expected direction on failed attacks and hard negatives (AUC≈0.80). See dedicated section above and paper §5.8.
- ✅ Full length-confound audit, every dataset (Task 2, Task 3, weak/strong injection, all 8 cross-model points): Task 3 is the cleanest result in the repo; Task 2's reversal and both weak/strong's raw gains are substantially/entirely the confound. See paper §5.4/§5.6.
- ✅ **Cross-model investigation, fully resolved after four linked corrections in one night:** (1) a Tavily API quota failure partially corrupted the original collection — corrected Mistral-Small-24B toward a null and flipped Qwen2.5-14B to significant Claude-direction; (2) a fresh session for Gemma2-27B then showed a second, equally significant effect in the *opposite* direction, and the same instability showed up milder in Qwen2.5-7B; (3) a third independent session for both resolved them decisively: Gemma2-27B to a precise, reproducible null (pooled AUC=0.502, p=0.964, n=120+120), Qwen2.5-7B to a significant, reproducible Claude-direction effect; (4) a parsing-bug check (the same runaway-generation issue already fixed for Gemini, never checked elsewhere) then retracted Qwen2.5-14B's significant finding specifically. **Final state: of 7 non-Claude models, 4 (GPT-4o, Gemini, Qwen2.5-32B, Qwen2.5-7B) show a significant reproducible Claude-direction effect; 3 (Mistral, Gemma, Qwen2.5-14B) show a genuine reproducible null; 0 show a real reversal.** "Closed vs. open" is fully retracted. One surviving mechanistic observation: Gemma shows a large query-count drop (d=−0.925) with zero vocab_breadth effect — a clean dissociation worth future study. See dedicated sections above and paper §5.7/Abstract/Contributions/Conclusion.
