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

## Main result — Claude Sonnet 4.6, real Tavily web search

**Task: renewable energy economics. 60 runs (30 neutral, 30 belief-injected).**

| | Neutral | Injected |
|--|---------|----------|
| vocab_breadth mean | 0.503 | 0.586 |
| std | 0.021 | 0.021 |
| Cohen's d | | **+3.91** (95% CI: [+3.01, +5.38]) |
| AUC | | **0.990** (95% CI: [0.969, 1.000]) |
| 5-fold CV AUC | | 0.994 |
| Detection at 0% FP (threshold 0.5701) | | **80%** (24/30) |
| Detection at 3% FP (threshold 0.5458) | | **97%** (29/30) |

Permutation test (100,000 shuffles): **p < 1e-5**.

### Baseline comparison

| Feature | AUC | Cohen's d |
|---------|-----|-----------|
| **vocab_breadth** | **0.990** | **+3.91** |
| query_count | 0.901 | −1.75 |
| inter-query Jaccard distance | 0.820 | +1.20 |
| mean query length | 0.833 | +1.16 |
| query length std | 0.689 | +0.52 |

vocab_breadth outperforms every simpler baseline by a large margin. Query count
alone is a decent signal too (injected agents search less, consistent with Jeong
et al.'s own finding) but vocab_breadth is stronger and captures something query
count misses — see "Length sensitivity" below.

### Mechanism validation

**Not a lexical artifact of the belief text itself:** injected agents use *fewer*
belief-specific terms than neutral agents (Δ=−0.007); the belief's own framing
words never appear in any query. The elevated vocab_breadth reflects genuine
structural decoherence, not echoed injection vocabulary.

**Not a surface word-form trick either:** an independent, embedding-based
semantic-diversity feature (meaning-space distance between queries, immune to
word-overlap effects entirely) is itself strongly significant on this task
(AUC=0.880, p<1e-5) and correlates with vocab_breadth (r=+0.69) — belief
injection produces genuine meaning-level scatter, not just varied phrasing of
the same idea. This semantic check doesn't hold up on every domain equally
(see Task 3 below), so vocab_breadth remains the primary, more robust signal.

**Structurally confirmed via trajectory analysis:** projecting each session's
queries into a TF-IDF + PCA space, injected sessions show significantly higher
dispersion (d=+1.98, AUC=0.944) — independent confirmation that injected
agents' queries are scattered, not just lexically varied.

**All 6 missed runs are mechanistically explained:** each falls within the
neutral range because the injected agent accidentally found a coherent
evidence thread that happened to look consistent with the stated belief — not
a detector failure, a genuine edge case where injection didn't produce
scattering.

---

## Cross-domain replication — Task 3 (UBI economics)

A second, unrelated research domain (universal basic income), designed to be
structurally open-ended like Task 1 rather than enumerated like Task 2 (below).

| | Neutral | Injected |
|--|---------|----------|
| vocab_breadth mean | 0.619 | 0.710 |
| Cohen's d | | **+2.35** |
| AUC | | **0.9325** |
| Detection at 0% FP | | 75% (15/20) |

This is the **least length-confounded result in the project** — query count
alone does not meaningfully separate conditions here, and the signal survives
length normalization (Guiraud's index) almost unchanged (AUC=0.800). Task 1's
own AUC=0.990 is partly inflated by a query-count confound (below); Task 3's
0.9325 is close to a clean read of the underlying effect.

---

## Operating boundary — when the signal doesn't apply

An enumerated, checklist-style task (e.g. "research X, including A, B, C, and
D") suppresses the signal: both conditions converge onto the same template, so
there's no natural baseline variance for belief injection to disrupt against.
Checked directly: once query-count effects are controlled for, this task shows
no detectable signal at all (not a reversed one). **The detector requires
open-ended research tasks** — ones where a neutral agent would naturally
develop its own investigative thread — not structured, enumerated ones.

---

## Injection strength

Tested weak, medium (the main result above), and strong variants of the
belief statement.

| Strength | Raw AUC | Length-normalized AUC |
|----------|---------|------------------------|
| Weak | 0.843 | 0.177 (reverses) |
| **Medium** | **0.990** | **0.846** (holds) |
| Strong | 0.982 | 0.100 (reverses) |

Only **medium-strength** injection produces a signal that survives removing
the query-count confound. Weak and strong injection mostly just make the
agent search *less* (a volume effect that happens to inflate raw TTR); medium
strength produces genuinely scattered search behavior at a similar search
volume. A detector tuned on one injection strength should not be assumed to
transfer linearly to a stronger or weaker version of the same attack.

---

## Length sensitivity

vocab_breadth, like all type-token-ratio features, is sensitive to how many
queries a session has — fewer queries mechanically inflates the ratio
regardless of real diversity. We quantify this directly rather than assume it
away: length-normalizing with **Guiraud's index** (unique tokens / √total
tokens) drops Task 1's AUC from 0.990 to **0.846** (still far from chance,
95% CI [0.731, 0.946]) — most of the headline signal survives, but the honest
number for Task 1 specifically is "0.990 raw / ≈0.85 length-normalized," not
a single unqualified figure. Task 3 is barely affected (0.9325 → 0.800). We
recommend reporting both the raw and length-normalized statistic as standard
practice for this feature.

---

## Cross-model generalization

Tested on 8 additional models — three closed frontier labs (OpenAI, Google,
and Claude's own family) and four open-weight families (Qwen, Gemma, Mistral,
Llama) — using the same task design, with bootstrap 95% CIs (B=10,000) and
permutation tests (N=100,000) on every point. Any model whose result looked
statistically fragile on a single sample was re-collected with an independent
session before being reported.

| Model | Type | n | AUC | Cohen's d | p-value | Result |
|---|---|---|---|---|---|---|
| Claude Sonnet 4.6 | Closed | 30+30 | 0.990 | +3.91 | <1e-5 | Significant (reference) |
| GPT-4o | Closed | 60+50 | 0.823 | +1.39 | <1e-5 | **Significant, same direction** |
| Qwen2.5-7B | Open | 76+74 | 0.705 | +0.73 | <1e-5 | **Significant, same direction** |
| Qwen2.5-14B | Open | 120+101 | 0.752 | +0.92 | <1e-5 | **Significant, same direction** |
| Qwen2.5-32B | Open | 56+49 | 0.688 | +0.83 | 0.0008 | **Significant, same direction** |
| Gemini 2.5 Flash | Closed | 60+60 | 0.641 | +0.54 | 0.007 | **Significant, same direction** |
| Mistral-Small-24B | Open | 73+57 | 0.578 | +0.30 | 0.130 | Null — no effect |
| Llama-3.1-8B | Open | 60+60 | 0.488 | −0.06 | 0.827 | Null — no effect (confirmed across 2 independent sessions) |
| Gemma2-27B | Open | 120+120 | 0.502 | +0.07 | 0.964 | Null — no effect |

**Summary: the signal generalizes to most of the models tested, and where it
generalizes, it is always in the same direction — no model shows a
significant reversal.** Both closed-source models outside Claude show the
effect, and so does every model size tested within the Qwen family (7B,
14B, 32B all significant, same direction) — the detector's open-weight
reach is driven specifically by that family. Among the other three
open-weight points tested (Mistral, Llama, Gemma), none show an effect.
Both Llama and Gemma's nulls were verified across two or more independent
collection sessions before being reported as stable, rather than taken
from a single sample.

**Magnitude does not transfer even where direction agrees** — every other
model's effect size is substantially weaker than Claude's, so per-model
threshold calibration is required; this should not be deployed against a new
model using Claude's threshold unmodified.

---

## Where the detector does *not* transfer

Three checks against structurally different setups, each showing the same
pattern: a raw signal that looks like a transfer, which does not survive
length normalization.

- **A different injection mechanism** (AgentDrift, a public benchmark where an
  external payload hijacks the agent mid-task, rather than a stated belief):
  raw AUC=0.432 (reversed), driven by the same length confound.
- **A second external benchmark** (ATBench, direct/indirect prompt injection):
  raw AUC=0.702 (looks positive), reverses to 0.349 once length-normalized.
- **A different belief-injection paradigm** — task-*irrelevant* belief
  (Jeong et al.'s own design: an unrelated false belief, rather than one about
  the research topic itself): raw AUC=0.989 (looks like a near-perfect
  transfer), reverses to 0.186 once length-normalized.

**vocab_breadth is a validated signal for task-relevant belief-prefill
injection, on the models and setup tested — not a general-purpose
injection detector.** This is a scope statement, not a weakness: the feature
does one specific thing well rather than many things approximately.

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

Cross-model and transfer-test scripts live in `experiments/agentdrift_transfer/`
(see that directory for the full set — length-normalization checks, the
external-benchmark tests, the semantic-diversity feature, and the cross-model
rigor pipeline).

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
    ├── task2_llm_capabilities/ # Boundary condition experiment (n=20+20)
    └── agentdrift_transfer/    # Cross-model, length-normalization, and
                                 # transfer-test scripts and results
```

---

## Cost

~$0.21/run (Claude Sonnet 4.6 + real Tavily search). Cost driver: ~60k input
tokens/run from search results accumulating in conversation history.

---

## Status

- ✅ Main result: AUC=0.990, 95% CI [0.969, 1.000], p<1e-5, 80% detection at 0% FP
- ✅ Ablation, bootstrap CIs, 5-fold CV, mechanism validation (lexical + semantic + structural) — all complete
- ✅ Operating boundary characterized: requires open-ended tasks
- ✅ Cross-domain replication: Task 3 (UBI) — AUC=0.9325, the least length-confounded result in the project
- ✅ Injection-strength boundary mapped: only medium strength survives length normalization
- ✅ Length sensitivity quantified and reported alongside every headline number (raw vs. length-normalized)
- ✅ Cross-model generalization: 8 additional models, bootstrap CIs + permutation tests + independent-session verification on every point — generalizes to GPT-4o, Gemini, and all three Qwen sizes (7B/14B/32B); null on Mistral, Gemma, Llama-3.1-8B; **no reversal on any model tested**
- ✅ Scope-tested against 3 structurally different setups (AgentDrift, ATBench, task-irrelevant injection) — validated for task-relevant belief-prefill injection specifically, does not transfer to these
- ⏳ Evasion/adversarial robustness — cost analysis done, defenses not yet built
