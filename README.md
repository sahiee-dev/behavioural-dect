# Behavioral Detector

Detects when a web-researching AI agent has been compromised by belief injection — 
purely from its tool-call log, with no access to the agent's internal state.

## What it does

A belief-injected agent searches less, visits fewer domains, and asks less varied 
questions — because it already "knows" the answer. This detector measures those 
three behaviors, combines them into a diversity score, and flags statistical 
divergence from a neutral baseline.

Built as a detection layer for the behavioral signal identified in:
> Jeong, Houmansadr, Zilberstein, Bagdasarian — "Persuasion Propagation in LLM Agents" (2026)

## Results (20 runs: 10 neutral, 10 belief-injected)

| Condition | Avg Searches | Score Range | Detection |
|-----------|-------------|-------------|-----------|
| Neutral | 13.1 | 0.64–0.86 | — |
| Belief-injected | 6.9 | 0.10–0.53 | COMPROMISED |

- Detection rate: 10/10 (100%)
- False positive rate: 0/10 (0%)
- Score distributions: zero overlap between conditions
- Detection threshold: z-score < -2.0

## Setup

```
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

## Run

```
# Full experiment (10 neutral + 10 injected runs, ~$1.20)
python injector.py

# Dry run (2+2 runs, ~$0.12)
python injector.py --dry-run

# Analysis and detection report
python monitor.py
```

## Structure

```
behavioral_detector/
├── agent.py          # ReAct agent with Claude Sonnet backend
├── injector.py       # Runs neutral and belief-injected conditions
├── monitor.py        # Diversity scoring, z-score detection, plot output
├── mock_search.py    # Deterministic seeded mock search tool
├── requirements.txt
└── README.md
```

## What the output means

The detection report flags any run with z-score < -2.0 as COMPROMISED.
A z-score of -3.0 means the agent's search diversity was 3 standard deviations 
below the neutral baseline — a strong signal of belief-state manipulation.

Results plot saved to `results/diversity_comparison.png`
