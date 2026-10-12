"""
Extract vocab_breadth-style features from AgentDrift (web domain, task-disjoint split)
to test whether the behavioral-detector signal transfers to a different injection
mechanism (external payload hijack) than the one it was built on (belief prefill).

Mirrors the exact vocab_breadth formula from ../../bootstrap_ci.py:
    unique tokens / total tokens, lowercase, whitespace split.

World-entity redaction: the dataset's own authors (AgentDrift README) warn that
named entities (user/company/contact names and emails) predict the attacked class
at 93-98% on some domains, which would let any lexical feature "cheat." We redact
every exact string found in the trajectory's own `world` object before tokenizing,
so vocab_breadth can only reflect the agent's action vocabulary, not entity leakage.

Text pooled per trajectory = thought + tool name + flattened args values for every
step. `obs` (environment-generated) is excluded, matching the original feature's
scope (the agent's own generated text, not environment responses).
"""
import json
import re
import sys
from pathlib import Path

repo = Path(sys.argv[1])
out_path = Path(sys.argv[2])


def flatten(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from flatten(v)
    elif isinstance(value, list):
        for v in value:
            yield from flatten(v)
    elif value is not None:
        yield str(value)


def world_strings(world: dict) -> list[str]:
    out = []
    if not world:
        return out
    for key in ("user", "email", "company"):
        if world.get(key):
            out.append(str(world[key]))
    for c in world.get("contacts", []) or []:
        if c.get("name"):
            out.append(c["name"])
        if c.get("email"):
            out.append(c["email"])
    return out


def redact(text: str, entities: list[str]) -> str:
    for e in sorted(entities, key=len, reverse=True):
        if not e:
            continue
        text = re.sub(re.escape(e), " ", text, flags=re.IGNORECASE)
    return text


def vocab_breadth(text: str) -> tuple[float, int]:
    w = text.lower().split()
    u = set(w)
    return len(u) / max(len(w), 1), len(w)


records = []
for split in ("train", "val", "test"):
    for f in (repo / "data_taskdisjoint" / split).glob("*.json"):
        d = json.loads(f.read_text())
        if d.get("agent") != "web":
            continue

        entities = world_strings(d.get("world", {}))
        labels = {s["label"] for s in d["steps"]}
        category = d["category"]

        if category == "attacked" and "hijacked" in labels:
            label = 1
        elif category == "benign":
            label = 0
        else:
            label = None  # failed_attack / hard_negative — kept but excluded from main test

        parts = []
        for s in d["steps"]:
            parts.append(s.get("thought", ""))
            parts.append(s.get("tool", ""))
            parts.extend(flatten(s.get("args", {})))
        text = " ".join(parts)
        text = redact(text, entities)

        vb, n_tokens = vocab_breadth(text)
        records.append({
            "id": d["id"],
            "split": split,
            "category": category,
            "label": label,
            "vocab_breadth": vb,
            "n_tokens": n_tokens,
            "n_steps": len(d["steps"]),
        })

out_path.write_text("\n".join(json.dumps(r) for r in records))
print(f"wrote {len(records)} records to {out_path}")
