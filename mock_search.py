"""
Deterministic mock web search tool.

Returns 5 realistic fake results for any query. Results are seeded from
the query hash so every run that issues the same query gets identical
search results — eliminating search randomness as a confound.
"""

import hashlib
import random
from typing import Any


_DOMAINS: list[str] = [
    "reuters.com", "bbc.com", "theguardian.com", "bloomberg.com",
    "ft.com", "economist.com", "nature.com", "science.org",
    "ieee.org", "sciencedirect.com", "iea.org", "irena.org",
    "eia.gov", "eurostat.ec.europa.eu", "ourworldindata.org",
    "carbonbrief.org", "renewableenergyworld.com", "pv-magazine.com",
    "windpowermonthly.com", "euractiv.com", "statista.com",
    "worldbank.org", "imf.org", "mckinsey.com", "deloitte.com",
    "spglobal.com", "woodmackenzie.com", "energymonitor.ai",
    "politico.eu", "eea.europa.eu", "energy.ec.europa.eu",
    "agora-energiewende.org", "ember-climate.org", "cleantechnica.com",
    "globalwindatlas.info",
]

_SNIPPETS: list[str] = [
    "New analysis reveals significant shifts in {topic} trends across the region.",
    "Report finds record growth in {topic} capacity, surpassing previous forecasts.",
    "Experts debate the long-term economic implications of changes in {topic}.",
    "Government policy reforms are accelerating investment in {topic} infrastructure.",
    "Survey of industry stakeholders shows mixed sentiment on {topic} progress.",
    "Comparative study highlights best practices and key barriers for {topic}.",
    "Researchers note wide regional variation that complicates uniform {topic} targets.",
    "New cost data challenges conventional assumptions about the trajectory of {topic}.",
    "International cooperation on {topic} reaches a critical juncture this year.",
    "Latest statistics show {topic} outperforming expectations in several key markets.",
]


def web_search(query: str) -> list[dict[str, Any]]:
    """
    Mock web search — deterministic output seeded from the query string.
    Returns 5 results, each from a unique domain.
    """
    seed = int(hashlib.md5(query.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    pool = _DOMAINS[:]
    rng.shuffle(pool)
    selected = pool[:5]

    topic_hint = " ".join(query.split()[:3]).lower()
    results = []
    for i, domain in enumerate(selected):
        snippet = rng.choice(_SNIPPETS).format(topic=topic_hint)
        slug = _slugify(query)
        results.append({
            "title": f"{query.title()} — {domain.split('.')[0].capitalize()} ({i + 1})",
            "url": f"https://www.{domain}/articles/{slug}-{i + 1}",
            "domain": domain,
            "snippet": snippet,
        })

    return results


def _slugify(text: str) -> str:
    return "-".join(text.lower().split())[:50]
