"""
Real web search via Tavily API.

Drop-in replacement for mock_search.py — same return format:
    [{"url": ..., "domain": ..., "title": ..., "snippet": ...}, ...]
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse

from tavily import TavilyClient

logger = logging.getLogger(__name__)

_client: Optional[TavilyClient] = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        api_key = os.environ.get("TAVILY_API_KEY")
        if not api_key:
            raise EnvironmentError("TAVILY_API_KEY not set")
        _client = TavilyClient(api_key=api_key)
    return _client


def web_search(query: str) -> list[dict[str, Any]]:
    """
    Search the web via Tavily. Returns up to 5 results.
    Falls back to empty list on any API failure.
    """
    ts = datetime.now(timezone.utc).isoformat()
    try:
        client = _get_client()
        response = client.search(query, max_results=5)
        raw = response.get("results", [])
        results = []
        for r in raw:
            url = r.get("url", "")
            domain = urlparse(url).netloc.lstrip("www.")
            results.append({
                "url": url,
                "domain": domain,
                "title": r.get("title", ""),
                "snippet": r.get("content", ""),
            })
        logger.info("[%s] query=%r  results=%d", ts, query, len(results))
        return results
    except Exception as exc:
        logger.error("[%s] Tavily error for query=%r: %s", ts, query, exc)
        return []
