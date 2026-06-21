"""
ReAct-style research agent backed by Claude Sonnet (Anthropic API).

The agent issues web_search tool calls until it decides it has gathered
enough information or reaches MAX_STEPS. Every tool call is recorded in a
structured JSON log that the monitor later analyses.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic

from real_search import web_search as _real_search

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024
MAX_STEPS = 15

_TOOL_DEF: dict[str, Any] = {
    "name": "web_search",
    "description": (
        "Search the web for information on a topic. "
        "Returns up to 5 results, each with a title, URL, domain, and snippet."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A concise search query string.",
            }
        },
        "required": ["query"],
    },
}


def run_agent(task: str, system_prompt: str, log_path: Path, search_type: str = "real", search_fn=None) -> dict[str, Any]:
    """
    Run the research agent on *task* under *system_prompt*.

    Args:
        task:          The research question posed to the agent.
        system_prompt: System message; may contain an injected belief statement.
        log_path:      Destination path for the JSON run log.
        search_type:   "real" (Tavily) or "mock" (deterministic).

    Returns:
        The complete log dict (also written to disk).
    """
    client = anthropic.Anthropic()
    _search = search_fn if search_fn is not None else _real_search

    log: dict[str, Any] = {
        "task": task,
        "system_prompt": system_prompt,
        "search_type": search_type,
        "steps": [],
        "final_answer": None,
        "total_searches": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
    }

    messages: list[dict[str, Any]] = [{"role": "user", "content": task}]
    step = 0

    while step < MAX_STEPS:
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            tools=[_TOOL_DEF],
            messages=messages,
        )

        # Append the full assistant turn to history (text + tool_use blocks)
        messages.append({"role": "assistant", "content": response.content})
        log["total_input_tokens"] += response.usage.input_tokens
        log["total_output_tokens"] += response.usage.output_tokens

        if response.stop_reason == "end_turn":
            final_text = next(
                (block.text for block in response.content if hasattr(block, "text")),
                "",
            )
            log["final_answer"] = final_text
            break

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                step += 1
                query: str = block.input.get("query", "")
                results = _search(query)

                log["steps"].append({
                    "step": step,
                    "action": "web_search",
                    "query": query,
                    "urls_returned": [r["url"] for r in results],
                    "domains_visited": [r["domain"] for r in results],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                log["total_searches"] += 1

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(results, indent=2),
                })

            messages.append({"role": "user", "content": tool_results})
        else:
            # Unexpected stop reason — surface it and exit cleanly
            log["final_answer"] = f"[stopped: {response.stop_reason}]"
            break

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    return log
