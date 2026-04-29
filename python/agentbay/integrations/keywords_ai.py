"""AgentBay integration for Keywords AI.

Adds memory operation tracking to Keywords AI's LLM monitoring.
Every store and recall is logged with latency, success/failure,
and metadata for observability dashboards.

Usage::

    from agentbay.integrations.keywords_ai import track_with_keywords_ai

    brain = AgentBay("ab_live_your_key")
    tracked_brain = track_with_keywords_ai(brain, keywords_api_key="kw-...")

    # All memory ops are now tracked in Keywords AI
    tracked_brain.store("pattern", title="test")
    tracked_brain.recall("pattern")
"""

from __future__ import annotations

import time
import json
from typing import Any, Dict, Optional


def track_with_keywords_ai(
    brain: Any,
    keywords_api_key: str,
    customer_id: str = "agentbay",
) -> Any:
    """Wrap an AgentBay instance with Keywords AI tracking.

    Every store() and recall() call is logged to Keywords AI
    with latency, success/failure, and metadata.

    Args:
        brain: AgentBay instance.
        keywords_api_key: Keywords AI API key.
        customer_id: Customer identifier for tracking.

    Returns:
        The same brain instance with patched methods.
    """
    import requests

    original_store = brain.store
    original_recall = brain.recall

    def _log(event_type: str, query: str, latency_ms: float, success: bool, result_count: int = 0):
        """Send event to Keywords AI."""
        try:
            requests.post(
                "https://api.keywordsai.co/api/request/log",
                json={
                    "customer_identifier": customer_id,
                    "metadata": {
                        "source": "agentbay",
                        "event_type": event_type,
                        "query": query[:200],
                        "latency_ms": round(latency_ms, 1),
                        "success": success,
                        "result_count": result_count,
                    },
                },
                headers={"Authorization": f"Bearer {keywords_api_key}"},
                timeout=3,
            )
        except Exception:
            pass

    def tracked_store(content: str, **kwargs: Any) -> Dict:
        start = time.time()
        try:
            result = original_store(content, **kwargs)
            _log("memory_store", kwargs.get("title", content[:80]), (time.time() - start) * 1000, True)
            return result
        except Exception as e:
            _log("memory_store", kwargs.get("title", content[:80]), (time.time() - start) * 1000, False)
            raise

    def tracked_recall(query: str, **kwargs: Any):
        start = time.time()
        try:
            results = original_recall(query, **kwargs)
            count = len(results) if isinstance(results, list) else 0
            _log("memory_recall", query, (time.time() - start) * 1000, True, count)
            return results
        except Exception as e:
            _log("memory_recall", query, (time.time() - start) * 1000, False)
            raise

    brain.store = tracked_store
    brain.recall = tracked_recall
    return brain
