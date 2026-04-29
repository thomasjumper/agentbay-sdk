"""AgentBay integration for Tavily Search.

Wraps Tavily web search with persistent memory. Search results are
automatically stored so repeated searches on the same topic don't
waste API credits.

Usage::

    from agentbay.integrations.tavily import MemoryEnhancedSearch

    search = MemoryEnhancedSearch(
        tavily_api_key="tvly-...",
        agentbay_api_key="ab_live_...",
        project_id="your-project"
    )

    # First call: searches Tavily, stores results in AgentBay
    results = search.search("latest Next.js features")

    # Second call: returns from memory (no Tavily API call)
    results = search.search("latest Next.js features")
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional


class MemoryEnhancedSearch:
    """Tavily search with AgentBay memory caching.

    First checks AgentBay for cached results. If not found,
    searches Tavily and stores the results for future queries.

    Saves Tavily API credits on repeated/similar queries.

    Args:
        tavily_api_key: Tavily API key.
        agentbay_api_key: AgentBay API key.
        project_id: Project ID for memory scoping.
        cache_ttl_hours: How long cached results are considered fresh (default: 24).
    """

    def __init__(
        self,
        tavily_api_key: str,
        agentbay_api_key: str,
        project_id: str | None = None,
        base_url: str = "https://www.aiagentsbay.com",
        cache_ttl_hours: int = 24,
    ):
        from agentbay import AgentBay
        self._brain = AgentBay(api_key=agentbay_api_key, project_id=project_id, base_url=base_url)
        self._tavily_key = tavily_api_key
        self._cache_ttl = cache_ttl_hours

    def search(self, query: str, max_results: int = 5, search_depth: str = "basic") -> List[Dict[str, Any]]:
        """Search with memory caching.

        1. Check AgentBay for cached results matching the query
        2. If found and fresh → return from memory (free)
        3. If not found → search Tavily, store in memory, return
        """
        # Check memory first
        cached = self._brain.recall(f"tavily:{query}", limit=1)
        if cached:
            top = cached[0]
            # Check if it's a tavily cache entry
            if top.get("type") == "CONTEXT" and "tavily-cache" in (top.get("tags") or []):
                try:
                    return json.loads(top.get("content", "[]"))
                except (json.JSONDecodeError, TypeError):
                    pass

        # Not cached — call Tavily
        results = self._call_tavily(query, max_results, search_depth)

        # Store in AgentBay for future queries
        if results:
            self._brain.store(
                json.dumps(results),
                title=f"tavily:{query}",
                type="CONTEXT",
                tier="working",  # Short-lived cache (24h TTL)
                tags=["tavily-cache", "web-search"],
            )

        return results

    def search_and_remember(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Search and store a permanent summary (not just cache).

        Unlike search(), this creates a lasting PATTERN entry from
        the search results — useful for research that should persist.
        """
        results = self.search(query, max_results)
        if not results:
            return {"results": [], "stored": False}

        # Create a summary
        summary_parts = [f"Web search: {query}"]
        for r in results[:3]:
            summary_parts.append(f"- {r.get('title', '?')}: {r.get('content', '')[:200]}")
        summary = "\n".join(summary_parts)

        stored = self._brain.store(
            summary,
            title=f"Research: {query}",
            type="CONTEXT",
            tier="semantic",  # Permanent
            tags=["web-research", "tavily"],
        )

        return {"results": results, "stored": True, "memory_id": stored.get("id")}

    def _call_tavily(self, query: str, max_results: int, search_depth: str) -> List[Dict]:
        """Call Tavily Search API."""
        import requests

        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self._tavily_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": search_depth,
                },
                timeout=15,
            )
            if resp.ok:
                data = resp.json()
                return data.get("results", [])
        except Exception:
            pass
        return []
