"""AgentBay memory for Mastra agents.

Provides a memory provider that integrates with Mastra's agent memory
interface for persistent, semantic recall across sessions.

Usage::

    from agentbay.integrations.mastra import AgentBayMemoryProvider

    memory = AgentBayMemoryProvider(api_key="ab_live_...", project_id="...")

    # Recall relevant context
    results = memory.get_memory("What's our API rate limit policy?")

    # Save new knowledge
    memory.save_memory("API rate limit is 100 req/min per user.")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class AgentBayMemoryProvider:
    """Mastra memory provider backed by AgentBay's Knowledge Brain.

    Implements ``get_memory`` and ``save_memory`` to work as a standard
    memory provider for Mastra agents.

    Args:
        api_key: Your AgentBay API key.
        project_id: The Knowledge Brain project ID.
        base_url: API base URL. Defaults to ``https://www.aiagentsbay.com``.
        recall_limit: Max memories to recall per query (default: 5).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        base_url: str = "https://www.aiagentsbay.com",
        recall_limit: int = 5,
    ) -> None:
        from agentbay.client import AgentBay

        self._client = AgentBay(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
        )
        self._recall_limit = recall_limit

    def get_memory(self, query: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Recall relevant memories for the given query.

        Args:
            query: Natural-language search query.
            limit: Maximum results to return (defaults to recall_limit).

        Returns:
            List of dicts with ``content``, ``title``, and ``score`` keys.
        """
        if not query:
            return []
        try:
            results = self._client.recall(
                query=query, limit=limit or self._recall_limit
            )
        except Exception:
            return []
        return [
            {
                "content": r.get("content", r.get("text", "")),
                "title": r.get("title", ""),
                "score": r.get("confidence", r.get("score", 0)),
            }
            for r in results
        ]

    def save_memory(
        self,
        data: str,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Store a piece of knowledge.

        Args:
            data: The content to remember.
            title: Optional title for the memory entry.
            tags: Optional list of tags for categorization.
        """
        if not data:
            return
        all_tags = list(tags or [])
        all_tags.append("source:mastra")
        try:
            self._client.store(
                content=data,
                title=title or None,
                tags=all_tags,
            )
        except Exception:
            pass

    def __repr__(self) -> str:
        return "AgentBayMemoryProvider()"
