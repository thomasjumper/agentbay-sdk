"""AgentBay memory for Agno autonomous agents.

Provides a memory backend that integrates with Agno's agent memory
interface for persistent recall and storage across sessions.

Usage::

    from agentbay.integrations.agno import AgentBayMemory

    memory = AgentBayMemory(api_key="ab_live_...", project_id="...")

    # Recall relevant context before agent action
    memories = memory.recall("deployment pipeline configuration")

    # Store learnings after agent action
    memory.store("Deploy pipeline uses blue-green strategy with 5min canary.")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class AgentBayMemory:
    """Agno agent memory backed by AgentBay's Knowledge Brain.

    Implements ``recall`` and ``store`` to work with Agno's memory interface
    for autonomous agents.

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

    def recall(
        self, context: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get relevant memories for the given context.

        Args:
            context: Natural-language description of what to recall.
            limit: Maximum results to return (defaults to recall_limit).

        Returns:
            List of dicts with ``content``, ``title``, and ``score`` keys.
        """
        if not context:
            return []
        try:
            results = self._client.recall(
                query=context, limit=limit or self._recall_limit
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

    def store(
        self,
        data: str,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Save a piece of knowledge.

        Args:
            data: The content to remember.
            title: Optional title for the memory entry.
            tags: Optional list of tags for categorization.
        """
        if not data:
            return
        all_tags = list(tags or [])
        all_tags.append("source:agno")
        try:
            self._client.store(
                content=data,
                title=title or None,
                tags=all_tags,
            )
        except Exception:
            pass

    def __repr__(self) -> str:
        return "AgentBayMemory()"
