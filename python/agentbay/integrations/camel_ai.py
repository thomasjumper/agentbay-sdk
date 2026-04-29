"""AgentBay memory for Camel AI multi-agent conversations.

Provides a shared memory backend for Camel AI agents, enabling
persistent recall and cross-agent knowledge sharing via user_id scoping.

Usage::

    pip install agentbay[camel]

    from agentbay.integrations.camel_ai import AgentBayMemory

    memory = AgentBayMemory(api_key="ab_live_...", project_id="...")

    # Recall shared context
    results = memory.retrieve("What's the project architecture?")

    # Store conversation context (scoped to an agent)
    memory.store("The frontend uses React with TypeScript.", user_id="architect")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class AgentBayMemory:
    """Camel AI memory backed by AgentBay's Knowledge Brain.

    Implements ``retrieve`` and ``store`` with optional ``user_id`` scoping
    for multi-agent conversations where each agent can have its own
    memory namespace while sharing the same project.

    Args:
        api_key: Your AgentBay API key.
        project_id: The Knowledge Brain project ID.
        base_url: API base URL. Defaults to ``https://www.aiagentsbay.com``.
        recall_limit: Max memories to recall per query (default: 5).
        user_id: Optional default agent/user ID for scoping memories.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        base_url: str = "https://www.aiagentsbay.com",
        recall_limit: int = 5,
        user_id: Optional[str] = None,
    ) -> None:
        from agentbay.client import AgentBay

        self._client = AgentBay(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
        )
        self._recall_limit = recall_limit
        self._user_id = user_id

    def retrieve(
        self,
        query: str,
        limit: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Recall relevant memories from the shared knowledge brain.

        Args:
            query: Natural-language search query.
            limit: Maximum results to return (defaults to recall_limit).
            user_id: Optional agent/user scope (overrides default).

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

    def store(
        self,
        message: str,
        title: Optional[str] = None,
        user_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Store a conversation message or piece of knowledge.

        Args:
            message: The content to remember.
            title: Optional title for the memory entry.
            user_id: Optional agent/user ID (overrides default).
            tags: Optional list of tags for categorization.
        """
        if not message:
            return
        effective_user = user_id or self._user_id
        all_tags = list(tags or [])
        all_tags.append("source:camel-ai")
        if effective_user:
            all_tags.append(f"agent:{effective_user}")
        try:
            self._client.store(
                content=message,
                title=title or None,
                tags=all_tags,
            )
        except Exception:
            pass

    def __repr__(self) -> str:
        uid = self._user_id
        return f"AgentBayMemory(user_id={uid!r})" if uid else "AgentBayMemory()"
