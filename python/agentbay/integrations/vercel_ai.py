"""AgentBay memory provider for Vercel AI SDK (Python).

Provides a memory provider that integrates with the Vercel AI SDK's
context and middleware patterns for persistent agent memory.

Usage::

    from agentbay.integrations.vercel_ai import AgentBayProvider

    provider = AgentBayProvider(api_key="ab_live_...", project_id="...")

    # Get context for a conversation
    context = provider.get_context(messages)

    # Save learnings from a response
    provider.save_context(messages, response)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class AgentBayProvider:
    """Vercel AI SDK memory provider backed by AgentBay's Knowledge Brain.

    Returns memories as formatted context strings suitable for injection
    into ``streamText`` / ``generateText`` system prompts or middleware.

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

    def get_context(self, messages: List[Dict[str, str]]) -> str:
        """Recall relevant memories based on conversation messages.

        Extracts the latest user message and searches for relevant context.

        Args:
            messages: List of message dicts with ``role`` and ``content`` keys.

        Returns:
            Formatted context string to prepend to system prompt.
        """
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break
        if not query:
            return ""

        try:
            results = self._client.recall(query=query, limit=self._recall_limit)
        except Exception:
            return ""

        if not results:
            return ""

        parts = ["[Relevant memories from AgentBay]"]
        for r in results:
            title = r.get("title", "")
            content = r.get("content", r.get("text", ""))
            if title:
                parts.append(f"- {title}: {content}")
            else:
                parts.append(f"- {content}")
        return "\n".join(parts)

    def save_context(
        self,
        messages: List[Dict[str, str]],
        response: str,
    ) -> None:
        """Store conversation context as a memory.

        Args:
            messages: The conversation messages.
            response: The assistant's response to store.
        """
        if not response:
            return

        # Extract last user message for context
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        content = f"Q: {user_msg}\nA: {response}" if user_msg else response
        try:
            self._client.store(
                content=content,
                tags=["source:vercel-ai"],
            )
        except Exception:
            pass

    def __repr__(self) -> str:
        return "AgentBayProvider()"
