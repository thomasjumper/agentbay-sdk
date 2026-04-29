"""AgentBay memory for LlamaIndex.

Provides a memory class that integrates AgentBay's Knowledge Brain with
LlamaIndex agents and query engines for persistent, semantic memory.

Usage::

    pip install agentbay[llamaindex]

    from agentbay.integrations.llamaindex import AgentBayMemory

    memory = AgentBayMemory(api_key="ab_live_...", project_id="...")

    # Use with a chat engine or agent
    context = memory.get("What do we know about the auth system?")
    memory.put("The auth system uses JWT tokens with 24h expiry.")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional LlamaIndex base class
# ---------------------------------------------------------------------------

try:
    from llama_index.core.memory import BaseMemory as _LIBaseMemory

    _HAS_LLAMAINDEX = True
except ImportError:
    _LIBaseMemory = object  # type: ignore[assignment,misc]
    _HAS_LLAMAINDEX = False


class AgentBayMemory(_LIBaseMemory):
    """LlamaIndex memory backed by AgentBay's Knowledge Brain.

    Implements ``get``, ``put``, ``get_all``, and ``reset`` to work as a
    drop-in memory class for LlamaIndex chat engines and agents.

    Args:
        api_key: Your AgentBay API key (``ab_live_...`` or ``ab_test_...``).
            Falls back to ``AGENTBAY_API_KEY`` env var if not provided.
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
        if _HAS_LLAMAINDEX:
            super().__init__()

        from agentbay.client import AgentBay

        self._client = AgentBay(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
        )
        self._project_id = project_id
        self._recall_limit = recall_limit

    def get(self, input: str, **kwargs: Any) -> str:
        """Recall relevant memories for the given input.

        Args:
            input: Natural-language query or user message.

        Returns:
            Formatted string of relevant memories, or empty string.
        """
        if not input:
            return ""
        try:
            results = self._client.recall(query=input, limit=self._recall_limit)
        except Exception:
            return ""
        if not results:
            return ""
        parts = []
        for r in results:
            title = r.get("title", "")
            content = r.get("content", r.get("text", ""))
            confidence = r.get("confidence", r.get("score", ""))
            if title:
                parts.append(f"[{title}] (confidence: {confidence})\n{content}")
            else:
                parts.append(content)
        return "\n\n".join(parts)

    def put(self, response: str, **kwargs: Any) -> None:
        """Store a response or piece of knowledge.

        Args:
            response: The content to remember.
        """
        if not response:
            return
        try:
            self._client.store(
                content=response,
                tags=["source:llamaindex"],
            )
        except Exception:
            pass

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all memories (performs a broad recall).

        Returns:
            List of memory dicts with ``content`` and ``score`` keys.
        """
        try:
            results = self._client.recall(query="*", limit=50)
        except Exception:
            return []
        return [
            {
                "content": r.get("content", r.get("text", "")),
                "score": r.get("confidence", r.get("score", 0)),
            }
            for r in results
        ]

    def reset(self) -> None:
        """No-op -- AgentBay memories are persistent by design."""
        pass

    def __repr__(self) -> str:
        return f"AgentBayMemory(project_id={self._project_id!r})"
