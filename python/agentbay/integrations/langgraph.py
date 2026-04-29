"""AgentBay memory for LangGraph.

Provides a checkpointer that persists LangGraph state to AgentBay's
Knowledge Brain, enabling cross-session and cross-agent state sharing.

Usage::

    pip install agentbay[langgraph]

    from agentbay.integrations.langgraph import AgentBayCheckpointer
    from langgraph.graph import StateGraph

    checkpointer = AgentBayCheckpointer(api_key="ab_live_...", project_id="...")
    graph = StateGraph(...).compile(checkpointer=checkpointer)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Optional LangGraph base class
# ---------------------------------------------------------------------------

try:
    from langgraph.checkpoint.base import BaseCheckpointSaver

    _HAS_LANGGRAPH = True
except ImportError:
    BaseCheckpointSaver = object  # type: ignore[assignment,misc]
    _HAS_LANGGRAPH = False


def _config_key(config: Dict[str, Any]) -> str:
    """Build a stable key from a LangGraph config dict."""
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    checkpoint_ns = config.get("configurable", {}).get("checkpoint_ns", "")
    return f"langgraph:{thread_id}:{checkpoint_ns}" if checkpoint_ns else f"langgraph:{thread_id}"


class AgentBayCheckpointer(BaseCheckpointSaver):
    """LangGraph checkpointer backed by AgentBay's Knowledge Brain.

    Stores graph state as CONTEXT-type memory entries keyed by thread ID,
    enabling persistent and cross-agent state sharing.

    Args:
        api_key: Your AgentBay API key.
        project_id: The Knowledge Brain project ID.
        base_url: API base URL. Defaults to ``https://www.aiagentsbay.com``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        base_url: str = "https://www.aiagentsbay.com",
    ) -> None:
        if not _HAS_LANGGRAPH:
            raise ImportError(
                "langgraph is required for AgentBayCheckpointer. "
                "Install it with: pip install agentbay[langgraph]"
            )
        super().__init__()

        from agentbay.client import AgentBay

        self._client = AgentBay(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
        )

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        new_versions: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Store a checkpoint as a memory entry.

        Args:
            config: LangGraph config with thread_id.
            checkpoint: The graph state to persist.
            metadata: Optional metadata about the checkpoint.

        Returns:
            Updated config dict.
        """
        key = _config_key(config)
        payload = json.dumps({
            "checkpoint": checkpoint,
            "metadata": metadata or {},
        })
        try:
            self._client.store(
                content=payload,
                title=key,
                type="CONTEXT",
                tags=["source:langgraph", f"thread:{key}"],
            )
        except Exception:
            pass
        return config

    def get_tuple(self, config: Dict[str, Any]) -> Optional[Any]:
        """Retrieve the latest checkpoint for a config.

        Args:
            config: LangGraph config with thread_id.

        Returns:
            CheckpointTuple or None if not found.
        """
        key = _config_key(config)
        try:
            results = self._client.recall(query=key, limit=1)
        except Exception:
            return None
        if not results:
            return None
        content = results[0].get("content", results[0].get("text", ""))
        try:
            data = json.loads(content)
            checkpoint = data.get("checkpoint", {})
            metadata = data.get("metadata", {})
        except (json.JSONDecodeError, AttributeError):
            return None

        if _HAS_LANGGRAPH:
            from langgraph.checkpoint.base import CheckpointTuple
            return CheckpointTuple(
                config=config,
                checkpoint=checkpoint,
                metadata=metadata,
            )
        return None

    def list(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> Iterator[Any]:
        """List checkpoints for a config.

        Args:
            config: LangGraph config to filter by.
            limit: Maximum number of checkpoints to return.

        Yields:
            CheckpointTuple instances.
        """
        key = _config_key(config) if config else "langgraph:"
        try:
            results = self._client.recall(query=key, limit=limit)
        except Exception:
            return
        for r in results:
            content = r.get("content", r.get("text", ""))
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, AttributeError):
                continue
            if _HAS_LANGGRAPH:
                from langgraph.checkpoint.base import CheckpointTuple
                yield CheckpointTuple(
                    config=config or {},
                    checkpoint=data.get("checkpoint", {}),
                    metadata=data.get("metadata", {}),
                )

    def __repr__(self) -> str:
        return "AgentBayCheckpointer()"
