"""AgentBay integration with AgentOps for observability.

Wraps an AgentBay client to log all store/recall operations to AgentOps,
providing timing, success/failure tracking, and operation metadata.

Usage::

    pip install agentbay agentops

    from agentbay import AgentBay
    from agentbay.integrations.agentops import track_memory_ops

    brain = AgentBay("ab_live_your_key", project_id="your-project")
    track_memory_ops(brain, agentops_api_key="your-agentops-key")

    # Now all store/recall calls are automatically logged to AgentOps
    brain.store("some knowledge", title="Example")
    brain.recall("what do I know?")
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Optional


def track_memory_ops(
    client: Any,
    agentops_api_key: Optional[str] = None,
    session_tags: Optional[list[str]] = None,
) -> Any:
    """Wrap an AgentBay client to log operations to AgentOps.

    Monkey-patches the client's ``store`` and ``recall`` methods to record
    events via the AgentOps SDK. Each operation is logged with:

    - Operation type (store / recall)
    - Timing (duration in ms)
    - Success / failure status
    - Result count (for recall)
    - Content preview (for store)

    Args:
        client: An ``AgentBay`` instance to wrap.
        agentops_api_key: AgentOps API key. Falls back to ``AGENTOPS_API_KEY``
            env var if not provided.
        session_tags: Optional tags for the AgentOps session.

    Returns:
        The same client instance (now wrapped).

    Example::

        from agentbay import AgentBay
        from agentbay.integrations.agentops import track_memory_ops

        brain = AgentBay("ab_live_key")
        track_memory_ops(brain, agentops_api_key="ao_key")

        # These calls are now tracked in AgentOps
        brain.store("Next.js 16 + Prisma stack", title="Stack info")
        results = brain.recall("what stack?")
    """
    try:
        import agentops
    except ImportError:
        raise ImportError(
            "agentops is required for this integration. "
            "Install it with: pip install agentops"
        )

    # Initialize AgentOps if not already initialized
    if agentops_api_key:
        agentops.init(api_key=agentops_api_key, tags=session_tags or ["agentbay"])

    original_store = client.store
    original_recall = client.recall

    @wraps(original_store)
    def tracked_store(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        error = None
        result = None
        try:
            result = original_store(*args, **kwargs)
            return result
        except Exception as e:
            error = e
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            content_preview = ""
            if args:
                content_preview = str(args[0])[:100]
            elif "content" in kwargs:
                content_preview = str(kwargs["content"])[:100]

            try:
                agentops.record(agentops.ActionEvent(
                    action_type="agentbay.store",
                    params={
                        "content_preview": content_preview,
                        "title": kwargs.get("title", ""),
                        "type": kwargs.get("type", ""),
                        "tier": kwargs.get("tier", "semantic"),
                    },
                    returns=str(result)[:200] if result else None,
                    result=(
                        agentops.ActionEvent.FAIL if error
                        else agentops.ActionEvent.SUCCESS
                    ),
                    duration=duration_ms,
                ))
            except Exception:
                pass  # Don't break the client if AgentOps fails

    @wraps(original_recall)
    def tracked_recall(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        error = None
        results = None
        try:
            results = original_recall(*args, **kwargs)
            return results
        except Exception as e:
            error = e
            raise
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            query = ""
            if args:
                query = str(args[0])[:100]
            elif "query" in kwargs:
                query = str(kwargs["query"])[:100]

            result_count = len(results) if results and isinstance(results, list) else 0

            try:
                agentops.record(agentops.ActionEvent(
                    action_type="agentbay.recall",
                    params={
                        "query": query,
                        "limit": kwargs.get("limit", 5),
                    },
                    returns=f"{result_count} results",
                    result=(
                        agentops.ActionEvent.FAIL if error
                        else agentops.ActionEvent.SUCCESS
                    ),
                    duration=duration_ms,
                ))
            except Exception:
                pass

    client.store = tracked_store
    client.recall = tracked_recall

    return client
