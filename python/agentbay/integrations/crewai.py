"""AgentBay memory backend for CrewAI.

Provides a drop-in memory backend that gives CrewAI agents persistent,
cross-session memory powered by AgentBay's Knowledge Brain -- semantic
search, confidence decay, and cross-agent sharing included.

Usage::

    pip install agentbay[crewai]

    from agentbay.integrations.crewai import AgentBayMemory
    from crewai import Agent, Crew

    memory = AgentBayMemory(api_key="ab_live_...", project_id="your-project")

    agent = Agent(
        role="Backend Developer",
        goal="Build reliable APIs",
        memory=memory,
        ...
    )

    # Memories persist across sessions and can be shared between agents
    crew = Crew(agents=[agent], tasks=[...])
    crew.kickoff()
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PITFALL_RE = re.compile(
    r"\b(?:bug|error|fix|crash|fail|broke|issue|problem|exception)\b",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(?:decided|chose|picked|went with|settled on|decision)\b",
    re.IGNORECASE,
)
_PROCEDURE_RE = re.compile(
    r"\b(?:step\s*\d|first.*then|how to|procedure|process|workflow)\b",
    re.IGNORECASE,
)


def _detect_type(text: str) -> str:
    """Auto-detect memory type from content."""
    if _PITFALL_RE.search(text):
        return "PITFALL"
    if _DECISION_RE.search(text):
        return "DECISION"
    if _PROCEDURE_RE.search(text):
        return "PROCEDURE"
    return "PATTERN"


def _extract_title(text: str, max_len: int = 100) -> str:
    """Extract a short title from the first sentence."""
    match = re.match(r"^(.+?[.!?])\s", text)
    if match and len(match.group(1)) <= max_len:
        return match.group(1)
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


# ---------------------------------------------------------------------------
# CrewAI Memory Backend
# ---------------------------------------------------------------------------


class AgentBayMemory:
    """Drop-in memory backend for CrewAI agents.

    Implements the CrewAI memory interface (``save``, ``search``, ``reset``)
    backed by AgentBay's Knowledge Brain for persistent, semantic memory.

    Features:
        - Automatic type detection (PATTERN, PITFALL, DECISION, PROCEDURE)
        - Automatic title extraction from content
        - Semantic search with confidence scoring
        - Cross-session persistence
        - Cross-agent memory sharing (within the same project)
        - Confidence decay over time

    Args:
        api_key: Your AgentBay API key (``ab_live_...`` or ``ab_test_...``).
            Falls back to ``AGENTBAY_API_KEY`` env var if not provided.
        project_id: The Knowledge Brain project ID to use.
            Falls back to ``AGENTBAY_PROJECT_ID`` env var if not provided.
        base_url: API base URL. Defaults to ``https://www.aiagentsbay.com``.

    Example::

        from agentbay.integrations.crewai import AgentBayMemory
        from crewai import Agent

        memory = AgentBayMemory(
            api_key="ab_live_your_key",
            project_id="your-project-id",
        )

        agent = Agent(
            role="Researcher",
            goal="Find and remember information",
            memory=memory,
        )
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        base_url: str = "https://www.aiagentsbay.com",
    ) -> None:
        from agentbay.client import AgentBay

        self.client = AgentBay(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
        )
        self.project_id = project_id

    # ------------------------------------------------------------------
    # CrewAI memory interface
    # ------------------------------------------------------------------

    def save(
        self,
        data: str,
        metadata: Optional[Dict[str, Any]] = None,
        agent: Optional[str] = None,
    ) -> None:
        """Store a memory entry.

        Called by CrewAI after task completion to persist learnings.

        Args:
            data: The content to remember.
            metadata: Optional metadata dict. Recognized keys:
                ``title`` -- custom title (auto-extracted if omitted),
                ``type`` -- memory type (auto-detected if omitted),
                ``tier`` -- storage tier (defaults to ``semantic``),
                ``tags`` -- list of tags for categorization.
            agent: Optional agent name/role (stored as a tag).
        """
        metadata = metadata or {}
        tags: List[str] = list(metadata.get("tags", []))
        if agent:
            tags.append(f"agent:{agent}")
        tags.append("source:crewai")

        title = metadata.get("title") or _extract_title(data)
        entry_type = metadata.get("type") or _detect_type(data)
        tier = metadata.get("tier", "semantic")

        self.client.store(
            content=data,
            title=title,
            type=entry_type,
            tier=tier,
            tags=tags or None,
        )

    def search(
        self,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Search memories by semantic similarity.

        Called by CrewAI before task execution to provide context.

        Args:
            query: Natural-language search query.
            limit: Maximum number of results (1-50).
            score_threshold: Minimum confidence score (0.0-1.0).

        Returns:
            List of dicts with ``context`` and ``score`` keys,
            matching the format CrewAI expects.
        """
        results = self.client.recall(query=query, limit=limit)

        formatted = []
        for r in results:
            score = r.get("confidence", r.get("score", 0))
            if score_threshold > 0 and score < score_threshold:
                continue
            formatted.append({
                "context": r.get("content", r.get("text", "")),
                "score": score,
            })

        return formatted

    def reset(self) -> None:
        """Reset is a no-op -- AgentBay memories are persistent by design.

        Use ``client.forget(knowledge_id)`` to archive individual entries
        if you need to remove specific memories.
        """
        pass

    def __repr__(self) -> str:
        return f"AgentBayMemory(project_id={self.project_id!r})"


# Backward compatibility alias
AgentBayCrewAIMemory = AgentBayMemory
