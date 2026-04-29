"""AgentBay memory for OpenAI Codex agents.

Provides hooks that inject recalled memories before each completion
and optionally store learnings from responses.

Usage::

    pip install agentbay

    from agentbay.integrations.codex import AgentBayCodexMemory

    memory = AgentBayCodexMemory(api_key="ab_live_...")

    # Wrap messages before sending to Codex
    messages = [{"role": "user", "content": "Fix the auth bug"}]
    enriched = memory.before_completion(messages)

    # After getting a response, store learnings
    memory.after_completion(messages, response_text)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# Patterns that suggest the response contains a learning worth storing
_LEARNING_RE = re.compile(
    r"(?:the (?:issue|problem|bug|fix|solution|cause) (?:was|is))"
    r"|(?:(?:we|i) (?:decided|chose|went with|settled on))"
    r"|(?:(?:always|never|make sure to|remember to))"
    r"|(?:(?:turns out|it works because|the trick is|the key (?:insight|thing)))",
    re.IGNORECASE,
)

_PITFALL_RE = re.compile(
    r"\b(?:bug|error|fix|crash|fail|broke|issue|exception)\b",
    re.IGNORECASE,
)
_DECISION_RE = re.compile(
    r"\b(?:decided|chose|picked|went with|settled on|decision)\b",
    re.IGNORECASE,
)


def _detect_type(text: str) -> str:
    if _PITFALL_RE.search(text):
        return "PITFALL"
    if _DECISION_RE.search(text):
        return "DECISION"
    return "PATTERN"


def _extract_title(text: str, max_len: int = 100) -> str:
    match = re.match(r"^(.+?[.!?])\s", text)
    if match and len(match.group(1)) <= max_len:
        return match.group(1)
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


class AgentBayCodexMemory:
    """Memory hooks for OpenAI Codex agent SDK.

    Provides ``before_completion`` and ``after_completion`` hooks that can be
    wired into Codex agent pipelines to give agents persistent memory.

    Args:
        api_key: Your AgentBay API key (``ab_live_...`` or ``ab_test_...``).
            Falls back to ``AGENTBAY_API_KEY`` env var.
        project_id: Knowledge Brain project ID.
            Falls back to ``AGENTBAY_PROJECT_ID`` env var.
        base_url: API base URL. Defaults to ``https://www.aiagentsbay.com``.
        auto_store: If True, automatically store learnings from responses.
            Defaults to True.
        recall_limit: Max memories to inject. Defaults to 5.

    Example::

        from agentbay.integrations.codex import AgentBayCodexMemory

        memory = AgentBayCodexMemory(api_key="ab_live_your_key")

        # In your Codex agent pipeline:
        messages = memory.before_completion(messages)
        response = codex.complete(messages)
        memory.after_completion(messages, response)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        base_url: str = "https://www.aiagentsbay.com",
        auto_store: bool = True,
        recall_limit: int = 5,
    ) -> None:
        from agentbay.client import AgentBay

        self.client = AgentBay(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
        )
        self.auto_store = auto_store
        self.recall_limit = recall_limit

    def before_completion(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Inject recalled memories into the message list.

        Finds the last user message, uses it as a recall query, and prepends
        any relevant memories as a system message.

        Args:
            messages: The message list (OpenAI chat format).

        Returns:
            A new message list with memories injected (original is not mutated).
        """
        # Find the last user message to use as query
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                query = content if isinstance(content, str) else str(content)
                break

        if not query:
            return list(messages)

        try:
            results = self.client.recall(query=query, limit=self.recall_limit)
        except Exception:
            return list(messages)

        if not results:
            return list(messages)

        # Format memories as context
        memory_lines = []
        for r in results:
            title = r.get("title", "")
            content = r.get("content", r.get("text", ""))
            conf = r.get("confidence", r.get("score", 0))
            prefix = f"[{title}] " if title else ""
            memory_lines.append(f"- {prefix}{content} (confidence: {conf:.2f})")

        memory_block = (
            "Relevant memories from previous sessions:\n"
            + "\n".join(memory_lines)
        )

        # Inject as a system message after any existing system message
        enriched = list(messages)
        insert_idx = 0
        for i, msg in enumerate(enriched):
            if msg.get("role") == "system":
                insert_idx = i + 1
                break

        enriched.insert(insert_idx, {
            "role": "system",
            "content": memory_block,
        })

        return enriched

    def after_completion(
        self,
        messages: List[Dict[str, Any]],
        response: str,
    ) -> None:
        """Optionally store learnings from the response.

        Scans the response for learning patterns (bug fixes, decisions, etc.)
        and stores them if ``auto_store`` is True.

        Args:
            messages: The original message list.
            response: The assistant's response text.
        """
        if not self.auto_store:
            return

        if not _LEARNING_RE.search(response):
            return

        # Extract the user's original question for context
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                user_query = content if isinstance(content, str) else str(content)
                break

        # Store with context
        content = response[:2000]
        if user_query:
            content = f"Q: {user_query[:200]}\nA: {content}"

        try:
            self.client.store(
                content=content,
                title=_extract_title(response),
                type=_detect_type(response),
                tier="semantic",
                tags=["source:codex"],
            )
        except Exception:
            pass  # Don't break the agent pipeline

    def __repr__(self) -> str:
        return f"AgentBayCodexMemory(auto_store={self.auto_store})"
