"""AgentBay memory for ElevenLabs voice agents.

Provides memory that integrates with ElevenLabs conversational AI,
giving voice agents persistent recall across sessions.

Usage::

    pip install agentbay[elevenlabs]

    from agentbay.integrations.elevenlabs import AgentBayVoiceMemory

    memory = AgentBayVoiceMemory(api_key="ab_live_...", project_id="...")

    # In your conversation handler:
    context = memory.on_message("What did we discuss last time?")
    # ... pass context to ElevenLabs agent ...
    memory.on_response("We discussed the quarterly report.")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class AgentBayVoiceMemory:
    """Voice agent memory backed by AgentBay's Knowledge Brain.

    Designed for streaming voice conversations with ElevenLabs.
    Call ``on_message`` when user speaks to get relevant context,
    and ``on_response`` after the agent responds to store learnings.

    Args:
        api_key: Your AgentBay API key.
        project_id: The Knowledge Brain project ID.
        base_url: API base URL. Defaults to ``https://www.aiagentsbay.com``.
        recall_limit: Max memories to recall per message (default: 3).
        user_id: Optional user identifier for scoping memories.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        base_url: str = "https://www.aiagentsbay.com",
        recall_limit: int = 3,
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
        self._last_user_text: str = ""

    def on_message(self, user_text: str) -> str:
        """Recall relevant memories when the user speaks.

        Args:
            user_text: Transcribed user speech.

        Returns:
            Formatted context string to inject into the agent prompt.
        """
        self._last_user_text = user_text
        if not user_text:
            return ""

        try:
            results = self._client.recall(query=user_text, limit=self._recall_limit)
        except Exception:
            return ""

        if not results:
            return ""

        parts = []
        for r in results:
            content = r.get("content", r.get("text", ""))
            parts.append(content)
        return "\n".join(parts)

    def on_response(self, agent_text: str) -> None:
        """Store learnings from the agent's response.

        Args:
            agent_text: The agent's spoken response text.
        """
        if not agent_text:
            return

        content = agent_text
        if self._last_user_text:
            content = f"User: {self._last_user_text}\nAgent: {agent_text}"

        tags = ["source:elevenlabs"]
        if self._user_id:
            tags.append(f"user:{self._user_id}")

        try:
            self._client.store(content=content, tags=tags)
        except Exception:
            pass

    def __repr__(self) -> str:
        return "AgentBayVoiceMemory()"
