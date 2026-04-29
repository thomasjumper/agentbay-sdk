"""AgentBay integration for LiveKit Agents.

Adds persistent memory to LiveKit real-time voice/video AI agents.
Recalls context when a participant speaks, stores learnings from responses.

Usage::

    from agentbay.integrations.livekit import AgentBayLiveKitMemory
    from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli

    memory = AgentBayLiveKitMemory("ab_live_your_key", project_id="your-project")

    async def entrypoint(ctx: JobContext):
        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        agent = VoicePipelineAgent(
            before_llm_cb=memory.before_llm,
            # ... other config
        )
        agent.start(ctx.room)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


class AgentBayLiveKitMemory:
    """Persistent memory for LiveKit voice/video agents.

    Provides callbacks for the LiveKit Agents framework:
    - before_llm: inject relevant memories before LLM call
    - after_llm: store learnings from the response
    - on_transcript: recall when participant speaks

    Args:
        api_key: AgentBay API key.
        project_id: Project ID.
        auto_recall: Inject memories before LLM (default: True).
        auto_store: Store learnings after LLM (default: True).
        recall_limit: Max memories to inject (default: 3).
    """

    def __init__(
        self,
        api_key: str,
        project_id: str | None = None,
        base_url: str = "https://www.aiagentsbay.com",
        auto_recall: bool = True,
        auto_store: bool = True,
        recall_limit: int = 3,
    ):
        from agentbay import AgentBay
        self._brain = AgentBay(api_key=api_key, project_id=project_id, base_url=base_url)
        self._auto_recall = auto_recall
        self._auto_store = auto_store
        self._recall_limit = recall_limit

    _LEARN_RE = re.compile(
        r"(?:the (?:issue|problem|fix|solution) (?:is|was))|(?:always |never |decided to|important:)",
        re.IGNORECASE,
    )

    async def before_llm(self, assistant: Any, chat_ctx: Any) -> None:
        """Inject relevant memories before LLM call.

        Use as before_llm_cb in VoicePipelineAgent.
        """
        if not self._auto_recall:
            return

        try:
            # Get the last user message
            messages = chat_ctx.messages if hasattr(chat_ctx, 'messages') else []
            last_user = ""
            for msg in reversed(messages):
                role = msg.role if hasattr(msg, 'role') else msg.get('role', '')
                if role == "user":
                    last_user = msg.content if hasattr(msg, 'content') else msg.get('content', '')
                    break

            if not last_user or len(last_user) < 5:
                return

            results = self._brain.recall(last_user[:300], limit=self._recall_limit)
            if not results:
                return

            memory_text = f"[Relevant context from memory ({len(results)} entries)]\n"
            for entry in results:
                memory_text += f"- {entry.get('title', '?')}: {entry.get('content', '')[:150]}\n"

            # Inject as system message
            if hasattr(chat_ctx, 'append'):
                chat_ctx.append(role="system", text=memory_text)
        except Exception:
            pass

    async def after_llm(self, assistant: Any, chat_ctx: Any, response_text: str) -> None:
        """Store learnings from LLM response.

        Call after the agent responds.
        """
        if not self._auto_store or len(response_text) < 50:
            return

        if not self._LEARN_RE.search(response_text):
            return

        try:
            title = response_text.split("\n")[0][:100] if "\n" in response_text else response_text[:100]
            lower = response_text.lower()
            mem_type = "PITFALL" if any(w in lower for w in ["bug", "error", "fix"]) else "PATTERN"
            self._brain.store(response_text[:2000], title=title, type=mem_type, tags=["livekit", "voice", "auto"])
        except Exception:
            pass

    def recall(self, query: str, limit: int = 3) -> List[Dict]:
        """Manual recall for custom pipelines."""
        return self._brain.recall(query, limit=limit)

    def store(self, content: str, title: str = "", type: str = "PATTERN") -> Dict:
        """Manual store for custom pipelines."""
        return self._brain.store(content, title=title or content[:80], type=type, tags=["livekit"])
