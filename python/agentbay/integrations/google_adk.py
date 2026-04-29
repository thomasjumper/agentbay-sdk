"""AgentBay integration for Google Agent Development Kit (ADK).

Adds persistent memory to Google ADK agents. Provides memory tools
and context injection for multi-turn conversations.

Usage::

    from agentbay.integrations.google_adk import AgentBayMemory
    from google.adk import Agent

    memory = AgentBayMemory("ab_live_your_key", project_id="your-project")

    agent = Agent(
        name="my-agent",
        model="gemini-2.0-flash",
        tools=[memory.as_tool()],
        before_model_callback=memory.before_callback,
        after_model_callback=memory.after_callback,
    )
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional


class AgentBayMemory:
    """Persistent memory for Google ADK agents.

    Provides:
    - Tool for agents to search/store memories
    - Before callback: inject relevant memories
    - After callback: auto-extract learnings

    Args:
        api_key: AgentBay API key.
        project_id: Project ID for scoping.
        auto_recall: Inject memories before model calls (default: True).
        auto_store: Extract learnings after model calls (default: True).
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

    # Learnable content patterns
    _LEARN_PATTERNS = re.compile(
        r"(?:the (?:issue|problem|fix|solution|pattern) (?:is|was))"
        r"|(?:always |never |decided to|bug:|pitfall:|important:)",
        re.IGNORECASE,
    )

    def as_tool(self) -> Dict[str, Any]:
        """Returns a Google ADK-compatible tool definition."""
        brain = self._brain

        def memory_tool(action: str, query: str = "", content: str = "", title: str = "", type: str = "PATTERN") -> str:
            """Persistent memory tool. Actions: search, store, health.

            Args:
                action: 'search' to find memories, 'store' to save, 'health' for stats.
                query: Search query (for search action).
                content: Content to store (for store action).
                title: Title for stored memory.
                type: PATTERN, PITFALL, DECISION, ARCHITECTURE, CONTEXT.
            """
            if action == "search":
                results = brain.recall(query or "recent", limit=5)
                if not results:
                    return "No memories found."
                return "\n".join(
                    f"[{e.get('type','?')}] {e.get('title','?')}: {e.get('content','')[:150]}"
                    for e in results
                )
            elif action == "store":
                if not content:
                    return "Error: content is required for store action."
                result = brain.store(content, title=title or content[:80], type=type, tags=["google-adk"])
                return f"Stored: {result.get('id', 'unknown')}"
            elif action == "health":
                h = brain.health()
                return f"Entries: {h.get('total_entries', 0)}, Search methods: {h.get('search_methods', [])}"
            return f"Unknown action: {action}. Use: search, store, health."

        return {
            "name": "persistent_memory",
            "description": "Search and store persistent knowledge across sessions. Use 'search' to find past patterns/decisions, 'store' to save new learnings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["search", "store", "health"]},
                    "query": {"type": "string", "description": "Search query"},
                    "content": {"type": "string", "description": "Content to store"},
                    "title": {"type": "string", "description": "Memory title"},
                    "type": {"type": "string", "enum": ["PATTERN", "PITFALL", "DECISION", "ARCHITECTURE", "CONTEXT"]},
                },
                "required": ["action"],
            },
            "_handler": memory_tool,
        }

    def before_callback(self, messages: List[Dict], **kwargs: Any) -> List[Dict]:
        """Inject relevant memories before model call.

        Use as before_model_callback in Google ADK Agent.
        """
        if not self._auto_recall or not messages:
            return messages

        # Extract last user message for recall query
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                parts = msg.get("parts", [])
                last_user = parts[0].get("text", "") if parts else str(msg.get("content", ""))
                break

        if not last_user or len(last_user) < 3:
            return messages

        try:
            results = self._brain.recall(last_user[:500], limit=self._recall_limit)
            if not results:
                return messages

            memory_text = f"[AgentBay — {len(results)} memories]\n"
            for entry in results:
                conf = int(entry.get("confidence", 0) * 100)
                memory_text += f"\n[{entry.get('type','?')}] {entry.get('title','?')} ({conf}%)\n"
                memory_text += entry.get("content", "")[:200] + "\n"

            # Inject as system message at the beginning
            memory_msg = {"role": "model", "parts": [{"text": memory_text}]}
            return [memory_msg] + messages
        except Exception:
            return messages

    def after_callback(self, response: Any, **kwargs: Any) -> None:
        """Extract and store learnings from model response.

        Use as after_model_callback in Google ADK Agent.
        """
        if not self._auto_store:
            return

        try:
            # Extract text from response
            text = ""
            if hasattr(response, "text"):
                text = response.text
            elif hasattr(response, "parts"):
                text = " ".join(p.get("text", "") for p in response.parts if isinstance(p, dict))
            elif isinstance(response, str):
                text = response

            if len(text) < 50:
                return

            if not self._LEARN_PATTERNS.search(text):
                return

            # Auto-detect type
            lower = text.lower()
            if any(w in lower for w in ["bug", "error", "fix", "crash"]):
                mem_type = "PITFALL"
            elif any(w in lower for w in ["decided", "chose", "went with"]):
                mem_type = "DECISION"
            else:
                mem_type = "PATTERN"

            title = text.split("\n")[0][:100] if "\n" in text else text[:100]
            self._brain.store(text[:2000], title=title, type=mem_type, tags=["google-adk", "auto"])
        except Exception:
            pass
