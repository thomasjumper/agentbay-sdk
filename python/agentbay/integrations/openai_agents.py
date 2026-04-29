"""AgentBay integration for OpenAI Agents SDK.

Adds persistent memory to OpenAI agents. Automatically recalls relevant
context before tool calls and stores learnings after responses.

Usage::

    from agentbay.integrations.openai_agents import AgentBayMemoryTool
    from agents import Agent, Runner

    memory = AgentBayMemoryTool("ab_live_your_key", project_id="your-project")

    agent = Agent(
        name="my-agent",
        instructions="You are a helpful assistant with persistent memory.",
        tools=[memory.search_tool(), memory.store_tool()],
    )

    result = Runner.run_sync(agent, "What did we discuss yesterday?")
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class AgentBayMemoryTool:
    """Memory tools for OpenAI Agents SDK.

    Provides search and store functions as agent tools,
    plus hooks for automatic memory management.

    Args:
        api_key: AgentBay API key.
        project_id: Project ID for scoping memories.
        base_url: API base URL.
    """

    def __init__(
        self,
        api_key: str,
        project_id: str | None = None,
        base_url: str = "https://www.aiagentsbay.com",
    ):
        from agentbay import AgentBay
        self._brain = AgentBay(api_key=api_key, project_id=project_id, base_url=base_url)
        self._project_id = project_id

    def search_tool(self) -> Dict[str, Any]:
        """Returns a tool definition for memory search.

        Compatible with OpenAI Agents SDK tool format.
        """
        brain = self._brain

        def search_memory(query: str, limit: int = 5) -> str:
            """Search persistent memory for relevant context."""
            results = brain.recall(query, limit=limit)
            if not results:
                return "No relevant memories found."
            output = []
            for entry in results:
                conf = int(entry.get("confidence", 0) * 100)
                output.append(f"[{entry.get('type', '?')}] {entry.get('title', '?')} ({conf}%)")
                content = entry.get("content", "")
                if content:
                    output.append(f"  {content[:200]}")
            return "\n".join(output)

        return {
            "type": "function",
            "function": {
                "name": "memory_search",
                "description": "Search your persistent memory for relevant context, patterns, pitfalls, and decisions from previous sessions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "What to search for"},
                        "limit": {"type": "integer", "description": "Max results (default: 5)", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            "_handler": search_memory,
        }

    def store_tool(self) -> Dict[str, Any]:
        """Returns a tool definition for memory storage."""
        brain = self._brain

        def store_memory(content: str, title: str = "", type: str = "PATTERN", tags: str = "") -> str:
            """Store a piece of knowledge to persistent memory."""
            tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
            result = brain.store(
                content,
                title=title or content[:80],
                type=type,
                tags=tag_list,
            )
            return f"Stored: {result.get('id', 'unknown')}"

        return {
            "type": "function",
            "function": {
                "name": "memory_store",
                "description": "Store a pattern, pitfall, decision, or learning to persistent memory for future sessions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The knowledge to remember"},
                        "title": {"type": "string", "description": "Short title"},
                        "type": {"type": "string", "enum": ["PATTERN", "PITFALL", "DECISION", "ARCHITECTURE", "CONTEXT"], "description": "Memory type"},
                        "tags": {"type": "string", "description": "Comma-separated tags"},
                    },
                    "required": ["content"],
                },
            },
            "_handler": store_memory,
        }

    def get_context(self, query: str, limit: int = 3) -> str:
        """Get formatted memory context for injection into system prompt.

        Call this before sending messages to the agent to pre-load context.
        """
        results = self._brain.recall(query, limit=limit)
        if not results:
            return ""

        lines = [f"[AgentBay — {len(results)} memories]"]
        for entry in results:
            conf = int(entry.get("confidence", 0) * 100)
            lines.append(f"\n[{entry.get('type', '?')}] {entry.get('title', '?')} ({conf}%)")
            lines.append(entry.get("content", "")[:300])
        return "\n".join(lines)

    def auto_store(self, content: str, type: str = "PATTERN") -> Optional[Dict]:
        """Store a memory entry. Call after agent produces useful output."""
        if len(content) < 50:
            return None
        title = content.split("\n")[0][:100] if "\n" in content else content[:100]
        return self._brain.store(content, title=title, type=type, tags=["openai-agents", "auto"])
