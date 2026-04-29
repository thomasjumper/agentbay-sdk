"""AgentBay memory for AutoGen / AG2.

Provides hook-based memory injection for AutoGen agents. Memories are
automatically recalled before replies and stored after replies, giving
your agents persistent, cross-session memory.

Usage::

    pip install agentbay[autogen]

    from agentbay.integrations.autogen import AgentBayMemory
    import autogen

    memory = AgentBayMemory(api_key="ab_live_...", project_id="your-project")

    assistant = autogen.AssistantAgent(
        "assistant",
        llm_config=llm_config,
    )

    # Register memory hooks
    memory.attach(assistant)

    # Or register hooks manually for more control:
    # assistant.register_hook("process_all_messages_before_reply", memory.before_reply)
    # assistant.register_hook("process_message_before_send", memory.after_reply)

Manual hook usage::

    # If you prefer explicit hook registration:
    assistant.register_hook(
        "process_all_messages_before_reply",
        memory.before_reply,
    )
    assistant.register_hook(
        "process_message_before_send",
        memory.after_reply,
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class AgentBayMemory:
    """Hook-based memory backend for AutoGen / AG2 agents.

    Injects relevant memories before agent replies and stores
    agent outputs as new memories.

    Features:
        - Automatic recall of relevant memories before each reply
        - Automatic storage of agent responses
        - Cross-session persistence via AgentBay's Knowledge Brain
        - Semantic search with confidence scoring
        - Easy attachment via ``attach()`` or manual hook registration

    Args:
        api_key: Your AgentBay API key (``ab_live_...`` or ``ab_test_...``).
            Falls back to ``AGENTBAY_API_KEY`` env var if not provided.
        project_id: The Knowledge Brain project ID.
            Falls back to ``AGENTBAY_PROJECT_ID`` env var if not provided.
        base_url: API base URL. Defaults to ``https://www.aiagentsbay.com``.
        recall_limit: Max memories to inject per reply (default: 3).
        auto_store: Whether to store agent replies automatically (default: True).

    Example::

        import autogen
        from agentbay.integrations.autogen import AgentBayMemory

        memory = AgentBayMemory(
            api_key="ab_live_...",
            project_id="your-project",
        )

        assistant = autogen.AssistantAgent("assistant", llm_config=llm_config)
        memory.attach(assistant)

        user = autogen.UserProxyAgent("user", human_input_mode="NEVER")
        user.initiate_chat(assistant, message="What do you remember about auth bugs?")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        base_url: str = "https://www.aiagentsbay.com",
        recall_limit: int = 3,
        auto_store: bool = True,
    ) -> None:
        from agentbay.client import AgentBay

        self.brain = AgentBay(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
        )
        self.recall_limit = recall_limit
        self.auto_store = auto_store

    def attach(self, agent: Any) -> None:
        """Register memory hooks on an AutoGen agent.

        Convenience method that registers both ``before_reply`` and
        ``after_reply`` hooks on the given agent.

        Args:
            agent: An AutoGen agent instance (AssistantAgent, etc.).
        """
        agent.register_hook(
            "process_all_messages_before_reply",
            self.before_reply,
        )
        if self.auto_store:
            agent.register_hook(
                "process_message_before_send",
                self.after_reply,
            )

    def before_reply(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Inject relevant memories before the agent replies.

        Registered as a ``process_all_messages_before_reply`` hook.
        Searches AgentBay for memories relevant to the last message
        and prepends them as a system message.

        Args:
            messages: The conversation messages so far.

        Returns:
            Messages with memory context prepended (if any found).
        """
        if not messages:
            return messages

        # Extract the last user/assistant message content
        last_content = ""
        for msg in reversed(messages):
            content = msg.get("content", "")
            if content and isinstance(content, str):
                last_content = content
                break

        if not last_content:
            return messages

        try:
            memories = self.brain.recall(
                query=last_content,
                limit=self.recall_limit,
            )
        except Exception:
            # Don't break the conversation if memory recall fails
            return messages

        if not memories:
            return messages

        # Format memories as context
        parts = []
        for m in memories:
            title = m.get("title", "")
            content = m.get("content", m.get("text", ""))
            confidence = m.get("confidence", m.get("score", ""))
            if title:
                parts.append(f"- [{title}] (confidence: {confidence}): {content}")
            else:
                parts.append(f"- {content}")

        context = "\n".join(parts)
        memory_msg: Dict[str, Any] = {
            "role": "system",
            "content": f"[Relevant memories from AgentBay]\n{context}",
        }

        return [memory_msg] + list(messages)

    def after_reply(
        self,
        message: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Store the agent's reply as a memory.

        Registered as a ``process_message_before_send`` hook.

        Args:
            message: The message about to be sent.

        Returns:
            The message unchanged (pass-through).
        """
        content = message.get("content", "")
        if content and isinstance(content, str) and len(content) > 10:
            try:
                self.brain.add(content)
            except Exception:
                # Don't break the conversation if storage fails
                pass

        return message

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Manually search memories.

        Useful for explicit recall outside the hook system.

        Args:
            query: Natural-language search query.
            limit: Maximum number of results.

        Returns:
            List of matching memory entries.
        """
        return self.brain.recall(query=query, limit=limit)

    def store(self, content: str, **kwargs: Any) -> Dict[str, Any]:
        """Manually store a memory.

        Useful for explicit storage outside the hook system.

        Args:
            content: The knowledge to store.
            **kwargs: Extra args passed to ``brain.store()``.

        Returns:
            Dict with the created entry.
        """
        return self.brain.store(content=content, **kwargs)

    def __repr__(self) -> str:
        return f"AgentBayMemory(recall_limit={self.recall_limit})"
