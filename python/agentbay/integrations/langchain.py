"""AgentBay memory for LangChain.

Provides both a BaseMemory backend (for automatic context injection) and
a Tool (for agent-controlled store/recall). Use whichever fits your pattern.

Usage with BaseMemory (automatic)::

    pip install agentbay[langchain]

    from agentbay.integrations.langchain import AgentBayMemory
    from langchain.chains import ConversationChain
    from langchain_openai import ChatOpenAI

    memory = AgentBayMemory(api_key="ab_live_...", project_id="your-project")

    chain = ConversationChain(llm=ChatOpenAI(), memory=memory)
    chain.predict(input="What do you know about our auth system?")

Usage with Tool (agent-controlled)::

    from agentbay.integrations.langchain import AgentBayMemoryTool
    from langchain.agents import initialize_agent, AgentType
    from langchain_openai import ChatOpenAI

    tool = AgentBayMemoryTool(api_key="ab_live_...", project_id="your-project")

    agent = initialize_agent(
        tools=[tool],
        llm=ChatOpenAI(),
        agent=AgentType.OPENAI_FUNCTIONS,
    )
    agent.run("Remember that the deploy key is stored in 1Password")
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type

from agentbay.client import AgentBay

# ---------------------------------------------------------------------------
# BaseMemory backend (automatic context injection)
# ---------------------------------------------------------------------------

try:
    from langchain_core.memory import BaseMemory as _LCBaseMemory

    _HAS_LANGCHAIN_MEMORY = True
except ImportError:
    _HAS_LANGCHAIN_MEMORY = False

try:
    from langchain_core.callbacks import CallbackManagerForToolRun
    from langchain_core.tools import BaseTool
    from pydantic import BaseModel, Field

    _HAS_LANGCHAIN_TOOLS = True
except ImportError:
    _HAS_LANGCHAIN_TOOLS = False


# ---------------------------------------------------------------------------
# BaseMemory implementation
# ---------------------------------------------------------------------------

if _HAS_LANGCHAIN_MEMORY:

    class AgentBayMemory(_LCBaseMemory):
        """LangChain BaseMemory backed by AgentBay's Knowledge Brain.

        Automatically recalls relevant memories before each LLM call and
        stores conversation turns after each call.

        Features:
            - Semantic search for relevant context before each LLM call
            - Automatic storage of conversation turns
            - Cross-session persistence
            - Configurable memory key, input/output keys
            - Confidence-scored recall with configurable limits

        Args:
            api_key: Your AgentBay API key.
            project_id: The Knowledge Brain project ID.
            base_url: API base URL (optional).
            memory_key: Variable name for injected context (default: ``agentbay_context``).
            input_key: Key for user input in the inputs dict (default: ``input``).
            output_key: Key for LLM output in the outputs dict (default: ``output``).
            recall_limit: Max memories to recall per query (default: 3).
            auto_store: Whether to automatically store conversation turns (default: True).

        Example::

            from agentbay.integrations.langchain import AgentBayMemory
            from langchain.chains import ConversationChain

            memory = AgentBayMemory(
                api_key="ab_live_...",
                project_id="your-project",
                memory_key="context",
            )
            chain = ConversationChain(llm=llm, memory=memory)
        """

        # Pydantic fields
        brain: Any = None
        memory_key: str = "agentbay_context"
        input_key: str = "input"
        output_key: str = "output"
        recall_limit: int = 3
        auto_store: bool = True

        class Config:
            arbitrary_types_allowed = True

        def __init__(
            self,
            api_key: Optional[str] = None,
            project_id: Optional[str] = None,
            base_url: str = "https://www.aiagentsbay.com",
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.brain = AgentBay(
                api_key=api_key,
                base_url=base_url,
                project_id=project_id,
            )

        @property
        def memory_variables(self) -> List[str]:
            """Return the list of memory variables this memory exposes."""
            return [self.memory_key]

        def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
            """Recall relevant memories based on the current input.

            Called by LangChain before each LLM invocation.

            Args:
                inputs: The current chain inputs.

            Returns:
                Dict with the memory_key mapped to recalled context.
            """
            query = inputs.get(self.input_key, "")
            if not query:
                return {self.memory_key: ""}

            try:
                results = self.brain.recall(query=query, limit=self.recall_limit)
            except Exception:
                # Don't break the chain if memory recall fails
                return {self.memory_key: ""}

            if not results:
                return {self.memory_key: ""}

            # Format results as readable context
            parts = []
            for r in results:
                title = r.get("title", "")
                content = r.get("content", r.get("text", ""))
                confidence = r.get("confidence", r.get("score", ""))
                if title:
                    parts.append(f"[{title}] (confidence: {confidence})\n{content}")
                else:
                    parts.append(content)

            return {self.memory_key: "\n\n".join(parts)}

        def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
            """Store the conversation turn as a memory.

            Called by LangChain after each LLM invocation.

            Args:
                inputs: The chain inputs (contains user query).
                outputs: The chain outputs (contains LLM response).
            """
            if not self.auto_store:
                return

            input_text = inputs.get(self.input_key, "")
            output_text = outputs.get(self.output_key, "")

            if not input_text or not output_text:
                return

            try:
                self.brain.add(f"Q: {input_text}\nA: {output_text}")
            except Exception:
                # Don't break the chain if memory storage fails
                pass

        def clear(self) -> None:
            """Clear is a no-op -- AgentBay memories are persistent by design.

            Use ``brain.forget(knowledge_id)`` to archive individual entries.
            """
            pass

else:

    class AgentBayMemory:  # type: ignore[no-redef]
        """Placeholder when langchain-core is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "langchain-core is required for AgentBayMemory. "
                "Install it with: pip install agentbay[langchain]"
            )


# ---------------------------------------------------------------------------
# Tool implementation (agent-controlled store/recall)
# ---------------------------------------------------------------------------

if _HAS_LANGCHAIN_TOOLS:

    class _MemoryInput(BaseModel):
        """Input schema for AgentBay memory tool."""

        action: str = Field(
            description='Action to perform: "store" or "recall".'
        )
        content: str = Field(
            description=(
                'For "store": the knowledge to save. '
                'For "recall": the search query.'
            )
        )
        title: str = Field(
            default="",
            description='Optional title for the memory (only used with "store").',
        )

    class AgentBayMemoryTool(BaseTool):
        """LangChain tool for storing and recalling agent memories via AgentBay.

        This tool lets LLM agents explicitly persist knowledge across sessions
        using AgentBay's Knowledge Brain -- semantic search, confidence decay,
        and cross-agent sharing included.

        Args:
            api_key: Your AgentBay API key.
            project_id: The Knowledge Brain project ID.
            base_url: API base URL (optional).

        Example::

            from agentbay.integrations.langchain import AgentBayMemoryTool

            tool = AgentBayMemoryTool(
                api_key="ab_live_...",
                project_id="your-project",
            )

            # Add to your agent's tool list
            agent = initialize_agent(tools=[tool], llm=llm, ...)
        """

        name: str = "agentbay_memory"
        description: str = (
            "Store or recall persistent memories. "
            'Use action="store" with content to save knowledge. '
            'Use action="recall" with content as the search query to retrieve memories.'
        )
        args_schema: Type[BaseModel] = _MemoryInput

        # Internal -- not part of the tool schema
        _client: AgentBay

        def __init__(
            self,
            api_key: Optional[str] = None,
            project_id: Optional[str] = None,
            base_url: str = "https://www.aiagentsbay.com",
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            object.__setattr__(
                self,
                "_client",
                AgentBay(
                    api_key=api_key,
                    base_url=base_url,
                    project_id=project_id,
                ),
            )

        def _run(
            self,
            action: str,
            content: str,
            title: str = "",
            run_manager: Optional[CallbackManagerForToolRun] = None,
        ) -> str:
            """Execute the memory tool."""
            if action == "store":
                result = self._client.store(
                    content=content,
                    title=title or None,
                )
                entry_id = result.get("id", result.get("entryId", "unknown"))
                return f"Stored memory (id: {entry_id})"

            elif action == "recall":
                results = self._client.recall(query=content)
                if not results:
                    return "No matching memories found."
                formatted = []
                for i, entry in enumerate(results, 1):
                    title_str = entry.get("title", "Untitled")
                    body = entry.get("content", entry.get("text", ""))
                    score = entry.get("confidence", entry.get("score", "?"))
                    formatted.append(f"{i}. [{title_str}] (confidence: {score})\n   {body}")
                return "\n\n".join(formatted)

            else:
                return f'Unknown action "{action}". Use "store" or "recall".'

else:

    class AgentBayMemoryTool:  # type: ignore[no-redef]
        """Placeholder when langchain-core is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "langchain-core is required for the LangChain integration. "
                "Install it with: pip install agentbay[langchain]"
            )
