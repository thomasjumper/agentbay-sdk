"""AgentBay integrations for popular AI frameworks.

Available integrations:

CrewAI (``pip install agentbay[crewai]``)::

    from agentbay.integrations.crewai import AgentBayMemory
    memory = AgentBayMemory(api_key="ab_live_...", project_id="...")
    agent = Agent(role="...", memory=memory)

LangChain - BaseMemory (``pip install agentbay[langchain]``)::

    from agentbay.integrations.langchain import AgentBayMemory
    memory = AgentBayMemory(api_key="ab_live_...", project_id="...")
    chain = ConversationChain(llm=llm, memory=memory)

LangChain - Tool (``pip install agentbay[langchain]``)::

    from agentbay.integrations.langchain import AgentBayMemoryTool
    tool = AgentBayMemoryTool(api_key="ab_live_...", project_id="...")
    agent = initialize_agent(tools=[tool], llm=llm, ...)

AutoGen / AG2 (``pip install agentbay[autogen]``)::

    from agentbay.integrations.autogen import AgentBayMemory
    memory = AgentBayMemory(api_key="ab_live_...", project_id="...")
    memory.attach(assistant_agent)

LlamaIndex (``pip install agentbay[llamaindex]``)::

    from agentbay.integrations.llamaindex import AgentBayMemory
    memory = AgentBayMemory(api_key="ab_live_...", project_id="...")
    context = memory.get("What do we know about auth?")

LangGraph (``pip install agentbay[langgraph]``)::

    from agentbay.integrations.langgraph import AgentBayCheckpointer
    checkpointer = AgentBayCheckpointer(api_key="ab_live_...", project_id="...")
    graph = StateGraph(...).compile(checkpointer=checkpointer)

Vercel AI SDK (no extra deps)::

    from agentbay.integrations.vercel_ai import AgentBayProvider
    provider = AgentBayProvider(api_key="ab_live_...", project_id="...")
    context = provider.get_context(messages)

ElevenLabs (``pip install agentbay[elevenlabs]``)::

    from agentbay.integrations.elevenlabs import AgentBayVoiceMemory
    memory = AgentBayVoiceMemory(api_key="ab_live_...", project_id="...")
    context = memory.on_message("What did we discuss?")

Pipecat (``pip install agentbay[pipecat]``)::

    from agentbay.integrations.pipecat import AgentBayProcessor
    processor = AgentBayProcessor(api_key="ab_live_...", project_id="...")
    pipeline = Pipeline([transport, processor, llm, tts])

Mastra (no extra deps)::

    from agentbay.integrations.mastra import AgentBayMemoryProvider
    memory = AgentBayMemoryProvider(api_key="ab_live_...", project_id="...")
    results = memory.get_memory("API rate limits")

Agno (no extra deps)::

    from agentbay.integrations.agno import AgentBayMemory
    memory = AgentBayMemory(api_key="ab_live_...", project_id="...")
    memories = memory.recall("deployment config")

Camel AI (``pip install agentbay[camel]``)::

    from agentbay.integrations.camel_ai import AgentBayMemory
    memory = AgentBayMemory(api_key="ab_live_...", project_id="...")
    results = memory.retrieve("project architecture")

OpenAI Codex (no extra deps)::

    from agentbay.integrations.codex import AgentBayCodexMemory
    memory = AgentBayCodexMemory(api_key="ab_live_...")
    enriched = memory.before_completion(messages)
    memory.after_completion(messages, response)

AgentOps Observability (``pip install agentbay agentops``)::

    from agentbay.integrations.agentops import track_memory_ops
    brain = AgentBay("ab_live_...")
    track_memory_ops(brain, agentops_api_key="ao_...")

MCP Installation Helper::

    python -m agentbay.integrations.mcp_install
    python -m agentbay.integrations.mcp_install cursor

Install all integrations at once::

    pip install agentbay[all]
"""
