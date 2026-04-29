# AgentBay Python SDK

Persistent memory for AI agents. 3 lines to give your agent a brain.

## Install

```bash
pip install agentbay
```

## Quick Start -- Auto-Memory (Recommended)

The `chat()` method wraps your LLM call with automatic memory. No manual store/recall needed.

```python
from agentbay import AgentBay

brain = AgentBay("ab_live_your_key", project_id="your-project-id")

# Memory happens automatically -- no manual store/recall needed
response = brain.chat([
    {"role": "user", "content": "fix the auth session expiry bug"}
])

# brain.chat() automatically:
# 1. Recalled relevant memories about auth and sessions
# 2. Injected them into the LLM context
# 3. Got the response from Claude
# 4. Extracted learnings and stored them for next time
```

### Using OpenAI

```python
response = brain.chat(
    [{"role": "user", "content": "refactor the payment module"}],
    model="gpt-4o",
    provider="openai",
)
```

### Passing extra LLM parameters

```python
response = brain.chat(
    [{"role": "user", "content": "optimize the database queries"}],
    max_tokens=8192,
    temperature=0.7,
)
```

### Disabling auto-memory

```python
# Recall only (don't store new learnings)
response = brain.chat(messages, auto_store=False)

# Store only (don't inject recalled memories)
response = brain.chat(messages, auto_recall=False)

# No memory at all (just use as a plain LLM wrapper)
response = brain.chat(messages, auto_recall=False, auto_store=False)
```

## Mem0-Compatible API

If you're migrating from Mem0, AgentBay supports the same `add()` / `search()` interface:

```python
brain = AgentBay("ab_live_your_key", project_id="your-project-id")

# Store with automatic type detection
brain.add("The auth bug was caused by expired JWT tokens not being refreshed")
brain.add("We decided to use PostgreSQL instead of MongoDB for ACID compliance")

# Search
results = brain.search("authentication issues")
for r in results:
    print(r["title"], r["confidence"])
```

## Manual Memory Control

For full control, use `store()` and `recall()` directly:

```python
from agentbay import AgentBay

brain = AgentBay("ab_live_your_key", project_id="your-project-id")
brain.store("Next.js 16 + Prisma + PostgreSQL", title="Project stack")
results = brain.recall("What stack does this project use?")
```

Or create a new brain on the fly:

```python
from agentbay import AgentBay

brain = AgentBay("ab_live_your_key")
brain.setup_brain("My Agent's Memory")
brain.store("Always use UTC timestamps", title="Convention", type="PREFERENCE")
```

## Core API

| Method | What it does |
|--------|-------------|
| `brain.chat(messages, model, provider, ...)` | LLM call with automatic memory |
| `brain.add(data)` | Store with auto-detection (Mem0-compatible) |
| `brain.search(query)` | Search memories (Mem0-compatible alias) |
| `brain.store(content, title, type, tier, tags)` | Save a memory (full control) |
| `brain.recall(query, limit, tier, tags)` | Search memories (semantic + keyword) |
| `brain.forget(knowledge_id)` | Archive a memory |
| `brain.verify(knowledge_id)` | Confirm a memory is still accurate |
| `brain.health()` | Get memory stats |
| `brain.setup_brain(name, description)` | Create a new Knowledge Brain |

## Memory Types

- `PATTERN` -- Learned behaviors and recurring themes
- `FACT` -- Verified information
- `PREFERENCE` -- User/agent preferences
- `PROCEDURE` -- Step-by-step processes
- `CONTEXT` -- Situational context
- `PITFALL` -- Bugs, errors, and fixes to avoid
- `DECISION` -- Architecture and design decisions

## With CrewAI

```bash
pip install agentbay[crewai]
```

```python
from crewai import Agent
from agentbay.integrations.crewai import AgentBayCrewAIMemory

memory = AgentBayCrewAIMemory(
    api_key="ab_live_your_key",
    project_id="your-project-id",
)

agent = Agent(
    role="Researcher",
    goal="Find and remember information",
    memory=memory,
)
```

## With LangChain

```bash
pip install agentbay[langchain]
```

```python
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from agentbay.integrations.langchain import AgentBayMemoryTool

tool = AgentBayMemoryTool(
    api_key="ab_live_your_key",
    project_id="your-project-id",
)

llm = ChatOpenAI()
agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
)
agent.run("Remember that deploys happen every Tuesday at 2pm UTC")
```

## Error Handling

```python
from agentbay import AgentBayError, AuthenticationError, RateLimitError

try:
    results = brain.recall("query")
except AuthenticationError:
    print("Bad API key")
except RateLimitError:
    print("Slow down")
except AgentBayError as e:
    print(f"Error {e.status_code}: {e}")
```

## Environment Variables

For `chat()`, set your LLM provider API key:

```bash
# For Anthropic (default provider)
export ANTHROPIC_API_KEY=sk-ant-...

# For OpenAI
export OPENAI_API_KEY=sk-...
```

Or pass it directly:

```python
response = brain.chat(messages, api_key="sk-ant-...")
```

## Links

- [AgentBay](https://www.aiagentsbay.com) -- AI agent memory platform
- [Quickstart](https://www.aiagentsbay.com/docs/quickstart)
- [Python SDK Reference](https://www.aiagentsbay.com/docs/python-sdk)
- [Benchmarks](https://www.aiagentsbay.com/benchmarks)
- [AgentBay vs Mem0](https://www.aiagentsbay.com/docs/vs-mem0)
- [MCP Server](https://www.npmjs.com/package/aiagentsbay-mcp)
- [Changelog](./CHANGELOG.md)

## Community

- Questions? [Join our Discord](https://discord.gg/DISCORD_INVITE_PLACEHOLDER)
- Bugs? [GitHub Issues](https://github.com/thomasjumper/agentbay-python/issues)
