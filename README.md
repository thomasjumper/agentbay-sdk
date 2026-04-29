# AgentBay SDKs

The public home for AgentBay's client SDKs. Install once, give your AI agents persistent memory.

```bash
pip install agentbay         # Python
npm install agentbay         # TypeScript / Node
npm install -g aiagentsbay-mcp  # MCP server (Claude Code, Cursor, Codex, OpenClaw)
```

## What lives here

| Directory | Package | Status |
|---|---|---|
| [`python/`](./python) | [`agentbay`](https://pypi.org/project/agentbay/) on PyPI | ✅ canonical source |
| `typescript/` | [`agentbay`](https://www.npmjs.com/package/agentbay) on npm | _moving here_ |
| `mcp-server/` | [`aiagentsbay-mcp`](https://www.npmjs.com/package/aiagentsbay-mcp) on npm | _moving here_ |

Until the TypeScript SDK and MCP server land in this repo, their source still ships from the AgentBay platform repo. They are still publicly installable from npm with the names above.

## What AgentBay does

```python
from agentbay import AgentBay

brain = AgentBay()  # local mode — no signup, runs on your machine
brain.store("JWT auth, 24h refresh tokens", title="Auth pattern", type="PATTERN")
brain.recall("authentication")
# → returns the entry with relevance score
```

When you want cloud sync, teams, or projects:

```python
brain = brain.login()  # opens browser → sign up → paste API key → done
# Local memories stay on your machine; new memories go to your cloud project.
```

Free cloud tier: 1,000 memories, 10,000 API calls/month, 5 projects. No credit card.

## Documentation

- Quickstart: <https://www.aiagentsbay.com/getting-started>
- Per-client guides: <https://www.aiagentsbay.com/docs>
- API reference: <https://www.aiagentsbay.com/docs/api>
- Pricing: <https://www.aiagentsbay.com/pricing>

## Reporting issues

Open an issue on this repo. For private support, contact <support@aiagentsbay.com>.

## License

MIT — see [LICENSE](./LICENSE).
