# Changelog

## 1.2.0 (2026-04-03)
- **FastEmbed bundled by default** — vector search works out of the box, no separate install needed
- Better error messages with fix URLs for common issues (401, 403, 404, 429, 500)
- `brain.login()` works in headless environments (servers, Docker, SSH) — prints URL instead of crashing
- First-use welcome message with quickstart hints (only shown once)
- Auto-update check — notifies when a newer version is available (once per day, non-blocking)
- `AGENTBAY_API_KEY` environment variable auto-detected in constructor
- Added CHANGELOG.md

## 1.1.0 (2026-04-02)
- Added offline Projects with sync protocol
- Added offline Teams with shared SQLite
- Added `brain.offline_project()` and `brain.offline_team()`

## 1.0.0 (2026-04-02)
- **Production-grade local mode**: SQLite + FTS5 + FastEmbed ONNX vectors + Ollama extraction
- **20 LLM providers** in brain.chat(): auto-detects from API keys
- `brain.chat()` auto-memory wrapper (recall before, store after)
- `brain.login()` browser-based auth with auto-migration
- `AgentBay.from_saved()` loads saved credentials
- `user_id` scoping for chatbot builders
- Local mode: Ollama → API → heuristic extraction cascade

## 0.5.0 (2026-04-01)
- Added `brain.login()` for browser-based authentication
- API key saved to `~/.agentbay/config.json` for future sessions
- `AgentBay.from_saved()` loads saved config

## 0.4.0 (2026-04-01)
- Local mode (SQLite, zero dependencies)
- 15 framework integrations (CrewAI, LangChain, AutoGen, LlamaIndex, LangGraph, etc.)
- Docker support with self-hosted server

## 0.3.0 (2026-03-31)
- Teams wrapper: `brain.team("id").chat()`
- Projects wrapper: `brain.project("id").chat()`
- `create_team()`, `create_project()` convenience methods

## 0.2.0 (2026-03-31)
- `brain.chat()` auto-memory wrapper
- Mem0-compatible `add()` and `search()` methods
- Auto-detection of learnable content in responses

## 0.1.0 (2026-03-30)
- Initial release
- `store()`, `recall()`, `verify()`, `forget()`, `health()`
- `setup_brain()` for Knowledge Brain creation
