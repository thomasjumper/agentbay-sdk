"""AgentBay client - 3 lines to give your agent a brain."""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from typing_extensions import TypeAlias

import time

from agentbay.auth import login_via_browser
from agentbay.config import (
    LOCAL_UPGRADE_PROMPT,
    load_config,
    load_saved_key,
    maybe_prompt_for_install_ping,
    save_api_key,
)

MemoryEntry: TypeAlias = Dict[str, Any]

__version__ = "1.5.0"


def _check_for_updates_background() -> None:
    """Check PyPI for newer version in a background thread. Non-blocking, once per day."""
    def _check() -> None:
        try:
            config_dir = Path.home() / ".agentbay"
            check_file = config_dir / ".last_update_check"

            if check_file.exists():
                last_check = float(check_file.read_text().strip())
                if time.time() - last_check < 86400:
                    return

            import urllib.request
            resp = urllib.request.urlopen("https://pypi.org/pypi/agentbay/json", timeout=3)
            data = json.loads(resp.read())
            latest = data["info"]["version"]

            if latest != __version__:
                print(f"\U0001f9e0 AgentBay {latest} available (you have {__version__}). Run: pip install --upgrade agentbay")

            config_dir.mkdir(exist_ok=True)
            check_file.write_text(str(time.time()))
        except Exception:
            pass

    threading.Thread(target=_check, daemon=True).start()

# Patterns that suggest something worth storing
_LEARNING_PATTERNS = re.compile(
    r"(?:the (?:issue|problem|bug|error|fix|solution|cause|reason) (?:was|is))"
    r"|(?:(?:we|i) (?:decided|chose|picked|went with|settled on))"
    r"|(?:the pattern is)"
    r"|(?:(?:always|never|make sure to|remember to|don't forget to))"
    r"|(?:(?:turns out|it works because|the trick is|the key (?:insight|thing)))",
    re.IGNORECASE,
)

# Patterns for auto-detecting memory type
_PITFALL_PATTERNS = re.compile(
    r"\b(?:bug|error|fix|crash|fail|broke|issue|problem|exception|traceback|stack\s*trace)\b",
    re.IGNORECASE,
)
_DECISION_PATTERNS = re.compile(
    r"\b(?:decided|chose|picked|went with|settled on|decision|choose|choosing)\b",
    re.IGNORECASE,
)
_PROCEDURE_PATTERNS = re.compile(
    r"\b(?:step\s*\d|first.*then|how to|procedure|process|workflow|instructions)\b",
    re.IGNORECASE,
)


def _detect_type(text: str) -> str:
    """Auto-detect memory type from content."""
    if _PITFALL_PATTERNS.search(text):
        return "PITFALL"
    if _DECISION_PATTERNS.search(text):
        return "DECISION"
    if _PROCEDURE_PATTERNS.search(text):
        return "PROCEDURE"
    return "PATTERN"


# ---------------------------------------------------------------------------
# Supported LLM providers
# ---------------------------------------------------------------------------

SUPPORTED_PROVIDERS = {
    # Tier 1 — Major cloud providers
    "anthropic": {"name": "anthropic", "module": "anthropic", "default_model": "claude-sonnet-4-20250514"},
    "openai": {"name": "openai", "module": "openai", "default_model": "gpt-4o"},
    "google": {"name": "google", "module": "google.generativeai", "default_model": "gemini-2.5-flash"},
    "xai": {"name": "xai", "module": "openai", "default_model": "grok-3-mini", "base_url": "https://api.x.ai/v1"},

    # Tier 2 — AI API providers (OpenAI-compatible)
    "mistral": {"name": "mistral", "module": "openai", "default_model": "mistral-large-latest", "base_url": "https://api.mistral.ai/v1"},
    "cohere": {"name": "cohere", "module": "cohere", "default_model": "command-r-plus"},
    "deepseek": {"name": "deepseek", "module": "openai", "default_model": "deepseek-chat", "base_url": "https://api.deepseek.com/v1"},
    "together": {"name": "together", "module": "openai", "default_model": "meta-llama/Llama-3.3-70B-Instruct", "base_url": "https://api.together.xyz/v1"},
    "fireworks": {"name": "fireworks", "module": "openai", "default_model": "accounts/fireworks/models/llama-v3p3-70b-instruct", "base_url": "https://api.fireworks.ai/inference/v1"},
    "groq": {"name": "groq", "module": "openai", "default_model": "llama-3.3-70b-versatile", "base_url": "https://api.groq.com/openai/v1"},
    "perplexity": {"name": "perplexity", "module": "openai", "default_model": "sonar-pro", "base_url": "https://api.perplexity.ai"},

    # Tier 3 — Enterprise/cloud
    "azure": {"name": "azure", "module": "openai", "default_model": "gpt-4o"},  # uses AZURE_OPENAI_ENDPOINT
    "bedrock": {"name": "bedrock", "module": "anthropic", "default_model": "claude-sonnet-4-20250514"},  # AWS Bedrock

    # Tier 4 — Local
    "ollama": {"name": "ollama", "module": "openai", "default_model": "llama3.3", "base_url": "http://localhost:11434/v1"},
    "lmstudio": {"name": "lmstudio", "module": "openai", "default_model": "local-model", "base_url": "http://localhost:1234/v1"},
    "llamacpp": {"name": "llamacpp", "module": "openai", "default_model": "local", "base_url": "http://localhost:8080/v1"},
}

# Env var name for each provider's API key
_PROVIDER_KEY_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "xai": "XAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "groq": "GROQ_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "bedrock": "AWS_ACCESS_KEY_ID",
}


def _extract_title(text: str, max_len: int = 100) -> str:
    """Extract a title from the first sentence or first N chars."""
    # Try to get the first sentence
    match = re.match(r"^(.+?[.!?])\s", text)
    if match and len(match.group(1)) <= max_len:
        return match.group(1)
    # Fall back to first max_len chars
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


class AgentBayError(Exception):
    """Base exception for AgentBay SDK errors."""

    def __init__(self, message: str, status_code: int | None = None, response: dict | None = None, help_url: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response
        self.help_url = help_url


class AuthenticationError(AgentBayError):
    """Raised when the API key is invalid or missing."""

    def __init__(self, message: str = "Invalid or expired API key", **kwargs: Any):
        super().__init__(message, help_url="https://www.aiagentsbay.com/dashboard/api-keys", **kwargs)


class NotFoundError(AgentBayError):
    """Raised when a resource is not found."""

    def __init__(self, message: str = "Resource not found", **kwargs: Any):
        super().__init__(message, help_url="https://www.aiagentsbay.com/dashboard", **kwargs)


class RateLimitError(AgentBayError):
    """Raised when the API rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", **kwargs: Any):
        super().__init__(message, help_url="https://www.aiagentsbay.com/pricing", **kwargs)


def _mark_onboarded() -> None:
    """Mark onboarding as complete so it doesn't run again."""
    try:
        config_dir = Path.home() / ".agentbay"
        config_dir.mkdir(exist_ok=True)
        (config_dir / ".onboarded").touch()
    except Exception:
        pass


class AgentBay:
    """Persistent memory for AI agents.

    Usage::

        from agentbay import AgentBay

        brain = AgentBay("ab_live_your_key", project_id="your-project-id")

        # Auto-memory wrapper -- the simplest way to use AgentBay
        response = brain.chat([
            {"role": "user", "content": "fix the auth session expiry bug"}
        ])

        # Or manual control
        brain.store("Next.js 16 + Prisma + PostgreSQL", title="Project stack")
        results = brain.recall("What stack does this project use?")

    Args:
        api_key: Your AgentBay API key (starts with ``ab_live_`` or ``ab_test_``).
        base_url: API base URL. Defaults to ``https://www.aiagentsbay.com``.
        project_id: Default project ID to use for all operations.
            Can be overridden per-call.
        timeout: Request timeout in seconds. Defaults to 30.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://www.aiagentsbay.com",
        project_id: str | None = None,
        timeout: int = 30,
    ) -> None:
        # Check environment variable
        if not api_key:
            api_key = os.environ.get("AGENTBAY_API_KEY")

        # Check saved config
        if not api_key:
            api_key = load_saved_key()

        if not api_key:
            # Local mode — user chose local or non-interactive environment
            from .local import LocalMemory

            self._local = LocalMemory(quiet=True)
            self._is_local = True
            self.api_key = None
            self.project_id = None
            self.base_url = base_url
            self.timeout = timeout
            self._session = None
            if not os.environ.get("AGENTBAY_QUIET"):
                print(LOCAL_UPGRADE_PROMPT)
                maybe_prompt_for_install_ping(version=__version__, base_url=base_url)
            return

        self._local = None
        self._is_local = False
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.project_id = project_id
        self.timeout = timeout

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": f"agentbay-python/{__version__}",
            }
        )
        self._show_welcome("cloud")
        if not os.environ.get("AGENTBAY_QUIET"):
            _check_for_updates_background()

    @staticmethod
    def _show_welcome(mode: str) -> None:
        """Show a one-time welcome message on first use."""
        if os.environ.get("AGENTBAY_QUIET"):
            return
        config_dir = Path.home() / ".agentbay"
        welcome_flag = config_dir / ".welcomed"
        if welcome_flag.exists():
            return
        try:
            config_dir.mkdir(exist_ok=True)
            welcome_flag.touch()
            if mode == "local":
                print(f"\U0001f9e0 AgentBay ready (local mode — unlimited, offline)")
                print(f"   Memories stored at: ~/.agentbay/")
                print(f"   Ready for cloud? Run: brain.login()")
            else:
                print(f"\U0001f9e0 AgentBay ready (cloud mode)")
                print(f"   Memories sync across devices, teams, and agents")
            print(f"   Try: brain.chat([{{'role': 'user', 'content': 'hello'}}])")
            print(f"   Docs: https://www.aiagentsbay.com/getting-started")
        except Exception:
            pass

    @staticmethod
    def _load_saved_key() -> str | None:
        """Load API key from saved config file."""
        return load_saved_key()

    @staticmethod
    def _save_key(api_key: str, base_url: str = "https://www.aiagentsbay.com") -> None:
        """Save API key to config for future sessions."""
        save_api_key(api_key, base_url=base_url)

    @staticmethod
    def _interactive_onboarding(base_url: str) -> str | None:
        """Interactive first-run onboarding. Only runs in terminals.

        Flow:
        1. Ask for API key (paste or skip)
        2. If skipped: Login / Create Account / Use Local
        3. Login/Create opens browser, then asks for key paste
        4. Key is saved for future sessions

        Returns the API key or None for local mode.
        """
        # Don't prompt if not a terminal (CI, imports, scripts)
        if not os.isatty(0) or os.environ.get("AGENTBAY_QUIET"):
            return None

        # Check if onboarding was already completed
        onboard_flag = Path.home() / ".agentbay" / ".onboarded"
        if onboard_flag.exists():
            return None

        print("\n\U0001f9e0 AgentBay Setup")
        print("=" * 40)
        print()

        # Step 1: Ask for API key
        try:
            key = input("Enter your API key (or press Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if key and key.startswith("ab_live_"):
            AgentBay._save_key(key)
            _mark_onboarded()
            return key

        if key and not key.startswith("ab_live_"):
            print("  Invalid key format. Keys start with ab_live_")
            print()

        # Step 2: Options
        print("\nOptions:")
        print("  [1] Log in to existing account")
        print("  [2] Create a new account")
        print("  [3] Use local mode (offline, no account needed)")
        print()

        try:
            choice = input("Choose [1/2/3]: ").strip()
        except (EOFError, KeyboardInterrupt):
            _mark_onboarded()
            return None

        if choice in ("1", "2"):
            # Open browser
            path = "/login" if choice == "1" else "/register"
            url = f"{base_url}{path}?utm_source=sdk&utm_medium=onboarding"

            browser_opened = False
            try:
                import webbrowser
                browser_opened = webbrowser.open(url)
            except Exception:
                pass

            if browser_opened:
                print(f"\n  Browser opened: {url}")
            else:
                print(f"\n  Open this URL in your browser:")
                print(f"  {url}")

            print()
            print("  After signing in, go to Dashboard → API Keys")
            print("  Create a key and paste it below.")
            print()

            try:
                key = input("Paste your API key: ").strip()
            except (EOFError, KeyboardInterrupt):
                _mark_onboarded()
                return None

            if key and key.startswith("ab_live_"):
                AgentBay._save_key(key)
                _mark_onboarded()
                print(f"\n  \u2713 Connected to AgentBay cloud!")
                return key
            else:
                print("  No valid key entered. Starting in local mode.")
                _mark_onboarded()
                return None

        # Choice 3 or anything else → local mode
        _mark_onboarded()
        return None


    # ------------------------------------------------------------------
    # chat() -- The auto-memory LLM wrapper
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        provider: str = "auto",
        project_id: str | None = None,
        user_id: str | None = None,
        auto_recall: bool = True,
        auto_store: bool = True,
        recall_limit: int = 3,
        **kwargs: Any,
    ) -> Any:
        """Wrap an LLM call with automatic memory recall and storage.

        This is the primary way to use AgentBay. Wrap your LLM call and
        memory happens automatically -- no manual store/recall needed.

        Supports 20+ LLM providers. Most are OpenAI-compatible and only
        need the ``openai`` package plus the right API key.

        Usage::

            from agentbay import AgentBay

            brain = AgentBay("ab_live_your_key", project_id="your-project-id")

            # Auto-detect provider from available API keys (default)
            response = brain.chat([
                {"role": "user", "content": "fix the auth session expiry bug"}
            ])

            # Anthropic
            response = brain.chat(
                [{"role": "user", "content": "fix the auth bug"}],
                provider="anthropic",
            )

            # OpenAI
            response = brain.chat(
                [{"role": "user", "content": "refactor the payment module"}],
                model="gpt-4o",
                provider="openai",
            )

            # xAI (Grok) -- uses OpenAI-compatible API
            response = brain.chat(
                [{"role": "user", "content": "explain quantum computing"}],
                provider="xai",
            )

            # Groq, DeepSeek, Together, Fireworks, Perplexity, Mistral...
            response = brain.chat(messages, provider="groq")
            response = brain.chat(messages, provider="deepseek")
            response = brain.chat(messages, provider="together")

            # Local LLMs (Ollama, LM Studio, llama.cpp)
            response = brain.chat(messages, provider="ollama")
            response = brain.chat(messages, provider="lmstudio")

            # Google Gemini
            response = brain.chat(messages, provider="google")

        Supported providers:

        - **Tier 1 (major clouds)**: ``anthropic``, ``openai``, ``google``, ``xai``
        - **Tier 2 (API providers)**: ``mistral``, ``cohere``, ``deepseek``,
          ``together``, ``fireworks``, ``groq``, ``perplexity``
        - **Tier 3 (enterprise)**: ``azure``, ``bedrock``
        - **Tier 4 (local)**: ``ollama``, ``lmstudio``, ``llamacpp``

        What happens under the hood:

        1. **Auto-recall**: The last user message is used to search your
           Knowledge Brain. Relevant memories are injected as context.
        2. **LLM call**: The enriched messages are sent to your chosen provider.
        3. **Auto-store**: The assistant response is scanned for learnings
           (bug fixes, decisions, patterns) and stored automatically.

        Args:
            messages: Chat messages in OpenAI format
                (``[{"role": "user", "content": "..."}]``).
            model: Model name to use. If None, uses the provider's default model.
            provider: LLM provider name or ``"auto"`` to detect from env vars.
                Defaults to ``"auto"``.
            project_id: Project for memory ops (overrides default).
            user_id: Optional user ID for memory scoping.
            auto_recall: Whether to recall relevant memories. Defaults to True.
            auto_store: Whether to store learnings from the response. Defaults to True.
            recall_limit: Maximum memories to recall (1-10). Defaults to 3.
            **kwargs: Extra keyword arguments passed to the LLM client
                (e.g. ``max_tokens``, ``temperature``, ``api_key``).

        Returns:
            The raw LLM response object (Anthropic ``Message``, OpenAI
            ``ChatCompletion``, Google ``GenerateContentResponse``, or
            Cohere ``ChatResponse``). Memory operations are a side effect.

        Raises:
            AgentBayError: If memory operations fail (LLM call still proceeds).
            ImportError: If the provider library is not installed.
        """
        # --- 0. Resolve provider & model ---
        if provider == "auto":
            provider = self._detect_provider()

        config = SUPPORTED_PROVIDERS.get(provider)
        if not config:
            raise AgentBayError(
                f"Unsupported provider: '{provider}'. "
                f"Supported: {list(SUPPORTED_PROVIDERS.keys())}"
            )
        model = model or config["default_model"]

        pid = None if self._is_local else self._resolve_project(project_id)

        # --- 1. Auto-recall ---
        enriched_messages = list(messages)  # shallow copy
        memory_context = ""
        if auto_recall:
            last_user_msg = self._extract_last_user_message(messages)
            if last_user_msg:
                try:
                    if self._is_local:
                        memories = self._local.recall(last_user_msg, limit=recall_limit, user_id=user_id)
                    else:
                        memories = self.recall(last_user_msg, project_id=pid, limit=recall_limit, user_id=user_id)
                    if memories:
                        memory_context = self._format_memory_context(memories)
                        enriched_messages = self._inject_memory_context(enriched_messages, memory_context, provider)
                except Exception:
                    # Memory recall failed -- proceed without it
                    pass

        # --- 2. Call the LLM ---
        response = self._call_llm(enriched_messages, model, provider, **kwargs)

        # --- 3. Auto-store (fire-and-forget in background thread) ---
        if auto_store:
            last_user_msg = self._extract_last_user_message(messages)
            assistant_text = self._extract_response_text(response, provider)
            if last_user_msg and assistant_text:
                thread = threading.Thread(
                    target=self._auto_store_learnings,
                    args=(last_user_msg, assistant_text, pid),
                    daemon=True,
                )
                thread.start()

        return response

    # ------------------------------------------------------------------
    # add() -- Mem0-compatible simple store
    # ------------------------------------------------------------------

    def add(
        self,
        data: str,
        user_id: str | None = None,
        agent_id: str | None = None,
        project_id: str | None = None,
        metadata: dict | None = None,
    ) -> MemoryEntry:
        """Store a memory with automatic type detection and title extraction.

        This is the Mem0-compatible API. Pass a string and AgentBay
        figures out the rest.

        Usage::

            brain.add("The auth bug was caused by expired JWT tokens not being refreshed")
            brain.add("We decided to use PostgreSQL instead of MongoDB for ACID compliance")
            brain.add("Always run migrations before deploying to staging")

        Args:
            data: The knowledge content to store.
            user_id: Optional user ID for scoping.
            agent_id: Optional agent ID for scoping.
            project_id: Project to store in (overrides default).
            metadata: Optional extra metadata dict to include.

        Returns:
            Dict with the created entry, including its ``id``.
        """
        if self._is_local:
            return self._local.add(data, user_id=user_id)

        pid = self._resolve_project(project_id)
        entry_type = _detect_type(data)
        title = _extract_title(data)

        body: Dict[str, Any] = {
            "content": data,
            "type": entry_type,
            "tier": "semantic",
            "title": title,
            "source": "sdk-auto",
        }
        if metadata:
            body["metadata"] = metadata

        return self._post(f"/api/v1/projects/{pid}/memory", body)

    # ------------------------------------------------------------------
    # search() -- Mem0-compatible recall alias
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        user_id: str | None = None,
        project_id: str | None = None,
        limit: int = 5,
    ) -> List[MemoryEntry]:
        """Search memories by semantic similarity.

        This is an alias for :meth:`recall` for Mem0 compatibility.

        Usage::

            results = brain.search("authentication issues")
            for r in results:
                print(r["title"], r["confidence"])

        Args:
            query: Natural-language search query.
            user_id: Optional user ID for scoping.
            project_id: Project to search in (overrides default).
            limit: Maximum number of results (1-50). Defaults to 5.

        Returns:
            List of matching entries with confidence scores.
        """
        if self._is_local:
            return self._local.search(query, user_id=user_id, limit=limit)
        return self.recall(query, project_id=project_id, limit=limit, user_id=user_id)

    # ------------------------------------------------------------------
    # Core memory operations
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        title: str | None = None,
        project_id: str | None = None,
        type: str = "PATTERN",
        tier: str = "semantic",
        tags: list[str] | None = None,
        user_id: str | None = None,
    ) -> MemoryEntry:
        """Store a memory in your Knowledge Brain.

        Args:
            content: The knowledge content to store.
            title: Optional short title for the entry.
            project_id: Project to store in (overrides default).
            type: Entry type -- PATTERN, FACT, PREFERENCE, PROCEDURE, CONTEXT.
            tier: Storage tier -- semantic, episodic, procedural.
            tags: Optional list of tags for categorization.
            user_id: Optional user ID for scoping. Stored as a ``user:<id>`` tag.

        Returns:
            Dict with the created entry, including its ``id``.

        Raises:
            AgentBayError: If the request fails.
        """
        if self._is_local:
            return self._local.store(
                content, title=title, type=type, tier=tier,
                tags=tags, user_id=user_id,
            )

        pid = self._resolve_project(project_id)
        body: Dict[str, Any] = {
            "content": content,
            "type": type,
            "tier": tier,
        }
        if title is not None:
            body["title"] = title

        # Merge user_id into tags as user:<id>
        effective_tags = list(tags) if tags else []
        if user_id is not None:
            effective_tags.append(f"user:{user_id}")
        if effective_tags:
            body["tags"] = effective_tags

        return self._post(f"/api/v1/projects/{pid}/memory", body)

    def recall(
        self,
        query: str,
        project_id: str | None = None,
        limit: int = 5,
        tier: str | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
    ) -> List[MemoryEntry]:
        """Search memories by semantic similarity.

        Args:
            query: Natural-language search query.
            project_id: Project to search in (overrides default).
            limit: Maximum number of results (1-50). Defaults to 5.
            tier: Filter by storage tier.
            tags: Filter by tags.
            user_id: Optional user ID for scoping. Filters to ``user:<id>`` tagged entries.

        Returns:
            List of matching entries with confidence scores.
        """
        if self._is_local:
            return self._local.recall(
                query, limit=limit, user_id=user_id,
                type=None, tags=tags,
            )

        pid = self._resolve_project(project_id)

        # Merge user_id into tags filter
        effective_tags = list(tags) if tags else []
        if user_id is not None:
            effective_tags.append(f"user:{user_id}")

        params: Dict[str, Any] = {"q": query, "limit": str(limit)}
        if tier is not None:
            params["tier"] = tier
        if effective_tags:
            params["tags"] = ",".join(effective_tags)
        resp = self._get(f"/api/v1/projects/{pid}/memory", params)
        if isinstance(resp, list):
            return resp
        return resp.get("results", resp.get("entries", []))

    def forget(
        self,
        knowledge_id: str,
        project_id: str | None = None,
    ) -> None:
        """Archive (soft-delete) a memory entry.

        Args:
            knowledge_id: The ID of the memory to archive.
            project_id: Project containing the entry (overrides default).
        """
        if self._is_local:
            self._local.forget(knowledge_id)
            return
        pid = self._resolve_project(project_id)
        self._delete(f"/api/v1/projects/{pid}/memory", {"knowledgeId": knowledge_id})

    def verify(
        self,
        knowledge_id: str,
        project_id: str | None = None,
    ) -> None:
        """Confirm a memory is still accurate, resetting its confidence decay.

        Args:
            knowledge_id: The ID of the memory to verify.
            project_id: Project containing the entry (overrides default).
        """
        pid = self._resolve_project(project_id)
        self._patch(f"/api/v1/projects/{pid}/memory", {"knowledgeId": knowledge_id, "action": "verify"})

    def health(
        self,
        project_id: str | None = None,
    ) -> Dict[str, Any]:
        """Get memory health statistics for a project.

        Returns entry counts, average confidence, stale entries, etc.

        Args:
            project_id: Project to check (overrides default).

        Returns:
            Dict with health metrics.
        """
        if self._is_local:
            return self._local.health()
        pid = self._resolve_project(project_id)
        return self._get(f"/api/v1/projects/{pid}/memory", {"action": "health"})

    # ------------------------------------------------------------------
    # Brain management
    # ------------------------------------------------------------------

    def setup_brain(
        self,
        name: str,
        description: str | None = None,
    ) -> Dict[str, Any]:
        """Create a new private Knowledge Brain for your agent.

        This provisions a project with vector search, confidence decay,
        and all memory features enabled.

        Args:
            name: Human-readable name for the brain.
            description: Optional description.

        Returns:
            Dict with brain/project details including ``projectId``.
        """
        body: Dict[str, Any] = {"name": name}
        if description is not None:
            body["description"] = description

        resp = self._post("/api/v1/brain/setup", body)

        # Auto-set as default project if none was configured.
        project_id = resp.get("projectId") or resp.get("project", {}).get("id")
        if project_id and self.project_id is None:
            self.project_id = project_id

        return resp

    # ------------------------------------------------------------------
    # chat() helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_last_user_message(messages: list[dict]) -> str | None:
        """Get the text of the last user message."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                # Handle Anthropic-style content blocks
                if isinstance(content, list):
                    texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
                    return " ".join(texts) if texts else None
                return None
        return None

    @staticmethod
    def _format_memory_context(memories: list[dict]) -> str:
        """Format recalled memories into a context string."""
        lines = ["[Memory context -- AgentBay recalled these relevant memories:]"]
        for mem in memories:
            title = mem.get("title", "Untitled")
            entry_type = mem.get("type", "PATTERN")
            confidence = mem.get("confidence", mem.get("score", 0))
            # Confidence may be 0-1 float or 0-100 int
            if isinstance(confidence, (int, float)) and confidence <= 1:
                confidence = int(confidence * 100)
            content = mem.get("content", "")
            lines.append(f"## {title} ({entry_type}, confidence: {confidence}%)")
            lines.append(content)
            lines.append("---")
        return "\n".join(lines)

    @staticmethod
    def _inject_memory_context(messages: list[dict], context: str, provider: str) -> list[dict]:
        """Inject memory context into the messages list."""
        enriched = list(messages)

        if provider == "anthropic":
            # For Anthropic, prepend as the first user message with context
            # or inject into existing system parameter (handled by caller via kwargs)
            # Simplest: insert a user message with context at the beginning,
            # followed by an assistant ack, before the real conversation.
            # But cleaner: prepend to the first user message.
            for i, msg in enumerate(enriched):
                if msg.get("role") == "user":
                    original_content = msg.get("content", "")
                    if isinstance(original_content, str):
                        enriched[i] = {
                            **msg,
                            "content": f"{context}\n\n{original_content}",
                        }
                    break
            else:
                # No user message found, prepend one
                enriched.insert(0, {"role": "user", "content": context})
        else:
            # For OpenAI-style, inject as system message
            if enriched and enriched[0].get("role") == "system":
                enriched[0] = {
                    **enriched[0],
                    "content": f"{context}\n\n{enriched[0].get('content', '')}",
                }
            else:
                enriched.insert(0, {"role": "system", "content": context})

        return enriched

    @staticmethod
    def _detect_provider() -> str:
        """Auto-detect the best available LLM provider from env vars or local servers."""
        # Check cloud providers in priority order
        for provider, env_var in _PROVIDER_KEY_MAP.items():
            if os.environ.get(env_var):
                return provider

        # Check for local LLMs
        try:
            import requests as _req
            if _req.get("http://localhost:11434/api/tags", timeout=1).status_code == 200:
                return "ollama"
        except Exception:
            pass
        try:
            import requests as _req
            if _req.get("http://localhost:1234/v1/models", timeout=1).status_code == 200:
                return "lmstudio"
        except Exception:
            pass
        try:
            import requests as _req
            if _req.get("http://localhost:8080/v1/models", timeout=1).status_code == 200:
                return "llamacpp"
        except Exception:
            pass

        raise AgentBayError(
            "No LLM provider detected. Set an API key env var "
            "(ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, etc.) "
            "or run a local LLM server (Ollama, LM Studio, llama.cpp)."
        )

    @staticmethod
    def _get_provider_key(config: dict, **kwargs: Any) -> str:
        """Get the API key for a provider from kwargs or env vars."""
        # Explicit api_key always wins
        api_key = kwargs.get("api_key")
        if api_key:
            return api_key

        provider_name = config.get("name", "")

        # Try provider-specific env var
        env_var = _PROVIDER_KEY_MAP.get(provider_name)
        if env_var:
            val = os.environ.get(env_var)
            if val:
                return val

        # For OpenAI-compatible providers, fall back to OPENAI_API_KEY
        if config.get("module") == "openai" and provider_name != "openai":
            val = os.environ.get("OPENAI_API_KEY")
            if val:
                return val

        # Local providers don't need a key
        base_url = config.get("base_url", "")
        if base_url.startswith("http://localhost"):
            return "not-needed"

        raise AgentBayError(
            f"No API key found for provider '{provider_name}'. "
            f"Set {_PROVIDER_KEY_MAP.get(provider_name, 'the appropriate env var')} "
            f"or pass api_key= to chat()."
        )

    @staticmethod
    def _call_llm(messages: list[dict], model: str, provider: str, **kwargs: Any) -> Any:
        """Call the LLM provider. Routes to the right backend based on provider config."""
        config = SUPPORTED_PROVIDERS[provider]  # already validated in chat()
        module = config["module"]

        # --- Anthropic (native) ---
        if module == "anthropic" and provider not in ("bedrock",):
            return AgentBay._call_anthropic(messages, model, config, **kwargs)

        # --- AWS Bedrock (Anthropic via Bedrock) ---
        if provider == "bedrock":
            return AgentBay._call_bedrock(messages, model, config, **kwargs)

        # --- Azure OpenAI ---
        if provider == "azure":
            return AgentBay._call_azure(messages, model, config, **kwargs)

        # --- Google Gemini ---
        if module == "google.generativeai":
            return AgentBay._call_google(messages, model, config, **kwargs)

        # --- Cohere ---
        if module == "cohere":
            return AgentBay._call_cohere(messages, model, config, **kwargs)

        # --- OpenAI-compatible (OpenAI, xAI, Mistral, DeepSeek, Together,
        #     Fireworks, Groq, Perplexity, Ollama, LM Studio, llama.cpp) ---
        return AgentBay._call_openai_compatible(messages, model, config, **kwargs)

    @staticmethod
    def _call_anthropic(messages: list[dict], model: str, config: dict, **kwargs: Any) -> Any:
        """Call the Anthropic API natively."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for provider='anthropic'. "
                "Install it with: pip install anthropic"
            )
        api_key = AgentBay._get_provider_key(config, **kwargs)
        kwargs.pop("api_key", None)

        # Anthropic uses 'system' as a top-level param, not in messages
        system_text = None
        anthropic_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
            else:
                anthropic_messages.append(msg)

        client = anthropic.Anthropic(api_key=api_key)

        call_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
        }
        if system_text:
            call_kwargs["system"] = system_text
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 4096

        call_kwargs.update(kwargs)
        return client.messages.create(**call_kwargs)

    @staticmethod
    def _call_bedrock(messages: list[dict], model: str, config: dict, **kwargs: Any) -> Any:
        """Call Anthropic models via AWS Bedrock."""
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "The 'anthropic' package is required for provider='bedrock'. "
                "Install it with: pip install anthropic"
            )

        kwargs.pop("api_key", None)

        # Anthropic uses 'system' as a top-level param, not in messages
        system_text = None
        anthropic_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_text = msg.get("content", "")
            else:
                anthropic_messages.append(msg)

        # Bedrock client uses AWS credentials from env/boto3
        client = anthropic.AnthropicBedrock(
            aws_region=os.environ.get("AWS_REGION", "us-east-1"),
        )

        call_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
        }
        if system_text:
            call_kwargs["system"] = system_text
        if "max_tokens" not in kwargs:
            kwargs["max_tokens"] = 4096

        call_kwargs.update(kwargs)
        return client.messages.create(**call_kwargs)

    @staticmethod
    def _call_azure(messages: list[dict], model: str, config: dict, **kwargs: Any) -> Any:
        """Call Azure OpenAI."""
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for provider='azure'. "
                "Install it with: pip install openai"
            )

        api_key = AgentBay._get_provider_key(config, **kwargs)
        kwargs.pop("api_key", None)
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
        if not endpoint:
            raise AgentBayError(
                "AZURE_OPENAI_ENDPOINT env var is required for provider='azure'."
            )

        client = openai.AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        call_kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        call_kwargs.update(kwargs)
        return client.chat.completions.create(**call_kwargs)

    @staticmethod
    def _call_openai_compatible(messages: list[dict], model: str, config: dict, **kwargs: Any) -> Any:
        """Call any OpenAI-compatible API (OpenAI, xAI, Mistral, DeepSeek,
        Together, Fireworks, Groq, Perplexity, Ollama, LM Studio, llama.cpp).
        """
        try:
            import openai
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for this provider. "
                "Install it with: pip install openai"
            )

        api_key = AgentBay._get_provider_key(config, **kwargs)
        kwargs.pop("api_key", None)
        base_url = config.get("base_url")

        client = openai.OpenAI(api_key=api_key, base_url=base_url)
        call_kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        call_kwargs.update(kwargs)
        return client.chat.completions.create(**call_kwargs)

    @staticmethod
    def _call_google(messages: list[dict], model: str, config: dict, **kwargs: Any) -> Any:
        """Call Google Gemini via the google-generativeai SDK."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "The 'google-generativeai' package is required for provider='google'. "
                "Install it with: pip install google-generativeai"
            )

        api_key = AgentBay._get_provider_key(config, **kwargs)
        kwargs.pop("api_key", None)
        genai.configure(api_key=api_key)

        # Convert OpenAI-format messages to Gemini format
        system_text = None
        gemini_history = []
        last_content = ""

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_text = content
            elif role == "assistant":
                gemini_history.append({"role": "model", "parts": [content]})
            elif role == "user":
                last_content = content
                # Don't add the last user message to history -- it goes in generate_content
                gemini_history.append({"role": "user", "parts": [content]})

        # Pop the last user message from history (it's the prompt to generate_content)
        if gemini_history and gemini_history[-1]["role"] == "user":
            last_msg = gemini_history.pop()
            last_content = last_msg["parts"][0]

        gen_config = {}
        if "temperature" in kwargs:
            gen_config["temperature"] = kwargs.pop("temperature")
        if "max_tokens" in kwargs:
            gen_config["max_output_tokens"] = kwargs.pop("max_tokens")

        gm = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_text,
            generation_config=gen_config or None,
        )

        chat = gm.start_chat(history=gemini_history) if gemini_history else gm.start_chat()
        return chat.send_message(last_content)

    @staticmethod
    def _call_cohere(messages: list[dict], model: str, config: dict, **kwargs: Any) -> Any:
        """Call Cohere's chat API."""
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "The 'cohere' package is required for provider='cohere'. "
                "Install it with: pip install cohere"
            )

        api_key = AgentBay._get_provider_key(config, **kwargs)
        kwargs.pop("api_key", None)

        client = cohere.ClientV2(api_key=api_key)

        # Cohere v2 chat accepts OpenAI-format messages directly
        call_kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        if "max_tokens" in kwargs:
            call_kwargs["max_tokens"] = kwargs.pop("max_tokens")
        if "temperature" in kwargs:
            call_kwargs["temperature"] = kwargs.pop("temperature")

        call_kwargs.update(kwargs)
        return client.chat(**call_kwargs)

    @staticmethod
    def _extract_response_text(response: Any, provider: str) -> str | None:
        """Extract the text content from an LLM response."""
        config = SUPPORTED_PROVIDERS.get(provider, {})
        module = config.get("module", "")
        try:
            if module == "anthropic" or provider in ("anthropic", "bedrock"):
                # Anthropic Message object
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return None
            elif module == "google.generativeai":
                # Google Gemini response
                return response.text
            elif module == "cohere":
                # Cohere v2 ChatResponse
                if hasattr(response, "message") and hasattr(response.message, "content"):
                    parts = response.message.content
                    if parts and hasattr(parts[0], "text"):
                        return parts[0].text
                return None
            else:
                # OpenAI-compatible (OpenAI, xAI, Groq, DeepSeek, Together, etc.)
                return response.choices[0].message.content
        except (AttributeError, IndexError, TypeError):
            return None
        return None

    def _auto_store_learnings(self, user_msg: str, assistant_text: str, project_id: str | None) -> None:
        """Extract and store learnings from a conversation (runs in background)."""
        try:
            # Check if the response contains something worth storing
            if not _LEARNING_PATTERNS.search(assistant_text):
                return

            # Extract the most relevant paragraph containing the learning
            paragraphs = assistant_text.split("\n\n")
            best_paragraph = ""
            for para in paragraphs:
                if _LEARNING_PATTERNS.search(para):
                    best_paragraph = para.strip()
                    break

            if not best_paragraph:
                best_paragraph = assistant_text[:500]

            # Build a concise memory entry
            entry_type = _detect_type(best_paragraph)
            title = _extract_title(best_paragraph)
            content = f"Q: {user_msg[:200]}\nA: {best_paragraph[:800]}"

            if self._is_local:
                self._local.store(
                    content=content,
                    title=title,
                    type=entry_type,
                    tags=["auto-learned", "chat"],
                )
            else:
                self.store(
                    content=content,
                    title=title,
                    project_id=project_id,
                    type=entry_type,
                    tags=["auto-learned", "chat"],
                )
        except Exception:
            # Fire-and-forget -- never crash the caller
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_project(self, project_id: str | None) -> str:
        pid = project_id or self.project_id
        if not pid:
            # Auto-setup: create a brain on first use so users don't have to
            try:
                result = self.setup_brain("My Brain")
                pid = result.get("projectId") or result.get("project", {}).get("id")
            except Exception:
                pass
        if not pid:
            raise AgentBayError(
                "No project_id provided. Either pass project_id to this method, "
                "set it in the constructor, or call setup_brain() first."
            )
        return pid

    def _post(self, path: str, body: Dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.post(url, json=body, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise AgentBayError(f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise AgentBayError(f"Request timed out after {self.timeout}s") from exc
        return self._handle_response(resp)

    def _get(self, path: str, params: Dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise AgentBayError(f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise AgentBayError(f"Request timed out after {self.timeout}s") from exc
        return self._handle_response(resp)

    def _patch(self, path: str, body: Dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.patch(url, json=body, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise AgentBayError(f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise AgentBayError(f"Request timed out after {self.timeout}s") from exc
        return self._handle_response(resp)

    def _delete(self, path: str, body: Dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.delete(url, json=body, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise AgentBayError(f"Connection failed: {exc}") from exc
        except requests.Timeout as exc:
            raise AgentBayError(f"Request timed out after {self.timeout}s") from exc
        return self._handle_response(resp)

    def _handle_response(self, resp: requests.Response) -> Any:
        if resp.status_code == 401:
            raise AuthenticationError(
                "Invalid or expired API key. Get a new one at https://www.aiagentsbay.com/dashboard/api-keys",
                status_code=401,
            )
        if resp.status_code == 403:
            raise AgentBayError(
                "Your plan doesn't include this feature. See plans at https://www.aiagentsbay.com/pricing",
                status_code=403,
                help_url="https://www.aiagentsbay.com/pricing",
            )
        if resp.status_code == 404:
            raise NotFoundError(
                "Project or memory entry not found. Check your project_id at https://www.aiagentsbay.com/dashboard",
                status_code=404,
            )
        if resp.status_code == 429:
            raise RateLimitError(
                "Rate limit exceeded. Upgrade your plan at https://www.aiagentsbay.com/pricing",
                status_code=429,
            )
        if resp.status_code >= 500:
            raise AgentBayError(
                f"AgentBay server error ({resp.status_code}). Check status at https://www.aiagentsbay.com/status",
                status_code=resp.status_code,
                help_url="https://www.aiagentsbay.com/status",
            )
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except ValueError:
                detail = {"text": resp.text}
            raise AgentBayError(
                f"API error {resp.status_code}: {detail.get('error', detail)}",
                status_code=resp.status_code,
                response=detail,
            )
        if resp.status_code == 204 or not resp.content:
            return {}
        return resp.json()

    # ------------------------------------------------------------------
    # Teams & Projects wrappers
    # ------------------------------------------------------------------

    def team(self, team_id: str) -> "TeamContext":
        """Create a team context. While active, chat() auto-shares with teammates.

        Args:
            team_id: The team ID to scope operations to.

        Returns:
            A :class:`TeamContext` that provides team-aware chat and recall.
        """
        return TeamContext(self, team_id)

    def project(self, project_id: str | None = None) -> "ProjectContext":
        """Create a project context. chat() auto-recalls from project memory,
        auto-stores to project, auto-onboards on first call.

        Args:
            project_id: Project to use (falls back to the brain's default).

        Returns:
            A :class:`ProjectContext` that provides project-aware chat and memory.
        """
        pid = project_id or self.project_id
        if not pid:
            raise AgentBayError(
                "No project_id provided. Either pass project_id, "
                "set it in the constructor, or call setup_brain() first."
            )
        return ProjectContext(self, pid)

    def create_team(self, name: str, agent_ids: list[str] | None = None) -> dict:
        """Create a team and optionally add agents.

        Args:
            name: Human-readable team name.
            agent_ids: Optional list of agent IDs to add as members.

        Returns:
            Dict with team details including ``teamId``.
        """
        body: Dict[str, Any] = {"name": name}
        if agent_ids:
            body["agentIds"] = agent_ids
        return self._post("/api/v1/teams", body)

    def create_project(self, name: str, description: str | None = None) -> dict:
        """Create a project.

        Args:
            name: Human-readable project name.
            description: Optional project description.

        Returns:
            Dict with project details including ``id``.
        """
        body: Dict[str, Any] = {"name": name}
        if description is not None:
            body["description"] = description
        return self._post("/api/v1/projects", body)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def upgrade(self, api_key: str, project_id: str | None = None) -> "AgentBay":
        """Upgrade from local to cloud. Migrates all local memories.

        Reads every entry from the local SQLite store and uploads it to
        the cloud Knowledge Brain via the API. Returns a new cloud-mode
        :class:`AgentBay` instance.

        Args:
            api_key: Your AgentBay API key.
            project_id: Cloud project ID to migrate into.

        Returns:
            A cloud :class:`AgentBay` instance with all local memories migrated.

        Raises:
            AgentBayError: If already using cloud mode.
        """
        if not self._is_local:
            raise AgentBayError("Already using cloud mode")
        return self._local.upgrade(api_key, project_id)

    def login(self, migrate: bool = True) -> "AgentBay":
        """Log in via browser and save the returned API key to local config.

        Opens the AgentBay CLI auth page in your browser and waits for the
        one-time loopback callback to arrive locally. Optionally migrates
        local memories to the cloud account that completed the login.

        Usage::

            brain = AgentBay()  # local mode
            brain.store("my pattern", title="test")

            # Ready for cloud? One command:
            brain = brain.login()
            # Opens browser → you sign up → paste key → done
            # All local memories are now in the cloud

        Returns:
            A cloud :class:`AgentBay` instance.
        """
        base = self.base_url if hasattr(self, 'base_url') and self.base_url else "https://www.aiagentsbay.com"
        try:
            result = login_via_browser(
                base_url=base,
                version=__version__,
                print_fn=print,
            )
        except Exception as exc:
            raise AgentBayError(str(exc)) from exc

        api_key = result.api_key
        print(f"✓ Key saved to {result.config_path}")

        # Migrate local memories if requested
        if migrate and self._is_local and self._local:
            cloud = self._local.upgrade(api_key)
            print(f"✓ Local memories migrated to cloud")
            return cloud

        cloud = AgentBay(api_key, base_url=base, project_id=getattr(self, "project_id", None))
        print(f"\n✓ Ready! You're now using AgentBay cloud.")
        return cloud

    @staticmethod
    def from_saved() -> "AgentBay":
        """Load AgentBay from saved config (~/.agentbay/config.json).

        If you previously ran ``brain.login()``, the API key was saved.
        This loads it so you don't have to pass the key every time.

        Usage::

            brain = AgentBay.from_saved()  # uses saved key
            brain.recall("my patterns")

        Returns:
            A cloud :class:`AgentBay` instance.

        Raises:
            AgentBayError: If no saved config found.
        """
        config = load_config()
        if not config:
            raise AgentBayError(
                "No saved config found. Run brain.login() first, or pass an API key."
            )

        api_key = config.get("apiKey") or config.get("api_key")
        if not api_key:
            raise AgentBayError("No API key in saved config. Run brain.login() first.")

        return AgentBay(
            api_key=api_key,
            base_url=config.get("baseUrl", "https://www.aiagentsbay.com"),
            project_id=config.get("projectId"),
        )

    def offline_project(self, project_name: str) -> 'OfflineProject':
        """Create or open an offline project. Works fully offline with local SQLite.

        Usage:
            proj = brain.offline_project("my-project")
            proj.ingest([files])
            proj.chat([messages])
            proj.sync("ab_live_key")  # when ready for cloud
        """
        from .offline import OfflineProject
        return OfflineProject(project_name)

    def offline_team(self, team_name: str) -> 'OfflineTeam':
        """Create or open an offline team. Agents on the same machine share memory.

        Usage:
            team = brain.offline_team("my-team")
            team.store("pattern", agent_name="claude")
            team.recall("pattern")  # sees all agents' entries
            team.sync("ab_live_key")  # when ready for cloud
        """
        from .offline import OfflineTeam
        return OfflineTeam(team_name)

    def __repr__(self) -> str:
        if self._is_local:
            return f"AgentBay(mode='local', db='{self._local.db_path}')"
        masked = f"{self.api_key[:8]}...{self.api_key[-4:]}" if len(self.api_key) > 12 else "***"
        return f"AgentBay(api_key='{masked}', project_id={self.project_id!r})"


# ======================================================================
# TeamContext
# ======================================================================


class TeamContext:
    """Context manager for team-aware memory.

    When using a ``TeamContext``, recall searches across all teammates'
    brains, and stored memories are tagged with ``scope='team'`` so they
    are visible to the entire team.

    Usage::

        brain = AgentBay("key", project_id="proj123")
        team = brain.team("team456")

        # chat() recalls from your brain + all teammates' brains
        response = team.chat([{"role": "user", "content": "fix the auth bug"}])

        # Memories auto-stored to your brain AND visible to team
    """

    def __init__(self, brain: AgentBay, team_id: str) -> None:
        self.brain = brain
        self.team_id = team_id

    def chat(
        self,
        messages: list[dict],
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        project_id: str | None = None,
        auto_recall: bool = True,
        auto_store: bool = True,
        recall_limit: int = 3,
        **kwargs: Any,
    ) -> Any:
        """Like brain.chat() but recalls from team members too.

        Memory recall searches across all teammates' brains via
        ``scope='team'``. Stored learnings are tagged with ``scope='team'``
        so teammates can see them.

        Args:
            messages: Chat messages in OpenAI format.
            model: Model name. Defaults to ``claude-sonnet-4-20250514``.
            provider: ``"anthropic"`` or ``"openai"``.
            project_id: Project for memory ops (overrides default).
            auto_recall: Whether to recall relevant memories. Defaults to True.
            auto_store: Whether to store learnings. Defaults to True.
            recall_limit: Max memories to recall (1-10). Defaults to 3.
            **kwargs: Extra keyword arguments passed to the LLM client.

        Returns:
            The raw LLM response object.
        """
        pid = self.brain._resolve_project(project_id)
        enriched_messages = list(messages)
        memory_context = ""

        # --- 1. Auto-recall from team scope ---
        if auto_recall:
            last_user_msg = AgentBay._extract_last_user_message(messages)
            if last_user_msg:
                try:
                    memories = self.recall(last_user_msg, project_id=pid, limit=recall_limit)
                    if memories:
                        memory_context = AgentBay._format_memory_context(memories)
                        enriched_messages = AgentBay._inject_memory_context(
                            enriched_messages, memory_context, provider
                        )
                except Exception:
                    pass

        # --- 2. Call the LLM ---
        response = AgentBay._call_llm(enriched_messages, model, provider, **kwargs)

        # --- 3. Auto-store with team scope ---
        if auto_store:
            last_user_msg = AgentBay._extract_last_user_message(messages)
            assistant_text = AgentBay._extract_response_text(response, provider)
            if last_user_msg and assistant_text:
                thread = threading.Thread(
                    target=self._auto_store_team_learnings,
                    args=(last_user_msg, assistant_text, pid),
                    daemon=True,
                )
                thread.start()

        return response

    def recall(
        self,
        query: str,
        project_id: str | None = None,
        limit: int = 5,
        tier: str | None = None,
        tags: list[str] | None = None,
    ) -> List[MemoryEntry]:
        """Recall from own brain + all teammates' brains.

        Args:
            query: Natural-language search query.
            project_id: Project to search in (overrides default).
            limit: Max results. Defaults to 5.
            tier: Filter by storage tier.
            tags: Filter by tags.

        Returns:
            Deduplicated list of matching entries from the team.
        """
        pid = self.brain._resolve_project(project_id)
        body: Dict[str, Any] = {
            "query": query,
            "limit": limit,
            "scope": "team",
            "teamId": self.team_id,
        }
        if tier is not None:
            body["tier"] = tier
        if tags is not None:
            body["tags"] = tags

        params: Dict[str, Any] = {"q": query, "limit": str(limit), "scope": "team"}
        if tags is not None:
            params["tags"] = ",".join(tags)
        resp = self.brain._get(f"/api/v1/projects/{pid}/memory", params)
        if isinstance(resp, list):
            return resp
        return resp.get("results", resp.get("entries", []))

    def members(self) -> List[Dict[str, Any]]:
        """List team members and their agents.

        Returns:
            List of dicts with member info (agent ID, name, role, etc.).
        """
        return self.brain._get(f"/api/v1/teams/{self.team_id}/members")

    def _auto_store_team_learnings(self, user_msg: str, assistant_text: str, project_id: str) -> None:
        """Extract and store learnings with team scope (runs in background)."""
        try:
            if not _LEARNING_PATTERNS.search(assistant_text):
                return

            paragraphs = assistant_text.split("\n\n")
            best_paragraph = ""
            for para in paragraphs:
                if _LEARNING_PATTERNS.search(para):
                    best_paragraph = para.strip()
                    break

            if not best_paragraph:
                best_paragraph = assistant_text[:500]

            entry_type = _detect_type(best_paragraph)
            title = _extract_title(best_paragraph)
            content = f"Q: {user_msg[:200]}\nA: {best_paragraph[:800]}"

            body: Dict[str, Any] = {
                "content": content,
                "type": entry_type,
                "tier": "semantic",
                "title": title,
                "tags": ["auto-learned", "chat", "team"],
                "scope": "team",
                "teamId": self.team_id,
            }
            self.brain._post(f"/api/v1/projects/{project_id}/memory", body)
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"TeamContext(team_id={self.team_id!r}, brain={self.brain!r})"


# ======================================================================
# ProjectContext
# ======================================================================


class ProjectContext:
    """Context manager for project-aware memory.

    When using a ``ProjectContext``, recall searches project memory
    (shared across all project agents), and stored memories go into the
    project's shared knowledge base. Auto-onboards on the first chat call.

    Usage::

        brain = AgentBay("key")
        proj = brain.project("proj123")

        # Auto-onboards on first call, recalls from project memory
        response = proj.chat([{"role": "user", "content": "refactor the API"}])
        # Learnings stored to project memory for all agents to see
    """

    def __init__(self, brain: AgentBay, project_id: str) -> None:
        self.brain = brain
        self.project_id = project_id
        self._onboarded = False
        self._onboard_brief: str | None = None

    def chat(
        self,
        messages: list[dict],
        model: str = "claude-sonnet-4-20250514",
        provider: str = "anthropic",
        auto_recall: bool = True,
        auto_store: bool = True,
        recall_limit: int = 3,
        **kwargs: Any,
    ) -> Any:
        """Like brain.chat() but with project memory.

        On the first call, automatically onboards (fetches project brief,
        open tasks, knowledge, and latest handoff) and prepends the brief
        to context. Subsequent calls just recall from project memory.

        Args:
            messages: Chat messages in OpenAI format.
            model: Model name. Defaults to ``claude-sonnet-4-20250514``.
            provider: ``"anthropic"`` or ``"openai"``.
            auto_recall: Whether to recall relevant memories. Defaults to True.
            auto_store: Whether to store learnings. Defaults to True.
            recall_limit: Max memories to recall (1-10). Defaults to 3.
            **kwargs: Extra keyword arguments passed to the LLM client.

        Returns:
            The raw LLM response object.
        """
        enriched_messages = list(messages)

        # --- 1. Auto-onboard on first call ---
        if not self._onboarded:
            try:
                onboard_data = self.onboard()
                self._onboarded = True
                brief_parts = []
                if isinstance(onboard_data, dict):
                    if onboard_data.get("brief"):
                        brief_parts.append(f"Project brief: {onboard_data['brief']}")
                    if onboard_data.get("handoff"):
                        brief_parts.append(f"Latest handoff: {onboard_data['handoff']}")
                    if onboard_data.get("tasks"):
                        tasks_str = ", ".join(
                            t.get("title", str(t)) for t in onboard_data["tasks"][:5]
                        ) if isinstance(onboard_data["tasks"], list) else str(onboard_data["tasks"])
                        brief_parts.append(f"Open tasks: {tasks_str}")
                if brief_parts:
                    self._onboard_brief = "\n".join(brief_parts)
            except Exception:
                self._onboarded = True  # Don't retry on failure

        # --- 2. Recall from project memory ---
        if auto_recall:
            last_user_msg = AgentBay._extract_last_user_message(messages)
            if last_user_msg:
                try:
                    memories = self.recall(last_user_msg, limit=recall_limit)
                    context_parts = []
                    if self._onboard_brief:
                        context_parts.append(
                            f"[Project onboarding context]\n{self._onboard_brief}"
                        )
                    if memories:
                        context_parts.append(AgentBay._format_memory_context(memories))
                    if context_parts:
                        full_context = "\n\n".join(context_parts)
                        enriched_messages = AgentBay._inject_memory_context(
                            enriched_messages, full_context, provider
                        )
                except Exception:
                    # Still inject onboard brief if recall failed
                    if self._onboard_brief:
                        enriched_messages = AgentBay._inject_memory_context(
                            enriched_messages,
                            f"[Project onboarding context]\n{self._onboard_brief}",
                            provider,
                        )
        elif self._onboard_brief:
            enriched_messages = AgentBay._inject_memory_context(
                enriched_messages,
                f"[Project onboarding context]\n{self._onboard_brief}",
                provider,
            )

        # --- 3. Call the LLM ---
        response = AgentBay._call_llm(enriched_messages, model, provider, **kwargs)

        # --- 4. Auto-store to project memory ---
        if auto_store:
            last_user_msg = AgentBay._extract_last_user_message(messages)
            assistant_text = AgentBay._extract_response_text(response, provider)
            if last_user_msg and assistant_text:
                thread = threading.Thread(
                    target=self._auto_store_project_learnings,
                    args=(last_user_msg, assistant_text),
                    daemon=True,
                )
                thread.start()

        return response

    def recall(
        self,
        query: str,
        limit: int = 5,
        tier: str | None = None,
        tags: list[str] | None = None,
    ) -> List[MemoryEntry]:
        """Recall from project memory.

        Args:
            query: Natural-language search query.
            limit: Max results. Defaults to 5.
            tier: Filter by storage tier.
            tags: Filter by tags.

        Returns:
            List of matching entries from project memory.
        """
        body: Dict[str, Any] = {
            "query": query,
            "limit": limit,
        }
        if tier is not None:
            body["tier"] = tier
        if tags is not None:
            body["tags"] = tags

        params: Dict[str, Any] = {"q": query, "limit": str(limit)}
        if tags is not None:
            params["tags"] = ",".join(tags)
        resp = self.brain._get(f"/api/v1/projects/{self.project_id}/memory", params)
        if isinstance(resp, list):
            return resp
        return resp.get("results", resp.get("entries", []))

    def store(
        self,
        content: str,
        title: str | None = None,
        type: str = "PATTERN",
        tier: str = "semantic",
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        """Store to project memory (visible to all project agents).

        Args:
            content: The knowledge content to store.
            title: Optional short title.
            type: Entry type -- PATTERN, FACT, PREFERENCE, PROCEDURE, CONTEXT.
            tier: Storage tier -- semantic, episodic, procedural.
            tags: Optional tags for categorization.

        Returns:
            Dict with the created entry.
        """
        body: Dict[str, Any] = {
            "content": content,
            "type": type,
            "tier": tier,
        }
        if title is not None:
            body["title"] = title
        if tags is not None:
            body["tags"] = tags

        return self.brain._post(f"/api/v1/projects/{self.project_id}/memory", body)

    def ingest(self, files: list[dict]) -> Dict[str, Any]:
        """Ingest files into project memory.

        Args:
            files: List of dicts with ``name`` and ``content`` keys.
                Example: ``[{"name": "README.md", "content": "# My Project..."}]``

        Returns:
            Dict with ingestion results.
        """
        body: Dict[str, Any] = {"files": files}
        return self.brain._post(f"/api/v1/projects/{self.project_id}/memory/ingest", body)

    def handoff(
        self,
        summary: str,
        completed_steps: list[str] | None = None,
        blockers: list[str] | None = None,
        next_steps: list[str] | None = None,
    ) -> MemoryEntry:
        """Hand off to the next agent with structured context.

        Stores a CONTEXT entry with structured handoff data so the next
        agent (or next session) can pick up where you left off.

        Args:
            summary: High-level summary of what was done.
            completed_steps: List of completed work items.
            blockers: List of known blockers or issues.
            next_steps: List of recommended next actions.

        Returns:
            Dict with the created handoff entry.
        """
        handoff_data: Dict[str, Any] = {"summary": summary}
        if completed_steps:
            handoff_data["completedSteps"] = completed_steps
        if blockers:
            handoff_data["blockers"] = blockers
        if next_steps:
            handoff_data["nextSteps"] = next_steps

        content_parts = [f"Handoff Summary: {summary}"]
        if completed_steps:
            content_parts.append("Completed: " + "; ".join(completed_steps))
        if blockers:
            content_parts.append("Blockers: " + "; ".join(blockers))
        if next_steps:
            content_parts.append("Next steps: " + "; ".join(next_steps))

        return self.store(
            content="\n".join(content_parts),
            title=f"Handoff: {_extract_title(summary, max_len=80)}",
            type="CONTEXT",
            tags=["handoff"],
        )

    def onboard(self) -> Dict[str, Any]:
        """Get project brief, open tasks, knowledge, and latest handoff.

        Returns:
            Dict with onboarding info (brief, tasks, knowledge, handoff).
        """
        return self.brain._post(f"/api/v1/projects/{self.project_id}/onboard", {})

    def _auto_store_project_learnings(self, user_msg: str, assistant_text: str) -> None:
        """Extract and store learnings to project memory (runs in background)."""
        try:
            if not _LEARNING_PATTERNS.search(assistant_text):
                return

            paragraphs = assistant_text.split("\n\n")
            best_paragraph = ""
            for para in paragraphs:
                if _LEARNING_PATTERNS.search(para):
                    best_paragraph = para.strip()
                    break

            if not best_paragraph:
                best_paragraph = assistant_text[:500]

            entry_type = _detect_type(best_paragraph)
            title = _extract_title(best_paragraph)
            content = f"Q: {user_msg[:200]}\nA: {best_paragraph[:800]}"

            self.store(
                content=content,
                title=title,
                type=entry_type,
                tags=["auto-learned", "chat", "project"],
            )
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"ProjectContext(project_id={self.project_id!r}, brain={self.brain!r})"
