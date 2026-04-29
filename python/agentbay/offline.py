"""Offline-capable Projects and Teams backed by local SQLite.

Work offline with full functionality. Sync to cloud when connected.
"""

import json, os, sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .local import LocalMemory
from .sync import SyncEngine


class OfflineProject:
    """Full project functionality backed by local SQLite + optional cloud sync.

    Usage:
        brain = AgentBay()
        proj = brain.offline_project("my-project")

        # Works fully offline
        proj.ingest([{"path": "src/auth.ts", "content": "..."}])
        proj.store("connection pooling pattern", title="DB tip")
        results = proj.recall("database")
        proj.handoff("done with API", next_steps=["test endpoints"])

        # Sync when connected
        proj.sync("ab_live_your_key")
    """

    def __init__(self, project_name: str, db_dir: str = None):
        if db_dir is None:
            db_dir = str(Path.home() / ".agentbay" / "projects")
        Path(db_dir).mkdir(parents=True, exist_ok=True)

        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_name)
        self.project_name = project_name
        self.db_path = str(Path(db_dir) / f"{safe_name}.db")
        self.memory = LocalMemory(db_path=self.db_path)
        self._syncer = None
        self._init_project_tables()

    def _init_project_tables(self):
        """Create project-specific tables (tasks, handoffs, files)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT NOT NULL DEFAULT 'OPEN',
                    priority TEXT NOT NULL DEFAULT 'MEDIUM',
                    assigned_to TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS handoffs (
                    id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    completed_steps TEXT DEFAULT '[]',
                    blockers TEXT DEFAULT '[]',
                    next_steps TEXT DEFAULT '[]',
                    files_modified TEXT DEFAULT '[]',
                    from_agent TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS project_files (
                    path TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    def store(self, content, **kwargs):
        """Store a memory in the project."""
        return self.memory.store(content, **kwargs)

    def recall(self, query, **kwargs):
        """Recall from project memory."""
        return self.memory.recall(query, **kwargs)

    def chat(self, messages, **kwargs):
        """Chat with auto-recall from project memory."""
        # Recall context
        last_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if last_msg:
            memories = self.recall(last_msg, limit=3)
            if memories:
                context = "\n---\n".join(
                    f"[{m['type']}] {m['title']}: {m['content']}" for m in memories
                )
                mem_msg = {"role": "system", "content": f"[Project memory]\n{context}"}
                messages = [mem_msg] + list(messages)

        # Call LLM (needs provider)
        from .client import AgentBay
        temp = AgentBay()  # local mode
        return temp.chat(messages, **kwargs)

    def ingest(self, files: List[Dict[str, str]], max_entries_per_file: int = 5):
        """Ingest files into project memory using local extraction.

        Uses Ollama -> API -> heuristics cascade for extraction.
        """
        from .local import _ollama_extract, _api_extract, _heuristic_extract

        total_stored = 0
        for f in files[:50]:  # max 50 files
            path = f.get("path", "unknown")
            content = f.get("content", "")
            if len(content) > 100_000:
                continue  # skip files > 100KB

            # Try extraction cascade
            entries = _ollama_extract(content[:4000], max_entries_per_file)
            source = "ollama"
            if entries is None:
                entries = _api_extract(content[:4000], max_entries_per_file)
                source = "api"
            if entries is None:
                entries = _heuristic_extract(content[:4000], max_entries_per_file)
                source = "heuristic"

            for entry in (entries or []):
                self.store(
                    content=entry.get("content", ""),
                    title=entry.get("title", ""),
                    type=entry.get("type", "PATTERN"),
                    tags=["ingest", f"file:{path}", f"source:{source}"],
                )
                total_stored += 1

            # Store file content for reference
            from datetime import datetime, timezone
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO project_files (path, content, updated_at)
                    VALUES (?, ?, ?)
                """, (path, content, datetime.now(timezone.utc).isoformat()))

        return {"files_processed": len(files), "entries_stored": total_stored}

    def create_task(self, title, description=None, priority="MEDIUM"):
        """Create a task in the project."""
        import uuid
        from datetime import datetime, timezone
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO tasks (id, title, description, priority, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (task_id, title, description, priority, now, now))
        return {"id": task_id, "title": title}

    def list_tasks(self, status=None):
        """List tasks in the project."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if status:
                rows = conn.execute("SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC", (status,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM tasks ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def claim_task(self, task_id, agent_name="local-agent"):
        """Claim a task."""
        from datetime import datetime, timezone
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE tasks SET status = 'IN_PROGRESS', assigned_to = ?, updated_at = ?
                WHERE id = ?
            """, (agent_name, datetime.now(timezone.utc).isoformat(), task_id))
        return {"claimed": True}

    def handoff(self, summary, completed_steps=None, blockers=None, next_steps=None, files_modified=None, from_agent=None):
        """Create a handoff for the next agent."""
        import uuid
        from datetime import datetime, timezone
        handoff_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO handoffs (id, summary, completed_steps, blockers, next_steps, files_modified, from_agent, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                handoff_id, summary,
                json.dumps(completed_steps or []),
                json.dumps(blockers or []),
                json.dumps(next_steps or []),
                json.dumps(files_modified or []),
                from_agent,
                datetime.now(timezone.utc).isoformat(),
            ))

        # Also store as a memory
        self.store(
            content=summary,
            title=f"Handoff: {summary[:60]}",
            type="CONTEXT",
            tier="episodic",
            tags=["handoff"],
        )

        return {"id": handoff_id}

    def resume(self):
        """Get the latest handoff to resume from."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM handoffs ORDER BY created_at DESC LIMIT 1").fetchone()
        if not row:
            return None
        result = dict(row)
        result["completed_steps"] = json.loads(result["completed_steps"])
        result["blockers"] = json.loads(result["blockers"])
        result["next_steps"] = json.loads(result["next_steps"])
        result["files_modified"] = json.loads(result["files_modified"])
        return result

    def sync(self, api_key: str = None, project_id: str = None):
        """Sync local project to cloud."""
        key = api_key or os.environ.get("AGENTBAY_API_KEY")
        if not key:
            raise Exception("API key required. Pass api_key or set AGENTBAY_API_KEY.")

        if self._syncer is None:
            self._syncer = SyncEngine(self.db_path, api_key=key)

        pid = project_id or self.project_name
        return self._syncer.sync(pid)

    def sync_status(self):
        """Check sync status."""
        if self._syncer is None:
            self._syncer = SyncEngine(self.db_path)
        return self._syncer.status()

    def health(self):
        """Project health stats."""
        mem_health = self.memory.health()
        with sqlite3.connect(self.db_path) as conn:
            task_count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            handoff_count = conn.execute("SELECT COUNT(*) FROM handoffs").fetchone()[0]
            file_count = conn.execute("SELECT COUNT(*) FROM project_files").fetchone()[0]
        return {
            **mem_health,
            "tasks": task_count,
            "handoffs": handoff_count,
            "files": file_count,
        }


class OfflineTeam:
    """Offline team — multiple agents share a SQLite database on the same machine.

    Usage:
        brain = AgentBay()
        team = brain.offline_team("my-team")

        # Agent A stores (visible to B)
        team.store("pattern found", title="Auth tip", agent_name="claude")

        # Agent B recalls (sees A's entries)
        team.recall("auth", agent_name="cursor")

        # Sync when connected
        team.sync("ab_live_your_key")
    """

    def __init__(self, team_name: str, db_dir: str = None):
        if db_dir is None:
            db_dir = str(Path.home() / ".agentbay" / "teams")
        Path(db_dir).mkdir(parents=True, exist_ok=True)

        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in team_name)
        self.team_name = team_name
        self.db_path = str(Path(db_dir) / f"{safe_name}.db")
        self.memory = LocalMemory(db_path=self.db_path)
        self._syncer = None

    def store(self, content, agent_name: str = "default", **kwargs):
        """Store a memory visible to all team members."""
        tags = kwargs.pop("tags", []) or []
        tags.append(f"agent:{agent_name}")
        return self.memory.store(content, tags=tags, **kwargs)

    def recall(self, query, agent_name: str = None, **kwargs):
        """Recall from team memory. All agents' entries are searched."""
        return self.memory.recall(query, **kwargs)

    def recall_from(self, query, agent_name: str, **kwargs):
        """Recall only from a specific agent's entries."""
        results = self.memory.recall(query, **kwargs)
        return [r for r in results if f"agent:{agent_name}" in r.get("tags", [])]

    def members(self):
        """List all agents that have stored memories in this team."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT DISTINCT tags FROM memories").fetchall()

        agents = set()
        for row in rows:
            tags = json.loads(row[0])
            for tag in tags:
                if tag.startswith("agent:"):
                    agents.add(tag[6:])
        return list(agents)

    def chat(self, messages, agent_name: str = "default", **kwargs):
        """Chat with team memory context."""
        last_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if last_msg:
            memories = self.recall(last_msg, limit=5)
            if memories:
                context = "\n---\n".join(
                    f"[{m['type']}] {m['title']} (by {next((t[6:] for t in m.get('tags',[]) if t.startswith('agent:')), '?')}): {m['content']}" for m in memories
                )
                mem_msg = {"role": "system", "content": f"[Team memory — shared by all agents]\n{context}"}
                messages = [mem_msg] + list(messages)

        from .client import AgentBay
        temp = AgentBay()
        response = temp.chat(messages, **kwargs)

        # Auto-store learnings for the team
        # (fire and forget in background)
        return response

    def sync(self, api_key: str = None, team_id: str = None):
        """Sync team memory to cloud."""
        key = api_key or os.environ.get("AGENTBAY_API_KEY")
        if not key:
            raise Exception("API key required for sync.")
        if self._syncer is None:
            self._syncer = SyncEngine(self.db_path, api_key=key)
        tid = team_id or self.team_name
        return self._syncer.sync(tid)

    def health(self):
        """Team health stats."""
        stats = self.memory.health()
        stats["members"] = self.members()
        return stats
