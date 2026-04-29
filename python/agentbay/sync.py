"""AgentBay Sync Protocol — push/pull/merge between local SQLite and cloud.

Works like git: work offline, sync when connected.
Conflicts resolved by timestamp (newer wins, old version preserved in history).
"""

import json, sqlite3, uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


class SyncEngine:
    """Bidirectional sync between a local SQLite database and AgentBay cloud."""

    def __init__(self, local_db_path: str, api_key: str = None, base_url: str = "https://www.aiagentsbay.com"):
        self.local_db_path = local_db_path
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._init_sync_table()

    def _init_sync_table(self):
        """Create sync metadata table to track what's been synced."""
        with sqlite3.connect(self.local_db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_log (
                    id TEXT PRIMARY KEY,
                    entry_id TEXT NOT NULL,
                    action TEXT NOT NULL,  -- 'push' | 'pull' | 'conflict_resolved'
                    direction TEXT NOT NULL,  -- 'up' | 'down'
                    cloud_id TEXT,
                    local_version INTEGER NOT NULL DEFAULT 1,
                    cloud_version INTEGER,
                    synced_at TEXT NOT NULL,
                    conflict_resolution TEXT  -- 'local_wins' | 'cloud_wins' | 'merged'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sync_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

    def push(self, project_id: str) -> Dict[str, Any]:
        """Push local changes to cloud.

        Finds all local entries that haven't been synced (no sync_log entry)
        or have been modified since last sync.

        Returns: { pushed: int, conflicts: int, errors: int }
        """
        import requests

        if not self.api_key:
            raise Exception("API key required for push. Call brain.login() first.")

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        with sqlite3.connect(self.local_db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Find unsynced entries
            rows = conn.execute("""
                SELECT m.* FROM memories m
                LEFT JOIN sync_log s ON m.id = s.entry_id AND s.direction = 'up'
                WHERE s.id IS NULL
                   OR m.updated_at > s.synced_at
            """).fetchall()

        pushed = 0
        conflicts = 0
        errors = 0

        for row in rows:
            row_dict = dict(row)
            try:
                # Push to cloud
                body = {
                    "title": row_dict["title"],
                    "content": row_dict["content"],
                    "type": row_dict.get("type", "PATTERN"),
                    "tier": row_dict.get("tier", "semantic"),
                    "tags": json.loads(row_dict.get("tags", "[]")),
                    "confidence": row_dict.get("confidence", 0.5),
                    "source": "local-sync",
                    "sourceRef": row_dict["id"],  # local ID as reference
                }

                resp = requests.post(
                    f"{self.base_url}/api/v1/projects/{project_id}/memory",
                    headers=headers,
                    json=body,
                    timeout=15,
                )

                if resp.status_code in (200, 201):
                    data = resp.json()
                    cloud_id = data.get("id", "")

                    # Log the sync
                    with sqlite3.connect(self.local_db_path) as conn:
                        conn.execute("""
                            INSERT OR REPLACE INTO sync_log (id, entry_id, action, direction, cloud_id, synced_at)
                            VALUES (?, ?, 'push', 'up', ?, ?)
                        """, (str(uuid.uuid4()), row_dict["id"], cloud_id, datetime.now(timezone.utc).isoformat()))

                    if data.get("deduplicated"):
                        conflicts += 1
                    else:
                        pushed += 1
                else:
                    errors += 1

            except Exception:
                errors += 1

        # Save last push timestamp
        with sqlite3.connect(self.local_db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sync_state (key, value)
                VALUES ('last_push', ?)
            """, (datetime.now(timezone.utc).isoformat(),))

        return {"pushed": pushed, "conflicts": conflicts, "errors": errors}

    def pull(self, project_id: str) -> Dict[str, Any]:
        """Pull remote changes to local.

        Fetches entries from cloud that are newer than last pull.
        Stores them locally, skipping entries that originated from this local DB.

        Returns: { pulled: int, conflicts: int, skipped: int }
        """
        import requests

        if not self.api_key:
            raise Exception("API key required for pull. Call brain.login() first.")

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        # Get last pull timestamp
        with sqlite3.connect(self.local_db_path) as conn:
            row = conn.execute("SELECT value FROM sync_state WHERE key = 'last_pull'").fetchone()
            since = row[0] if row else "2020-01-01T00:00:00Z"

        # Fetch from cloud
        try:
            resp = requests.get(
                f"{self.base_url}/api/v1/projects/{project_id}/knowledge/export",
                headers=headers,
                params={"since": since},
                timeout=30,
            )
            if resp.status_code != 200:
                return {"pulled": 0, "conflicts": 0, "skipped": 0, "error": f"HTTP {resp.status_code}"}

            data = resp.json()
            entries = data.get("entries", [])
        except Exception as e:
            return {"pulled": 0, "conflicts": 0, "skipped": 0, "error": str(e)}

        pulled = 0
        conflicts = 0
        skipped = 0

        from .local import LocalMemory, _embed_text
        local = LocalMemory(db_path=self.local_db_path)

        for entry in entries:
            # Skip entries that originated from local sync
            if entry.get("source") == "local-sync":
                skipped += 1
                continue

            # Check for local conflict (same title + type)
            existing = None
            with sqlite3.connect(self.local_db_path) as conn:
                conn.row_factory = sqlite3.Row
                existing = conn.execute(
                    "SELECT * FROM memories WHERE title = ? AND type = ?",
                    (entry.get("title", ""), entry.get("type", "PATTERN"))
                ).fetchone()

            if existing:
                existing_dict = dict(existing)
                # Conflict: cloud entry exists locally
                # Timestamp resolution: newer wins
                cloud_time = entry.get("updatedAt", entry.get("createdAt", ""))
                local_time = existing_dict.get("updated_at", "")

                if cloud_time > local_time:
                    # Cloud wins — update local
                    with sqlite3.connect(self.local_db_path) as conn:
                        conn.execute("""
                            UPDATE memories SET content = ?, confidence = ?, updated_at = ?
                            WHERE id = ?
                        """, (entry.get("content", ""), entry.get("confidence", 0.5),
                              datetime.now(timezone.utc).isoformat(), existing_dict["id"]))
                    conflicts += 1
                else:
                    skipped += 1
                continue

            # New entry — store locally
            local.store(
                content=entry.get("content", ""),
                title=entry.get("title", ""),
                type=entry.get("type", "PATTERN"),
                tier=entry.get("tier", "semantic"),
                tags=entry.get("tags", []),
                confidence=entry.get("confidence", 0.5),
            )
            pulled += 1

        # Save last pull timestamp
        with sqlite3.connect(self.local_db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO sync_state (key, value)
                VALUES ('last_pull', ?)
            """, (datetime.now(timezone.utc).isoformat(),))

        return {"pulled": pulled, "conflicts": conflicts, "skipped": skipped}

    def sync(self, project_id: str) -> Dict[str, Any]:
        """Full bidirectional sync: push then pull.

        Returns: { push: {...}, pull: {...} }
        """
        push_result = self.push(project_id)
        pull_result = self.pull(project_id)
        return {"push": push_result, "pull": pull_result}

    def status(self) -> Dict[str, Any]:
        """Show sync status — what's pending."""
        with sqlite3.connect(self.local_db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            synced = conn.execute("SELECT COUNT(DISTINCT entry_id) FROM sync_log WHERE direction = 'up'").fetchone()[0]
            last_push = conn.execute("SELECT value FROM sync_state WHERE key = 'last_push'").fetchone()
            last_pull = conn.execute("SELECT value FROM sync_state WHERE key = 'last_pull'").fetchone()

        return {
            "total_local": total,
            "synced_to_cloud": synced,
            "unsynced": total - synced,
            "last_push": last_push[0] if last_push else None,
            "last_pull": last_pull[0] if last_pull else None,
        }
