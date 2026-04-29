"""Production-grade local memory engine.

Zero external services required. Pure Python + SQLite.

Optional upgrades (auto-detected at init):
- pip install fastembed   -> vector search via ONNX Runtime (no GPU needed)
- ollama running locally  -> AI-powered knowledge extraction
- ANTHROPIC_API_KEY set   -> cloud extraction via Claude Haiku
- OPENAI_API_KEY set      -> cloud extraction via GPT-4o-mini
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import struct
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# =====================================================================
# Optional imports -- all guarded, nothing required beyond stdlib
# =====================================================================

_fastembed_cls = None   # TextEmbedding class (or None)
_embed_model = None     # singleton model instance (or None)
_fastembed_checked = False


def _get_embedder():
    """Lazy-load FastEmbed model on first use.

    Downloads ``all-MiniLM-L6-v2`` (~22 MB ONNX) on the very first call.
    Returns *None* if fastembed is not installed.
    """
    global _fastembed_cls, _embed_model, _fastembed_checked
    if _fastembed_checked:
        return _embed_model
    _fastembed_checked = True
    try:
        from fastembed import TextEmbedding  # type: ignore[import-untyped]
        _fastembed_cls = TextEmbedding
        _embed_model = _fastembed_cls(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except ImportError:
        _embed_model = None
    return _embed_model


def _embed_text(text: str) -> Optional[List[float]]:
    """Generate an embedding vector for *text*.

    Returns *None* when FastEmbed is not available.
    """
    model = _get_embedder()
    if model is None:
        return None
    embeddings = list(model.embed([text]))
    return embeddings[0].tolist() if embeddings else None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _pack_embedding(embedding: List[float]) -> bytes:
    """Pack a float list into a compact BLOB for SQLite storage."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _unpack_embedding(blob: bytes) -> List[float]:
    """Unpack a BLOB back into a float list."""
    n = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f"{n}f", blob))


# =====================================================================
# Extraction helpers -- cascading: Ollama -> API -> heuristic
# =====================================================================

def _ollama_extract(text: str, max_entries: int = 3) -> Optional[List[dict]]:
    """Extract knowledge entries using a local Ollama model.

    Picks the smallest available model to minimise latency.
    Returns *None* on any failure (Ollama not running, no models, etc.).
    """
    try:
        import requests as _requests  # noqa: F811
    except ImportError:
        return None

    try:
        resp = _requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code != 200:
            return None
        models = resp.json().get("models", [])
        if not models:
            return None

        # Prefer small models for fast extraction
        model_name = models[0]["name"]
        _small = ("gemma:2b", "qwen:0.5b", "phi3:mini", "gemma3:4b", "qwen3:4b")
        for m in models:
            if any(s in m["name"] for s in _small):
                model_name = m["name"]
                break

        prompt = (
            f"Extract up to {max_entries} reusable knowledge entries from this text.\n"
            "For each, provide: title (short), content (1-2 sentences), "
            "type (PATTERN, PITFALL, DECISION, or ARCHITECTURE).\n"
            'Return ONLY a JSON array: [{"title":"...","content":"...","type":"..."}]\n'
            "If nothing worth extracting, return [].\n\n"
            f"Text:\n{text[:2000]}"
        )

        resp = _requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False, "format": "json"},
            timeout=30,
        )
        if resp.status_code != 200:
            return None

        result = resp.json().get("response", "")
        match = re.search(r"\[.*\]", result, re.DOTALL)
        if match:
            entries = json.loads(match.group())
            return [e for e in entries if isinstance(e, dict) and "title" in e]
        return None
    except Exception:
        return None


def _api_extract(text: str, max_entries: int = 3) -> Optional[List[dict]]:
    """Extract knowledge via Anthropic or OpenAI cloud API.

    Checks ANTHROPIC_API_KEY first, then OPENAI_API_KEY.
    Returns *None* when no key is set or on any error.
    """
    _prompt = (
        f"Extract up to {max_entries} knowledge entries as a JSON array "
        '[{{"title":"...","content":"...","type":"PATTERN|PITFALL|DECISION|ARCHITECTURE"}}]. '
        f"Text:\n{text[:2000]}"
    )

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic  # type: ignore[import-untyped]
            client = anthropic.Anthropic(api_key=anthropic_key)
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=500,
                messages=[{"role": "user", "content": _prompt}],
            )
            result = resp.content[0].text
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai  # type: ignore[import-untyped]
            client = openai.OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=500,
                messages=[{"role": "user", "content": _prompt}],
            )
            result = resp.choices[0].message.content
            match = re.search(r"\[.*\]", result, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass

    return None


def _heuristic_extract(text: str, max_entries: int = 3) -> List[dict]:
    """Rule-based knowledge extraction -- always works, zero deps.

    Scans for common learning indicators and pulls out the sentence.
    """
    patterns = [
        (r"(?:the (?:issue|problem|bug|fix|solution|cause) (?:was|is)\s+.+?[.!?\n])", "PITFALL"),
        (r"(?:(?:always|never|make sure to|remember to|don't forget to)\s+.+?[.!?\n])", "PATTERN"),
        (r"(?:(?:we|i) (?:decided|chose|picked|went with|settled on)\s+.+?[.!?\n])", "DECISION"),
        (r"(?:(?:turns out|it works because|the trick is|the key (?:insight|thing))\s+.+?[.!?\n])", "PATTERN"),
        (r"(?:use\s+\S+\s+(?:instead of|rather than|not)\s+.+?[.!?\n])", "PATTERN"),
    ]
    entries: List[dict] = []
    seen: set = set()

    for pattern, entry_type in patterns:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            sentence = m.group(0).strip().rstrip(".")
            if sentence in seen or len(sentence) < 15:
                continue
            seen.add(sentence)
            entries.append({
                "title": _auto_title(sentence),
                "content": sentence,
                "type": entry_type,
            })
            if len(entries) >= max_entries:
                return entries

    return entries


# =====================================================================
# Text helpers
# =====================================================================

def _auto_title(text: str, max_len: int = 100) -> str:
    """Derive a short title from the first sentence or first N chars."""
    match = re.match(r"^(.+?[.!?])\s", text)
    if match and len(match.group(1)) <= max_len:
        return match.group(1)
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


def _auto_summary(text: str, max_len: int = 150) -> str:
    """Generate a brief summary: first sentence or first *max_len* chars."""
    match = re.match(r"^(.+?[.!?])\s", text)
    if match and len(match.group(1)) <= max_len:
        return match.group(1)
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


def _auto_type(text: str) -> str:
    """Auto-detect memory type from content keywords."""
    lower = text.lower()
    if re.search(r"\b(?:bug|error|fix|crash|fail|broke|issue|problem|exception)\b", lower):
        return "PITFALL"
    if re.search(r"\b(?:decided|chose|picked|went with|settled on|decision)\b", lower):
        return "DECISION"
    if re.search(r"\b(?:step\s*\d|first.*then|how to|procedure|process|workflow)\b", lower):
        return "PROCEDURE"
    if re.search(r"\b(?:architecture|component|layer|service|module|system design)\b", lower):
        return "ARCHITECTURE"
    return "PATTERN"


# =====================================================================
# Main class
# =====================================================================

class LocalMemory:
    """SQLite-backed local memory engine with FTS5 and optional vector search.

    Usage::

        from agentbay.local import LocalMemory

        mem = LocalMemory()                    # ~/.agentbay/local.db
        mem.store("Always use connection pooling", title="DB pattern")
        results = mem.recall("database connection")

    Capabilities are auto-detected and reported at init:
    - FTS5 full-text search (always on -- built into SQLite)
    - Vector cosine similarity (requires ``pip install fastembed``)
    - AI extraction via Ollama, Anthropic, or OpenAI (auto-detected)
    """

    def __init__(self, db_path: Optional[str] = None, quiet: bool = False):
        """Initialise the local memory engine.

        Args:
            db_path: Path to SQLite database file. Defaults to
                ``~/.agentbay/local.db``.
            quiet: If *True*, suppress the capability report printed at init.
        """
        if db_path is None:
            db_dir = Path.home() / ".agentbay"
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / "local.db")
        self.db_path = db_path
        self._init_db()
        self._migrate_schema()

        if not quiet:
            self._print_capabilities()

    # ------------------------------------------------------------------
    # Database initialisation
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Create tables, FTS5 virtual table, triggers, and indexes."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")

            # Main storage table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    type TEXT NOT NULL DEFAULT 'PATTERN',
                    tier TEXT NOT NULL DEFAULT 'semantic',
                    tags TEXT NOT NULL DEFAULT '[]',
                    user_id TEXT,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    embedding BLOB,
                    summary TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    helpful_count INTEGER NOT NULL DEFAULT 0
                )
            """)

            # FTS5 virtual table for fast full-text search
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    title, content, tags,
                    content=memories,
                    content_rowid=rowid
                )
            """)

            # Triggers to keep FTS5 in sync with the main table
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, title, content, tags)
                    VALUES (new.rowid, new.title, new.content, new.tags);
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, title, content, tags)
                    VALUES ('delete', old.rowid, old.title, old.content, old.tags);
                END
            """)
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, title, content, tags)
                    VALUES ('delete', old.rowid, old.title, old.content, old.tags);
                    INSERT INTO memories_fts(rowid, title, content, tags)
                    VALUES (new.rowid, new.title, new.content, new.tags);
                END
            """)

            # Indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_tier ON memories(tier)")

    def _migrate_schema(self) -> None:
        """Add columns that may be missing from older databases."""
        with sqlite3.connect(self.db_path) as conn:
            existing = {
                row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()
            }
            if "embedding" not in existing:
                conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
            if "summary" not in existing:
                conn.execute("ALTER TABLE memories ADD COLUMN summary TEXT")

    def _print_capabilities(self) -> None:
        """Print a one-time report of available search and extraction methods."""
        lines = ["[LocalMemory] capabilities:"]

        # Search methods
        lines.append("  search : FTS5 full-text (always on)")
        if _get_embedder() is not None:
            lines.append("  search : vector cosine similarity (fastembed)")
        lines.append("  search : keyword TF-IDF (always on)")

        # Extraction methods
        has_ollama = False
        try:
            import requests as _req
            resp = _req.get("http://localhost:11434/api/tags", timeout=1)
            if resp.status_code == 200 and resp.json().get("models"):
                has_ollama = True
        except Exception:
            pass
        if has_ollama:
            lines.append("  extract: Ollama (local LLM)")
        if os.environ.get("ANTHROPIC_API_KEY"):
            lines.append("  extract: Anthropic API")
        if os.environ.get("OPENAI_API_KEY"):
            lines.append("  extract: OpenAI API")
        lines.append("  extract: heuristic patterns (always on)")

        print("\n".join(lines))

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        title: Optional[str] = None,
        type: str = "PATTERN",
        tier: str = "semantic",
        tags: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """Store a memory entry with embedding and auto-summary.

        Performs semantic deduplication when embeddings are available
        (cosine similarity > 0.9), otherwise falls back to title matching.

        Args:
            content: The knowledge content to store.
            title: Short title (auto-generated if *None*).
            type: Entry type -- PATTERN, PITFALL, DECISION, PROCEDURE,
                ARCHITECTURE, FACT, PREFERENCE, CONTEXT.
            tier: Storage tier -- semantic, episodic, procedural.
            tags: Optional list of tags for filtering.
            user_id: Optional user ID for scoping.
            confidence: Initial confidence (0.0--1.0). Defaults to 0.5.

        Returns:
            Dict with ``id`` and ``deduplicated`` keys.
        """
        if title is None:
            title = _auto_title(content)

        tags_json = json.dumps(tags or [])
        now = datetime.now(timezone.utc).isoformat()

        # Generate embedding (returns None if fastembed not installed)
        embed_input = f"{title} {content}"
        embedding = _embed_text(embed_input)
        embedding_blob = _pack_embedding(embedding) if embedding else None

        # Auto-summary
        summary = _auto_summary(content)

        # --- Semantic dedup via embeddings ---
        # RF-13: previous threshold of 0.9 collapsed many distinct memories
        # whose only shared structure was a templated phrase (e.g., 20 entries
        # of "feature flag N uses codeword N" deduped to 11). Now uses 0.97
        # (near-identical only) AND requires the candidate match to share
        # the same title (case-insensitive) — distinct user-supplied titles
        # are treated as distinct memories regardless of semantic similarity.
        if embedding:
            existing = self._find_similar(
                embedding, threshold=0.97, user_id=user_id, title=title,
            )
            if existing:
                self._update_existing(existing["id"], content, tags_json, confidence, now)
                return {"id": existing["id"], "deduplicated": True}
        else:
            # Fallback: title + type dedup
            with sqlite3.connect(self.db_path) as conn:
                row = conn.execute(
                    "SELECT id FROM memories WHERE title = ? AND type = ? "
                    "AND (user_id = ? OR (user_id IS NULL AND ? IS NULL))",
                    (title, type, user_id, user_id),
                ).fetchone()
                if row:
                    self._update_existing(row[0], content, tags_json, confidence, now)
                    return {"id": row[0], "deduplicated": True}

        # --- Insert new entry ---
        entry_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO memories
                   (id, title, content, type, tier, tags, user_id, confidence,
                    embedding, summary, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (entry_id, title, content, type, tier, tags_json, user_id,
                 confidence, embedding_blob, summary, now, now),
            )
        return {"id": entry_id, "deduplicated": False}

    def _update_existing(
        self, entry_id: str, content: str, tags_json: str,
        confidence: float, now: str,
    ) -> None:
        """Update an existing entry during deduplication."""
        summary = _auto_summary(content)
        embedding = _embed_text(content)
        embedding_blob = _pack_embedding(embedding) if embedding else None

        with sqlite3.connect(self.db_path) as conn:
            if embedding_blob:
                conn.execute(
                    "UPDATE memories SET content = ?, tags = ?, confidence = ?, "
                    "summary = ?, embedding = ?, updated_at = ? WHERE id = ?",
                    (content, tags_json, confidence, summary, embedding_blob, now, entry_id),
                )
            else:
                conn.execute(
                    "UPDATE memories SET content = ?, tags = ?, confidence = ?, "
                    "summary = ?, updated_at = ? WHERE id = ?",
                    (content, tags_json, confidence, summary, now, entry_id),
                )

    # ------------------------------------------------------------------
    # Recall -- 3-strategy fusion search
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        limit: int = 5,
        user_id: Optional[str] = None,
        type: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search using 3-strategy Reciprocal Rank Fusion.

        1. **FTS5** full-text search (inverted index, fast at any scale)
        2. **Vector** cosine similarity (if embeddings available)
        3. **Keyword** TF-IDF scoring (catches exact matches FTS5 may miss)

        Results are fused with RRF (k=30 for FTS5/vector, k=60 for keyword),
        boosted by confidence, and the top *limit* entries are returned.

        Args:
            query: Natural-language search string.
            limit: Maximum results. Defaults to 5.
            user_id: Filter to a specific user.
            type: Filter by entry type.
            tags: Entry must contain **all** listed tags.

        Returns:
            List of dicts with all entry fields plus ``score``.
        """
        results_by_id: Dict[str, Dict[str, Any]] = {}
        filter_tags = set(tags or [])

        # Strategy 1: FTS5 full-text search
        fts_results = self._fts5_search(query, user_id=user_id, type=type, limit=limit * 3)
        for rank, row in enumerate(fts_results):
            if filter_tags and not filter_tags.issubset(set(json.loads(row.get("tags", "[]"))
                                                           if isinstance(row.get("tags"), str)
                                                           else row.get("tags", []))):
                continue
            eid = row["id"]
            if eid not in results_by_id:
                results_by_id[eid] = {"entry": row, "score": 0.0}
            results_by_id[eid]["score"] += 1.0 / (30 + rank)

        # Strategy 2: Vector cosine similarity
        query_embedding = _embed_text(query)
        if query_embedding:
            vector_results = self._vector_search(
                query_embedding, user_id=user_id, type=type, limit=limit * 3,
            )
            for rank, (eid, _sim, row) in enumerate(vector_results):
                if filter_tags:
                    entry_tags = json.loads(row.get("tags", "[]")) if isinstance(row.get("tags"), str) else row.get("tags", [])
                    if not filter_tags.issubset(set(entry_tags)):
                        continue
                if eid not in results_by_id:
                    results_by_id[eid] = {"entry": row, "score": 0.0}
                results_by_id[eid]["score"] += 1.0 / (30 + rank)

        # Strategy 3: Keyword TF-IDF
        keyword_results = self._keyword_search(
            query, user_id=user_id, type=type, limit=limit * 3,
        )
        for rank, row in enumerate(keyword_results):
            if filter_tags:
                entry_tags = json.loads(row.get("tags", "[]")) if isinstance(row.get("tags"), str) else row.get("tags", [])
                if not filter_tags.issubset(set(entry_tags)):
                    continue
            eid = row["id"]
            if eid not in results_by_id:
                results_by_id[eid] = {"entry": row, "score": 0.0}
            results_by_id[eid]["score"] += 1.0 / (60 + rank)

        # Confidence boost
        for data in results_by_id.values():
            conf = data["entry"].get("confidence", 0.5)
            data["score"] *= 0.5 + 0.5 * conf

        # Sort, slice, and format
        sorted_results = sorted(
            results_by_id.values(), key=lambda x: x["score"], reverse=True,
        )

        final: List[Dict[str, Any]] = []
        ids_to_bump: List[str] = []
        for item in sorted_results[:limit]:
            entry = item["entry"]
            tags_val = entry.get("tags", "[]")
            if isinstance(tags_val, str):
                tags_val = json.loads(tags_val)
            final.append({
                "id": entry["id"],
                "title": entry["title"],
                "content": entry["content"],
                "type": entry["type"],
                "tier": entry["tier"],
                "tags": tags_val,
                "confidence": entry["confidence"],
                "summary": entry.get("summary"),
                "score": item["score"],
            })
            ids_to_bump.append(entry["id"])

        # Bump access counts
        if ids_to_bump:
            with sqlite3.connect(self.db_path) as conn:
                for eid in ids_to_bump:
                    conn.execute(
                        "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                        (eid,),
                    )

        return final

    # ------------------------------------------------------------------
    # Search strategies
    # ------------------------------------------------------------------

    def _fts5_search(
        self, query: str, user_id: Optional[str] = None,
        type: Optional[str] = None, limit: int = 15,
    ) -> List[Dict[str, Any]]:
        """Full-text search via the FTS5 virtual table."""
        # Build an FTS5 query: quote each token for safety
        tokens = [w for w in re.split(r"\W+", query) if len(w) >= 2]
        if not tokens:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in tokens)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            try:
                # Join FTS5 results back to the main table
                sql = """
                    SELECT m.*
                    FROM memories_fts fts
                    JOIN memories m ON m.rowid = fts.rowid
                    WHERE memories_fts MATCH ?
                """
                params: List[Any] = [fts_query]
                if user_id is not None:
                    sql += " AND m.user_id = ?"
                    params.append(user_id)
                if type is not None:
                    sql += " AND m.type = ?"
                    params.append(type)
                sql += " ORDER BY rank LIMIT ?"
                params.append(limit)

                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            except sqlite3.OperationalError:
                # FTS table might be out of sync; fall back gracefully
                return []

    def _vector_search(
        self, query_embedding: List[float], user_id: Optional[str] = None,
        type: Optional[str] = None, limit: int = 15,
    ) -> List[tuple]:
        """Brute-force cosine similarity over stored embeddings.

        Returns list of ``(entry_id, similarity, row_dict)`` tuples,
        sorted descending by similarity, filtered to sim >= 0.2.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            conditions = ["embedding IS NOT NULL"]
            params: List[Any] = []
            if user_id is not None:
                conditions.append("user_id = ?")
                params.append(user_id)
            if type is not None:
                conditions.append("type = ?")
                params.append(type)
            where = " AND ".join(conditions)
            rows = conn.execute(f"SELECT * FROM memories WHERE {where}", params).fetchall()

        scored = []
        for row in rows:
            row_dict = dict(row)
            entry_emb = _unpack_embedding(row_dict["embedding"])
            sim = _cosine_similarity(query_embedding, entry_emb)
            if sim >= 0.2:
                scored.append((row_dict["id"], sim, row_dict))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def _keyword_search(
        self, query: str, user_id: Optional[str] = None,
        type: Optional[str] = None, limit: int = 15,
    ) -> List[Dict[str, Any]]:
        """TF-IDF-inspired keyword scoring over all entries."""
        conditions: List[str] = []
        params: List[Any] = []
        if user_id is not None:
            conditions.append("user_id = ?")
            params.append(user_id)
        if type is not None:
            conditions.append("type = ?")
            params.append(type)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(f"SELECT * FROM memories {where}", params).fetchall()

        words = [w.lower() for w in re.split(r"\W+", query) if len(w) >= 2]
        if not words:
            return []

        doc_count = max(len(rows), 1)

        # Pre-compute document frequency for IDF
        word_doc_freq: Dict[str, int] = {}
        for w in words:
            count = 0
            for row in rows:
                rd = dict(row)
                text = (rd["title"] + " " + rd["content"]).lower()
                if w in text:
                    count += 1
            word_doc_freq[w] = max(count, 1)

        scored: List[tuple] = []
        for row in rows:
            rd = dict(row)
            title_lower = rd["title"].lower()
            content_lower = rd["content"].lower()
            tags_lower = rd["tags"].lower() if isinstance(rd["tags"], str) else ""
            full_text = f"{title_lower} {content_lower} {tags_lower}"
            score = 0.0

            for w in words:
                idf = math.log(doc_count / word_doc_freq.get(w, 1)) + 1

                if w in title_lower:
                    score += 3.0 * idf
                if w in tags_lower:
                    score += 2.0 * idf
                if w in content_lower:
                    tf = content_lower.count(w)
                    score += (1 + math.log(tf)) * idf if tf > 0 else 0
                elif len(w) >= 4:
                    # Fuzzy: substring containment
                    for text_word in re.split(r"\W+", full_text):
                        if len(text_word) >= 4 and (w in text_word or text_word in w):
                            score += 0.5 * idf
                            break

            if score > 0:
                scored.append((score, rd))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:limit]]

    # ------------------------------------------------------------------
    # Semantic dedup helper
    # ------------------------------------------------------------------

    def _find_similar(
        self, query_embedding: List[float], threshold: float = 0.97,
        user_id: Optional[str] = None, title: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find the first entry with cosine similarity >= *threshold*.

        When *title* is provided, candidates are pre-filtered to entries
        whose title matches case-insensitively (RF-13: prevents distinct
        user-supplied titles from being collapsed under semantic dedup).
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            conditions = ["embedding IS NOT NULL"]
            params: List[Any] = []
            if user_id is not None:
                conditions.append("user_id = ?")
                params.append(user_id)
            if title is not None:
                conditions.append("LOWER(title) = LOWER(?)")
                params.append(title)
            where = " AND ".join(conditions)
            rows = conn.execute(
                f"SELECT * FROM memories WHERE {where}", params,
            ).fetchall()

        for row in rows:
            rd = dict(row)
            entry_emb = _unpack_embedding(rd["embedding"])
            sim = _cosine_similarity(query_embedding, entry_emb)
            if sim >= threshold:
                return rd
        return None

    # ------------------------------------------------------------------
    # Auto-learn -- extraction cascade
    # ------------------------------------------------------------------

    def auto_learn(
        self, text: str, context_type: str = "general", max_entries: int = 3,
    ) -> Dict[str, Any]:
        """Extract and store learnings from text.

        Tries extraction methods in order:

        1. **Ollama** -- local LLM, fully offline
        2. **API** -- Anthropic or OpenAI (requires key in env)
        3. **Heuristic** -- regex patterns (always works)

        Args:
            text: Raw text to extract knowledge from.
            context_type: Hint for extraction context (unused by heuristic).
            max_entries: Maximum entries to extract. Defaults to 3.

        Returns:
            Dict with ``extracted`` count, ``source`` name, and ``entries`` list.
        """
        entries = _ollama_extract(text, max_entries)
        source = "ollama"

        if entries is None:
            entries = _api_extract(text, max_entries)
            source = "api"

        if entries is None:
            entries = _heuristic_extract(text, max_entries)
            source = "heuristic"

        stored: List[Dict[str, Any]] = []
        for entry in entries or []:
            result = self.store(
                content=entry.get("content", ""),
                title=entry.get("title", ""),
                type=entry.get("type", "PATTERN"),
                tier="episodic",
                confidence=0.6 if source in ("ollama", "api") else 0.4,
            )
            stored.append(result)

        return {"extracted": len(stored), "source": source, "entries": stored}

    # ------------------------------------------------------------------
    # Compatibility aliases (Mem0-style API)
    # ------------------------------------------------------------------

    def add(
        self,
        data: str,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """Mem0-compatible store. Auto-detects title and type.

        Args:
            data: The knowledge content to store.
            user_id: Optional user ID for scoping.
            agent_id: Ignored (API compatibility).
            metadata: Ignored (API compatibility).

        Returns:
            Dict with ``id`` and ``deduplicated`` keys.
        """
        return self.store(
            content=data,
            title=_auto_title(data),
            type=_auto_type(data),
            tier="semantic",
            user_id=user_id,
        )

    def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Mem0-compatible alias for :meth:`recall`.

        Args:
            query: Natural-language search query.
            user_id: Optional user ID for scoping.
            limit: Maximum results. Defaults to 5.

        Returns:
            List of matching entries with scores.
        """
        return self.recall(query, limit=limit, user_id=user_id)

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def forget(self, memory_id: str) -> None:
        """Delete a memory by ID.

        Args:
            memory_id: The UUID of the memory to delete.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    # ------------------------------------------------------------------
    # Health / stats
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Return memory statistics.

        Returns:
            Dict with ``total_entries``, ``by_tier``, ``by_type``,
            ``total_tokens`` (approximate), ``has_embeddings`` count,
            and ``search_methods`` list.
        """
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

            by_tier: Dict[str, int] = {}
            for row in conn.execute("SELECT tier, COUNT(*) FROM memories GROUP BY tier"):
                by_tier[row[0]] = row[1]

            by_type: Dict[str, int] = {}
            for row in conn.execute("SELECT type, COUNT(*) FROM memories GROUP BY type"):
                by_type[row[0]] = row[1]

            total_chars = conn.execute(
                "SELECT COALESCE(SUM(LENGTH(content) + LENGTH(title)), 0) FROM memories"
            ).fetchone()[0]
            total_tokens = total_chars // 4

            has_embeddings = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
            ).fetchone()[0]

        search_methods = ["fts5", "keyword"]
        if _get_embedder() is not None:
            search_methods.append("vector")

        return {
            "total_entries": total,
            "by_tier": by_tier,
            "by_type": by_type,
            "total_tokens": total_tokens,
            "has_embeddings": has_embeddings,
            "search_methods": search_methods,
        }

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self) -> List[Dict[str, Any]]:
        """Export all memories as a list of dicts (excluding raw embeddings).

        Returns:
            List of all memory entries ordered by creation time.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM memories ORDER BY created_at"
            ).fetchall()

        return [
            {
                "id": r["id"],
                "title": r["title"],
                "content": r["content"],
                "type": r["type"],
                "tier": r["tier"],
                "tags": json.loads(r["tags"]) if isinstance(r["tags"], str) else r["tags"],
                "user_id": r["user_id"],
                "confidence": r["confidence"],
                "summary": r["summary"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "access_count": r["access_count"],
                "helpful_count": r["helpful_count"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Upgrade to cloud
    # ------------------------------------------------------------------

    def upgrade(
        self, api_key: str, project_id: Optional[str] = None,
    ) -> Any:
        """Migrate local memories to cloud AgentBay.

        Reads all local entries, stores each one in the cloud, and
        returns a fully initialised cloud client.  Embedding metadata
        (has_embedding flag and dimension count) is passed as tags so
        the cloud can re-embed if desired.

        Args:
            api_key: Your AgentBay API key.
            project_id: Cloud project ID to migrate into.

        Returns:
            A cloud :class:`AgentBay` instance with all local memories migrated.
        """
        from .client import AgentBay

        cloud = AgentBay(api_key=api_key, project_id=project_id)
        entries = self.export()

        for entry in entries:
            entry_tags = list(entry.get("tags") or [])
            if entry.get("user_id"):
                entry_tags.append(f"user:{entry['user_id']}")
            # Mark entries that had local embeddings
            entry_tags.append("migrated:local")

            cloud.store(
                content=entry["content"],
                title=entry["title"],
                type=entry["type"],
                tier=entry["tier"],
                tags=entry_tags if entry_tags else None,
            )

        return cloud

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        stats = self.health()
        methods = ", ".join(stats.get("search_methods", []))
        return (
            f"LocalMemory(db='{self.db_path}', "
            f"entries={stats['total_entries']}, "
            f"search=[{methods}])"
        )
