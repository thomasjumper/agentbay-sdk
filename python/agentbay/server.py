"""AgentBay Local Server — self-hosted REST API backed by SQLite.

Run directly:
    python -m agentbay.server

Run via Docker:
    docker compose up

Exposes the same API as the cloud at http://localhost:8787
"""

import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from .local import LocalMemory

DB_PATH = os.environ.get("AGENTBAY_DB_PATH", None)
HOST = os.environ.get("AGENTBAY_HOST", "0.0.0.0")
PORT = int(os.environ.get("AGENTBAY_PORT", "8787"))
# API key auth: set AGENTBAY_SERVER_KEY to require Bearer token auth
# If not set, server runs without auth (local development only)
SERVER_KEY = os.environ.get("AGENTBAY_SERVER_KEY", "")

memory = LocalMemory(db_path=DB_PATH)


class AgentBayHandler(BaseHTTPRequestHandler):
    """Minimal REST API matching the cloud endpoints."""

    def _check_auth(self) -> bool:
        """Check Bearer token if AGENTBAY_SERVER_KEY is set."""
        if not SERVER_KEY:
            return True  # No auth configured
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {SERVER_KEY}":
            return True
        self._json_response({"error": "Unauthorized. Set Authorization: Bearer <AGENTBAY_SERVER_KEY>"}, 401)
        return False

    def _json_response(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_OPTIONS(self):
        self._json_response({})

    def do_GET(self):
        if not self._check_auth():
            return

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        if parsed.path == "/health" or parsed.path == "/api/v1/health":
            stats = memory.health()
            self._json_response({
                "status": "ok",
                "mode": "local",
                "metrics": stats,
            })
            return

        # Memory recall: GET /memory?q=...
        if parsed.path in ("/memory", "/api/v1/memory", "/recall"):
            query = params.get("q", [""])[0]
            if not query:
                self._json_response({"error": "q parameter required"}, 400)
                return
            limit = int(params.get("limit", ["5"])[0])
            user_id = params.get("user_id", [None])[0]
            type_filter = params.get("type", [None])[0]
            results = memory.recall(query, limit=limit, user_id=user_id, type=type_filter)
            self._json_response({
                "entries": results,
                "totalTokens": sum(len(r.get("content", "")) // 4 for r in results),
                "searchMode": "local(keyword+tfidf+fuzzy)",
            })
            return

        # Memory health: GET /memory/health
        if parsed.path in ("/memory/health", "/api/v1/memory/health"):
            self._json_response(memory.health())
            return

        # Export: GET /memory/export
        if parsed.path in ("/memory/export", "/api/v1/memory/export"):
            self._json_response({"entries": memory.export()})
            return

        self._json_response({"error": "Not found", "endpoints": [
            "GET /health",
            "GET /memory?q=...",
            "POST /memory",
            "DELETE /memory",
            "GET /memory/health",
            "GET /memory/export",
        ]}, 404)

    def do_POST(self):
        if not self._check_auth():
            return

        parsed = urlparse(self.path)
        body = self._read_body()

        # Memory store: POST /memory
        if parsed.path in ("/memory", "/api/v1/memory"):
            content = body.get("content", "")
            if not content:
                self._json_response({"error": "content required"}, 400)
                return
            result = memory.store(
                content=content,
                title=body.get("title"),
                type=body.get("type", "PATTERN"),
                tier=body.get("tier", "semantic"),
                tags=body.get("tags"),
                user_id=body.get("user_id"),
                confidence=body.get("confidence", 0.5),
            )
            self._json_response(result, 201)
            return

        # Mem0-compatible: POST /add
        if parsed.path in ("/add", "/api/v1/add"):
            data = body.get("data", body.get("content", ""))
            if not data:
                self._json_response({"error": "data required"}, 400)
                return
            result = memory.add(data, user_id=body.get("user_id"))
            self._json_response(result, 201)
            return

        # Mem0-compatible: POST /search
        if parsed.path in ("/search", "/api/v1/search"):
            query = body.get("query", "")
            if not query:
                self._json_response({"error": "query required"}, 400)
                return
            results = memory.search(query, user_id=body.get("user_id"), limit=body.get("limit", 5))
            self._json_response({"results": results})
            return

        self._json_response({"error": "Not found"}, 404)

    def do_DELETE(self):
        if not self._check_auth():
            return

        parsed = urlparse(self.path)
        body = self._read_body()

        if parsed.path in ("/memory", "/api/v1/memory"):
            memory_id = body.get("id", body.get("knowledgeId", ""))
            if not memory_id:
                self._json_response({"error": "id required"}, 400)
                return
            memory.forget(memory_id)
            self._json_response({"deleted": True})
            return

        self._json_response({"error": "Not found"}, 404)

    def log_message(self, format, *args):
        # Cleaner logging
        print(f"[agentbay-local] {args[0]}")


def main():
    print(f"""
    ╔══════════════════════════════════════╗
    ║       AgentBay Local Server          ║
    ║                                      ║
    ║  http://{HOST}:{PORT}               ║
    ║  SQLite: {memory.db_path}           ║
    ║                                      ║
    ║  Endpoints:                          ║
    ║    GET  /health                      ║
    ║    GET  /memory?q=...               ║
    ║    POST /memory                      ║
    ║    POST /add    (Mem0 compatible)    ║
    ║    POST /search (Mem0 compatible)    ║
    ╚══════════════════════════════════════╝
    """)
    server = HTTPServer((HOST, PORT), AgentBayHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[agentbay-local] Shutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
