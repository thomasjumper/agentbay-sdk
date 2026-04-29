"""AgentBay integration for Dify.

Provides an external knowledge base API that Dify can connect to
for persistent memory retrieval across conversations.

Dify connects via HTTP API. Deploy agentbay.server or use the cloud API.

Setup in Dify:
1. Go to Knowledge → External Knowledge API
2. Add URL: https://www.aiagentsbay.com/api/v1/projects/{project_id}/memory
3. Set Authorization header: Bearer ab_live_your_key
4. Map query parameter to the search field

Or use this helper to create a Dify-compatible endpoint::

    from agentbay.integrations.dify import DifyMemoryAPI
    api = DifyMemoryAPI("ab_live_your_key", project_id="your-project")
    # Returns config for Dify's external knowledge API
    print(api.dify_config())
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class DifyMemoryAPI:
    """Adapter for Dify's External Knowledge Base API.

    Provides methods that map to Dify's expected request/response format.

    Args:
        api_key: AgentBay API key.
        project_id: Project ID for scoping.
        base_url: AgentBay API URL.
    """

    def __init__(
        self,
        api_key: str,
        project_id: str,
        base_url: str = "https://www.aiagentsbay.com",
    ):
        from agentbay import AgentBay
        self._brain = AgentBay(api_key=api_key, project_id=project_id, base_url=base_url)
        self._project_id = project_id
        self._base_url = base_url
        self._api_key = api_key

    def dify_config(self) -> Dict[str, Any]:
        """Returns configuration for Dify's External Knowledge API setup.

        Paste this into Dify's External Knowledge API settings.
        """
        return {
            "name": "AgentBay Memory",
            "description": "Persistent memory across conversations — patterns, pitfalls, decisions, architecture.",
            "api_endpoint": f"{self._base_url}/api/v1/projects/{self._project_id}/memory",
            "api_key": self._api_key,
            "method": "GET",
            "headers": {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            "query_param": "q",
            "limit_param": "limit",
            "response_path": "entries",
            "content_field": "content",
            "title_field": "title",
            "score_field": "confidence",
        }

    def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve memories in Dify's expected response format.

        Returns list of dicts with: content, title, score, metadata.
        """
        results = self._brain.recall(query, limit=limit)
        return [
            {
                "content": entry.get("content", ""),
                "title": entry.get("title", ""),
                "score": entry.get("confidence", 0),
                "metadata": {
                    "type": entry.get("type", "PATTERN"),
                    "tags": entry.get("tags", []),
                    "source": "agentbay",
                },
            }
            for entry in results
        ]

    def store(self, content: str, title: str = "", type: str = "PATTERN", tags: Optional[List[str]] = None) -> Dict:
        """Store a memory from a Dify workflow."""
        return self._brain.store(
            content,
            title=title or content[:80],
            type=type,
            tags=(tags or []) + ["dify"],
        )
