"""AgentBay integration for Flowise.

Provides a custom tool node that Flowise can use to add persistent
memory to any chatflow or agentflow.

Setup in Flowise:
1. Add a "Custom Tool" node to your flow
2. Set the tool function to call AgentBay's REST API
3. Or use this helper to generate the Flowise tool definition

Example — Flowise Custom Tool (JavaScript)::

    // In Flowise Custom Tool node:
    const response = await fetch(
        'https://www.aiagentsbay.com/api/v1/projects/YOUR_PROJECT/memory?q=' +
        encodeURIComponent(input) + '&limit=3',
        { headers: { 'Authorization': 'Bearer ab_live_YOUR_KEY' } }
    );
    const data = await response.json();
    return data.entries.map(e => e.title + ': ' + e.content).join('\\n');
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class FlowiseMemoryTool:
    """Memory tool for Flowise chatflows and agentflows.

    Generates Flowise-compatible tool definitions and provides
    a Python adapter for Flowise's custom tool API.

    Args:
        api_key: AgentBay API key.
        project_id: Project ID.
        base_url: API URL.
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

    def flowise_tool_config(self) -> Dict[str, Any]:
        """Returns configuration for Flowise's Custom Tool node.

        Copy this into Flowise's tool definition.
        """
        return {
            "name": "agentbay_memory",
            "description": "Search persistent memory for relevant patterns, pitfalls, decisions, and context from previous conversations.",
            "type": "function",
            "function": {
                "name": "search_memory",
                "description": "Search AgentBay persistent memory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for in memory",
                        },
                    },
                    "required": ["query"],
                },
            },
            "api_config": {
                "url": f"{self._base_url}/api/v1/projects/{self._project_id}/memory",
                "method": "GET",
                "headers": {
                    "Authorization": f"Bearer {self._api_key}",
                },
                "query_mapping": {
                    "q": "{{query}}",
                    "limit": "3",
                },
            },
        }

    def flowise_js_snippet(self) -> str:
        """Returns JavaScript code for Flowise Custom Tool node."""
        return f"""// AgentBay Memory Search — paste into Flowise Custom Tool
const API_KEY = '{self._api_key}';
const PROJECT = '{self._project_id}';
const BASE = '{self._base_url}';

async function searchMemory(query) {{
  const resp = await fetch(
    `${{BASE}}/api/v1/projects/${{PROJECT}}/memory?q=${{encodeURIComponent(query)}}&limit=3`,
    {{ headers: {{ 'Authorization': `Bearer ${{API_KEY}}` }} }}
  );
  const data = await resp.json();
  if (!data.entries || !data.entries.length) return 'No relevant memories found.';
  return data.entries.map(e => `[${{e.type}}] ${{e.title}}: ${{e.content.substring(0, 200)}}`).join('\\n');
}}

// Flowise passes the input as the first argument
return await searchMemory($input);"""

    def search(self, query: str, limit: int = 3) -> List[Dict]:
        """Search memories (Python adapter)."""
        return self._brain.recall(query, limit=limit)

    def store(self, content: str, title: str = "", type: str = "PATTERN") -> Dict:
        """Store a memory (Python adapter)."""
        return self._brain.store(content, title=title or content[:80], type=type, tags=["flowise"])
