"""AgentBay integration for Raycast.

Provides a simple HTTP API that Raycast AI extensions can call
for persistent memory search and storage.

Raycast connects via its Script Command or AI Extension system.
This module provides the handler functions.

Usage — Raycast Script Command::

    #!/bin/bash
    # @raycast.schemaVersion 1
    # @raycast.title Search Memory
    # @raycast.mode inline
    # @raycast.argument1 { "type": "text", "placeholder": "query" }

    curl -s "https://www.aiagentsbay.com/api/v1/projects/YOUR_PROJECT/memory?q=$1&limit=3" \\
      -H "Authorization: Bearer ab_live_YOUR_KEY" | python3 -c "
    import sys,json
    for e in json.load(sys.stdin).get('entries',[]):
        print(f'[{e[\"type\"]}] {e[\"title\"]}: {e[\"content\"][:100]}')
    "

Usage — Python helper::

    from agentbay.integrations.raycast import RaycastMemory

    memory = RaycastMemory("ab_live_your_key", project_id="your-project")
    print(memory.generate_script_commands())
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class RaycastMemory:
    """AgentBay memory access for Raycast.

    Generates Raycast Script Commands and provides Python helpers
    for building Raycast AI extensions with persistent memory.

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
        self._api_key = api_key
        self._project_id = project_id
        self._base_url = base_url

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        """Search memories."""
        return self._brain.recall(query, limit=limit)

    def store(self, content: str, title: str = "", type: str = "PATTERN") -> Dict:
        """Store a memory."""
        return self._brain.store(content, title=title or content[:80], type=type, tags=["raycast"])

    def format_results(self, results: List[Dict]) -> str:
        """Format results for Raycast inline display."""
        if not results:
            return "No memories found."
        lines = []
        for e in results:
            conf = int(e.get("confidence", 0) * 100)
            lines.append(f"[{e.get('type','?')}] {e.get('title','?')} ({conf}%)")
            content = e.get("content", "")[:150]
            if content:
                lines.append(f"  {content}")
        return "\n".join(lines)

    def generate_script_commands(self) -> str:
        """Generate Raycast Script Commands for memory operations.

        Returns bash scripts ready to save as Raycast commands.
        """
        search_script = f"""#!/bin/bash
# @raycast.schemaVersion 1
# @raycast.title Search AgentBay Memory
# @raycast.mode inline
# @raycast.icon 🧠
# @raycast.argument1 {{ "type": "text", "placeholder": "search query" }}

curl -s "{self._base_url}/api/v1/projects/{self._project_id}/memory?q=$1&limit=3" \\
  -H "Authorization: Bearer {self._api_key}" | python3 -c "
import sys,json
for e in json.load(sys.stdin).get('entries',[]):
    print(f'[{{e[\"type\"]}}] {{e[\"title\"]}}: {{e[\"content\"][:100]}}')
"
"""

        store_script = f"""#!/bin/bash
# @raycast.schemaVersion 1
# @raycast.title Store to AgentBay Memory
# @raycast.mode silent
# @raycast.icon 🧠
# @raycast.argument1 {{ "type": "text", "placeholder": "what to remember" }}

curl -s -X POST "{self._base_url}/api/v1/projects/{self._project_id}/memory" \\
  -H "Authorization: Bearer {self._api_key}" \\
  -H "Content-Type: application/json" \\
  -d "{{\\\"action\\\":\\\"store\\\",\\\"title\\\":\\\"Raycast note\\\",\\\"content\\\":\\\"$1\\\",\\\"type\\\":\\\"CONTEXT\\\",\\\"tags\\\":[\\\"raycast\\\"]}}" > /dev/null

echo "Stored to AgentBay memory"
"""

        return f"=== search-memory.sh ===\n{search_script}\n\n=== store-memory.sh ===\n{store_script}"
