"""AgentBay integration for AWS Bedrock Agents.

Adds persistent memory to AWS Bedrock agent action groups and
knowledge bases. Works as an external data source that Bedrock
queries for context.

Usage::

    from agentbay.integrations.aws_bedrock import AgentBayBedrockMemory

    memory = AgentBayBedrockMemory("ab_live_your_key", project_id="your-project")

    # As a Lambda function for Bedrock action group:
    def handler(event, context):
        return memory.handle_action_group(event)

    # Or as a pre-processing step:
    context = memory.get_context("authentication patterns")
    # → inject into Bedrock agent's system prompt
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


class AgentBayBedrockMemory:
    """Persistent memory for AWS Bedrock agents.

    Can be used as:
    1. Action group handler (Lambda function returns memory results)
    2. Knowledge base supplement (pre-fetch context for system prompt)
    3. Post-invocation learner (store agent outputs)

    Args:
        api_key: AgentBay API key.
        project_id: Project ID.
        base_url: API URL.
    """

    def __init__(
        self,
        api_key: str,
        project_id: str | None = None,
        base_url: str = "https://www.aiagentsbay.com",
    ):
        from agentbay import AgentBay
        self._brain = AgentBay(api_key=api_key, project_id=project_id, base_url=base_url)

    def handle_action_group(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a Bedrock agent action group invocation.

        Deploy this as a Lambda function and configure it as an
        action group in your Bedrock agent.

        Supports actions: SearchMemory, StoreMemory, GetHealth
        """
        action = event.get("actionGroup", "")
        api_path = event.get("apiPath", "")
        parameters = {p["name"]: p["value"] for p in event.get("parameters", [])}

        if api_path == "/search" or action == "SearchMemory":
            query = parameters.get("query", "")
            limit = int(parameters.get("limit", "5"))
            results = self._brain.recall(query, limit=limit)
            body = json.dumps([{
                "title": e.get("title", ""),
                "content": e.get("content", ""),
                "type": e.get("type", ""),
                "confidence": e.get("confidence", 0),
            } for e in results])

        elif api_path == "/store" or action == "StoreMemory":
            content = parameters.get("content", "")
            title = parameters.get("title", content[:80])
            mem_type = parameters.get("type", "PATTERN")
            result = self._brain.store(content, title=title, type=mem_type, tags=["bedrock"])
            body = json.dumps({"id": result.get("id", ""), "stored": True})

        elif api_path == "/health" or action == "GetHealth":
            health = self._brain.health()
            body = json.dumps(health)

        else:
            body = json.dumps({"error": f"Unknown action: {action}"})

        return {
            "messageVersion": "1.0",
            "response": {
                "actionGroup": action,
                "apiPath": api_path,
                "httpMethod": event.get("httpMethod", "GET"),
                "httpStatusCode": 200,
                "responseBody": {"application/json": {"body": body}},
            },
        }

    def get_context(self, query: str, limit: int = 3) -> str:
        """Get formatted context for injection into Bedrock system prompt.

        Call before invoking the Bedrock agent to pre-load relevant memories.
        """
        results = self._brain.recall(query, limit=limit)
        if not results:
            return ""
        lines = [f"[AgentBay — {len(results)} memories]"]
        for e in results:
            lines.append(f"[{e.get('type','?')}] {e.get('title','?')}: {e.get('content','')[:200]}")
        return "\n".join(lines)

    def store_output(self, output: str, type: str = "PATTERN", tags: Optional[List[str]] = None) -> Optional[Dict]:
        """Store agent output as a memory. Call after agent responds."""
        if len(output) < 50:
            return None
        title = output.split("\n")[0][:100] if "\n" in output else output[:100]
        return self._brain.store(output[:2000], title=title, type=type, tags=(tags or []) + ["bedrock"])

    def openapi_schema(self) -> Dict[str, Any]:
        """Returns OpenAPI schema for Bedrock action group configuration."""
        return {
            "openapi": "3.0.0",
            "info": {"title": "AgentBay Memory", "version": "1.0.0"},
            "paths": {
                "/search": {
                    "get": {
                        "summary": "Search persistent memory",
                        "operationId": "SearchMemory",
                        "parameters": [
                            {"name": "query", "in": "query", "required": True, "schema": {"type": "string"}},
                            {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 5}},
                        ],
                    }
                },
                "/store": {
                    "post": {
                        "summary": "Store a memory",
                        "operationId": "StoreMemory",
                        "requestBody": {"content": {"application/json": {"schema": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "title": {"type": "string"},
                                "type": {"type": "string", "enum": ["PATTERN","PITFALL","DECISION","ARCHITECTURE","CONTEXT"]},
                            },
                            "required": ["content"],
                        }}}},
                    }
                },
            },
        }
