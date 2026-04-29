"""AgentBay memory for Pipecat pipelines.

Provides a pipeline processor that injects memories before LLM frames
and captures responses after, giving Pipecat bots persistent memory.

Usage::

    pip install agentbay[pipecat]

    from agentbay.integrations.pipecat import AgentBayProcessor

    processor = AgentBayProcessor(api_key="ab_live_...", project_id="...")

    # Add to your Pipecat pipeline
    pipeline = Pipeline([transport, processor, llm, tts])
"""

from __future__ import annotations

from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional Pipecat base class
# ---------------------------------------------------------------------------

try:
    from pipecat.processors.frame_processor import FrameProcessor

    _HAS_PIPECAT = True
except ImportError:
    FrameProcessor = object  # type: ignore[assignment,misc]
    _HAS_PIPECAT = False


class AgentBayProcessor(FrameProcessor):
    """Pipecat pipeline processor backed by AgentBay's Knowledge Brain.

    Intercepts user transcript frames to recall relevant memories and
    injects them as context. Captures LLM response frames to store learnings.

    Args:
        api_key: Your AgentBay API key.
        project_id: The Knowledge Brain project ID.
        base_url: API base URL. Defaults to ``https://www.aiagentsbay.com``.
        recall_limit: Max memories to recall per frame (default: 3).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        base_url: str = "https://www.aiagentsbay.com",
        recall_limit: int = 3,
        **kwargs: Any,
    ) -> None:
        if not _HAS_PIPECAT:
            raise ImportError(
                "pipecat-ai is required for AgentBayProcessor. "
                "Install it with: pip install agentbay[pipecat]"
            )
        super().__init__(**kwargs)

        from agentbay.client import AgentBay

        self._client = AgentBay(
            api_key=api_key,
            base_url=base_url,
            project_id=project_id,
        )
        self._recall_limit = recall_limit
        self._last_user_text: str = ""

    async def process_frame(self, frame: Any, direction: Any = None) -> None:
        """Process a pipeline frame, injecting or capturing memories.

        Handles TranscriptionFrame (user speech) by recalling relevant
        memories and TextFrame (LLM output) by storing them.

        Args:
            frame: The Pipecat frame to process.
            direction: Frame direction in the pipeline.
        """
        frame_type = type(frame).__name__

        if frame_type == "TranscriptionFrame":
            text = getattr(frame, "text", "")
            self._last_user_text = text
            if text:
                try:
                    results = self._client.recall(query=text, limit=self._recall_limit)
                    if results:
                        context = "\n".join(
                            r.get("content", r.get("text", ""))
                            for r in results
                        )
                        # Attach context to frame metadata if possible
                        if hasattr(frame, "metadata"):
                            frame.metadata = frame.metadata or {}
                            frame.metadata["agentbay_context"] = context
                except Exception:
                    pass

        elif frame_type in ("TextFrame", "LLMResponseEndFrame"):
            text = getattr(frame, "text", "")
            if text and self._last_user_text:
                try:
                    self._client.store(
                        content=f"User: {self._last_user_text}\nBot: {text}",
                        tags=["source:pipecat"],
                    )
                except Exception:
                    pass

        # Pass frame downstream
        if direction is not None:
            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame)

    def __repr__(self) -> str:
        return "AgentBayProcessor()"
