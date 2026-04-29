"""AgentBay - Persistent memory for AI agents."""

from agentbay.client import AgentBay, AgentBayError, AuthenticationError, NotFoundError, RateLimitError
from agentbay.local import LocalMemory
from agentbay.offline import OfflineProject, OfflineTeam
from agentbay.sync import SyncEngine
from agentbay.telemetry import (
    enable_error_reporting,
    disable_error_reporting,
    get_telemetry_status,
    is_error_reporting_enabled,
    report_exception,
    error_reporting_decorator,
)

__version__ = "1.5.2"
__all__ = [
    "AgentBay",
    "AgentBayError",
    "AuthenticationError",
    "NotFoundError",
    "RateLimitError",
    "LocalMemory",
    "OfflineProject",
    "OfflineTeam",
    "SyncEngine",
    # Telemetry helpers (opt-in; defaults to disabled)
    "enable_error_reporting",
    "disable_error_reporting",
    "get_telemetry_status",
    "is_error_reporting_enabled",
    "report_exception",
    "error_reporting_decorator",
]
