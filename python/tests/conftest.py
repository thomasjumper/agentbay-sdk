"""Shared test configuration."""

import os
import sys

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Suppress welcome messages during tests
os.environ["AGENTBAY_QUIET"] = "1"
