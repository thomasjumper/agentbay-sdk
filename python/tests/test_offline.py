"""Tests for offline projects and teams."""

import os
import tempfile
import pytest
from agentbay.offline import OfflineProject, OfflineTeam


@pytest.fixture
def proj():
    with tempfile.TemporaryDirectory() as td:
        yield OfflineProject("test-project", db_dir=td)


@pytest.fixture
def team():
    with tempfile.TemporaryDirectory() as td:
        yield OfflineTeam("test-team", db_dir=td)


class TestOfflineProject:
    def test_create_project(self, proj):
        assert proj is not None

    def test_store_and_recall(self, proj):
        proj.store("Auth uses JWT", title="Auth pattern")
        results = proj.recall("authentication")
        assert len(results) >= 1

    def test_create_task(self, proj):
        task_id = proj.create_task("Fix auth bug", description="Session expiry issue", priority="high")
        assert task_id

    def test_list_tasks(self, proj):
        proj.create_task("Task 1")
        proj.create_task("Task 2")
        tasks = proj.list_tasks()
        assert len(tasks) >= 2

    def test_claim_task(self, proj):
        result = proj.create_task("Claimable task")
        task_id = result if isinstance(result, str) else result.get("id", result)
        proj.claim_task(str(task_id), "agent-1")
        # Verify claim worked by listing all tasks
        all_tasks = proj.list_tasks()
        claimed = [t for t in all_tasks if t.get("assigned_to") == "agent-1"]
        assert len(claimed) >= 1

    def test_handoff(self, proj):
        proj.handoff(
            summary="Fixed auth bug",
            completed_steps=["Identified root cause", "Applied fix"],
            blockers=["Need review"],
            next_steps=["Deploy to staging"],
            files_modified=["src/auth.ts"],
            from_agent="agent-1",
        )
        # Should not crash

    def test_resume(self, proj):
        proj.handoff(summary="Some work done", from_agent="agent-1")
        context = proj.resume()
        assert context is not None

    def test_health(self, proj):
        h = proj.health()
        assert "memory" in h or "total_entries" in h or isinstance(h, dict)


class TestOfflineTeam:
    def test_create_team(self, team):
        assert team is not None

    def test_store_with_agent(self, team):
        team.store("Pattern from Claude", agent_name="claude")
        team.store("Pattern from Moonsa", agent_name="moonsa")
        members = team.members()
        assert "claude" in members
        assert "moonsa" in members

    def test_recall_from_team(self, team):
        team.store("JWT pattern", agent_name="claude")
        results = team.recall("JWT")
        assert len(results) >= 1

    def test_recall_from_specific_agent(self, team):
        team.store("Claude's pattern", agent_name="claude")
        team.store("Moonsa's pattern", agent_name="moonsa")
        results = team.recall_from("pattern", agent_name="claude")
        assert len(results) >= 1

    def test_health(self, team):
        h = team.health()
        assert isinstance(h, dict)
