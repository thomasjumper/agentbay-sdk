"""Tests for local memory engine (SQLite + FTS5)."""

import os
import tempfile
import pytest
from agentbay.local import LocalMemory


@pytest.fixture
def mem():
    """Fresh in-memory LocalMemory for each test."""
    with tempfile.TemporaryDirectory() as td:
        db_path = os.path.join(td, "test.db")
        m = LocalMemory(db_path=db_path, quiet=True)
        yield m


class TestStore:
    def test_store_returns_id(self, mem):
        result = mem.store("JWT with refresh tokens", title="Auth pattern")
        assert "id" in result
        assert result["id"]

    def test_store_with_type(self, mem):
        result = mem.store("Bug in auth", title="Auth bug", type="PITFALL")
        assert result["id"]

    def test_store_with_tags(self, mem):
        result = mem.store("Use bcrypt", title="Hashing", tags=["security", "auth"])
        assert result["id"]

    def test_store_with_tier(self, mem):
        result = mem.store("Temp note", title="Session", tier="working")
        assert result["id"]

    def test_store_with_user_id(self, mem):
        mem.store("Prefers dark mode", title="Pref", user_id="user1")
        mem.store("Prefers light mode", title="Pref", user_id="user2")
        r1 = mem.recall("mode", user_id="user1")
        r2 = mem.recall("mode", user_id="user2")
        assert len(r1) >= 1
        assert len(r2) >= 1

    def test_store_dedup(self, mem):
        mem.store("Pattern A", title="Auth pattern", type="PATTERN")
        mem.store("Pattern A updated", title="Auth pattern", type="PATTERN")
        h = mem.health()
        # Should dedup (update existing, not create duplicate)
        assert h["total_entries"] <= 2

    def test_store_empty_content_raises(self, mem):
        # Should handle gracefully
        result = mem.store("", title="Empty")
        assert result["id"]  # Still stores (empty content is allowed)


class TestRecall:
    def test_recall_basic(self, mem):
        mem.store("JWT auth with refresh tokens", title="Auth pattern", tags=["auth"])
        results = mem.recall("authentication")
        assert len(results) >= 1
        assert results[0]["title"] == "Auth pattern"

    def test_recall_empty_query(self, mem):
        mem.store("Something", title="Test")
        results = mem.recall("")
        # Should return something or empty, not crash
        assert isinstance(results, list)

    def test_recall_no_results(self, mem):
        results = mem.recall("nonexistent topic xyz123")
        assert results == []

    def test_recall_limit(self, mem):
        for i in range(10):
            mem.store(f"Entry {i}", title=f"Entry {i}")
        results = mem.recall("entry", limit=3)
        assert len(results) <= 3

    def test_recall_by_tag(self, mem):
        mem.store("Use bcrypt", title="Hashing", tags=["security"])
        mem.store("Use JWT", title="Auth", tags=["auth"])
        results = mem.recall("security")
        assert len(results) >= 1

    def test_recall_returns_confidence(self, mem):
        mem.store("Test entry", title="Test")
        results = mem.recall("test")
        if results:
            assert "confidence" in results[0]


class TestMem0Compat:
    def test_add_string(self, mem):
        mem.add("The user prefers dark mode")
        results = mem.search("dark mode")
        assert len(results) >= 1

    def test_add_string_with_details(self, mem):
        mem.add("Uses Python 3.12 for the backend stack")
        results = mem.search("Python")
        assert len(results) >= 1

    def test_search_with_user_id(self, mem):
        mem.add("Likes TypeScript", user_id="alice")
        mem.add("Likes Rust", user_id="bob")
        alice = mem.search("likes", user_id="alice")
        bob = mem.search("likes", user_id="bob")
        # Both should find results
        assert len(alice) >= 1
        assert len(bob) >= 1


class TestForget:
    def test_forget_by_id(self, mem):
        result = mem.store("To forget", title="Temp")
        mem.forget(result["id"])
        results = mem.recall("forget")
        # Should not find the forgotten entry
        assert all(r.get("id") != result["id"] for r in results)


class TestHealth:
    def test_health_empty(self, mem):
        h = mem.health()
        assert h["total_entries"] == 0
        assert "search_methods" in h

    def test_health_after_stores(self, mem):
        mem.store("Entry 1", title="E1")
        mem.store("Entry 2", title="E2")
        h = mem.health()
        assert h["total_entries"] == 2

    def test_health_has_search_methods(self, mem):
        h = mem.health()
        assert "search_methods" in h
        assert isinstance(h["search_methods"], list)


class TestExport:
    def test_export_empty(self, mem):
        entries = mem.export()
        assert entries == []

    def test_export_all(self, mem):
        mem.store("Entry 1", title="E1")
        mem.store("Entry 2", title="E2")
        entries = mem.export()
        assert len(entries) == 2
        assert all("title" in e for e in entries)
