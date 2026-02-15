"""Tests for Entry IDs and timestamps."""

import pytest
from datetime import datetime, timezone

from loom import StringEntry, FileEntry, Context, reset_generator


class TestEntryID:
    def setup_method(self):
        # Reset generator for reproducible tests (4-char entry IDs)
        reset_generator(seed=42, length=4)

    def test_entry_has_id(self):
        entry = StringEntry("Hello")
        assert hasattr(entry, "id")
        assert len(entry.id) == 4

    def test_entry_has_created_at(self):
        before = datetime.now(timezone.utc)
        entry = StringEntry("Hello")
        after = datetime.now(timezone.utc)
        
        assert hasattr(entry, "created_at")
        assert before <= entry.created_at <= after

    def test_each_entry_unique_id(self):
        entries = [StringEntry(f"Entry {i}") for i in range(100)]
        ids = [e.id for e in entries]
        
        assert len(ids) == len(set(ids))

    def test_file_entry_has_id(self, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        entry = FileEntry(test_file)
        assert len(entry.id) == 4
        assert entry.created_at is not None


class TestContextEntries:
    def setup_method(self):
        reset_generator(seed=42, length=4)

    def test_entries_iterates_all(self):
        ctx = Context("test")
        ctx.foundation.add("Foundation")
        ctx.convo.add("Convo 1")
        ctx.convo.add("Convo 2")
        ctx.step.add("Step")
        
        all_entries = list(ctx.entries())
        assert len(all_entries) == 4

    def test_entries_filtered_by_section(self):
        ctx = Context("test")
        ctx.foundation.add("Foundation")
        ctx.convo.add("Convo 1")
        ctx.convo.add("Convo 2")
        
        convo_entries = list(ctx.entries("convo"))
        assert len(convo_entries) == 2
        
        foundation_entries = list(ctx.entries("foundation"))
        assert len(foundation_entries) == 1

    def test_entries_section_case_insensitive(self):
        ctx = Context("test")
        ctx.convo.add("Test")
        
        assert len(list(ctx.entries("CONVO"))) == 1
        assert len(list(ctx.entries("Convo"))) == 1
        assert len(list(ctx.entries("convo"))) == 1

    def test_get_by_id(self):
        ctx = Context("test")
        entry = ctx.convo.add("Find me!")
        
        found = ctx.get(entry.id)
        assert found is entry

    def test_get_nonexistent_returns_none(self):
        ctx = Context("test")
        ctx.convo.add("Something")
        
        found = ctx.get("zzz")
        assert found is None


class TestRedactByID:
    def setup_method(self):
        reset_generator(seed=42, length=4)

    def test_redact_by_id_string(self):
        ctx = Context("test")
        entry = ctx.convo.add("Remove me")
        entry_id = entry.id
        
        result = ctx.redact(entry_id)
        
        assert result is True
        assert ctx.get(entry_id) is None

    def test_redact_nonexistent_id_returns_false(self):
        ctx = Context("test")
        ctx.convo.add("Keep me")
        
        result = ctx.redact("zzz")
        
        assert result is False

    def test_redact_by_id_with_tombstone(self):
        ctx = Context("test")
        entry = ctx.convo.add("Sensitive data")
        entry_id = entry.id
        
        ctx.redact(entry_id, tombstone="[REDACTED]")
        
        result = ctx.compile()
        assert "[REDACTED]" in result
        assert "Sensitive data" not in result


class TestEntryRelease:
    """Tests for Entry ID release and recycling."""

    def setup_method(self):
        reset_generator(seed=42, length=4)

    def test_entry_release_clears_id(self):
        entry = StringEntry("Test")
        old_id = entry.id
        assert old_id is not None

        entry.release()
        assert entry.id is None

    def test_entry_release_is_idempotent(self):
        entry = StringEntry("Test")
        entry.release()
        entry.release()  # Should not raise
        assert entry.id is None

    def test_redact_releases_id(self):
        from loom.ids import _generator_entry

        ctx = Context("test")
        entry = ctx.convo.add("Remove me")
        remaining_before = _generator_entry.remaining

        ctx.redact(entry)

        assert entry.id is None
        assert _generator_entry.remaining == remaining_before + 1

    def test_redact_with_tombstone_releases_old_id(self):
        ctx = Context("test")
        entry = ctx.convo.add("Remove me")

        ctx.redact(entry, tombstone="[GONE]")

        # Old entry's ID is released (set to None)
        assert entry.id is None
        # Tombstone entry exists and has an ID
        tombstone_entry = ctx.convo.entries[0]
        assert tombstone_entry.id is not None
        assert tombstone_entry.compile() == "[GONE]"

    def test_section_clear_releases_ids(self):
        from loom.ids import _generator_entry

        ctx = Context("test")
        entries = [ctx.step.add(f"Entry {i}") for i in range(10)]
        remaining_before = _generator_entry.remaining

        ctx.step.clear()

        assert _generator_entry.remaining == remaining_before + 10
        for entry in entries:
            assert entry.id is None
