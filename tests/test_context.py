"""Basic tests for Context."""

import pytest
from loom import Context, StringEntry, reset_generator, reset_context_generator


@pytest.fixture(autouse=True)
def fast_generators():
    """Use short IDs for fast tests."""
    reset_generator(seed=42, length=2)
    reset_context_generator(seed=42, length=2)


class TestContext:
    def test_create_context(self):
        ctx = Context("test")
        assert ctx.name == "test"
        assert ctx.router is None
        assert len(ctx.visitors) == 0

    def test_add_entry_to_section(self):
        ctx = Context("test")
        ctx.foundation.add("Hello, world!")
        assert len(ctx.foundation.entries) == 1

    def test_add_string_auto_wraps(self):
        ctx = Context("test")
        entry = ctx.foundation.add("Hello!")
        assert isinstance(entry, StringEntry)

    def test_compile_empty_context(self):
        ctx = Context("test")
        result = ctx.compile()
        assert result == ""

    def test_compile_with_foundation(self):
        ctx = Context("test")
        ctx.foundation.add("You are a helpful assistant.")
        result = ctx.compile()
        assert "FOUNDATION" in result
        assert "helpful assistant" in result

    def test_visitor_inclusion(self):
        main = Context("main")
        visitor = Context("security")
        visitor.foundation.add("Always validate input.")

        main.include(visitor)
        assert len(main.visitors) == 1

        result = main.compile()
        assert "security" in result.lower()
        assert "validate input" in result

    def test_visitor_exclusion(self):
        main = Context("main")
        visitor = Context("security")

        main.include(visitor)
        main.exclude(visitor)
        assert len(main.visitors) == 0

    def test_step_cleared_after_compile(self):
        ctx = Context("test")
        ctx.step.add("Current task output")
        assert len(ctx.step.entries) == 1

        ctx.compile(clear_volatile=True)
        assert len(ctx.step.entries) == 0

    def test_step_preserved_when_not_clearing(self):
        ctx = Context("test")
        ctx.step.add("Current task output")

        ctx.compile(clear_volatile=False)
        assert len(ctx.step.entries) == 1

    def test_deduplication(self):
        main = Context("main")
        visitor = Context("visitor")

        # Same content in both
        main.foundation.add("Shared rule")
        visitor.foundation.add("Shared rule")

        main.include(visitor)
        result = main.compile()

        # Should only appear once
        assert result.count("Shared rule") == 1


class TestRemember:
    """Tests for Context.remember() - promoting entries to CONVO."""

    def test_remember_string(self):
        ctx = Context("test")
        ctx.remember("Important finding")

        assert len(ctx.convo.entries) == 1
        result = ctx.compile()
        assert "Important finding" in result

    def test_remember_with_summarize(self):
        ctx = Context("test")
        ctx.remember("API returns JSON", summarize=True)

        result = ctx.compile()
        assert "Learned: API returns JSON" in result

    def test_remember_entry_from_step(self):
        ctx = Context("test")

        # Simulate a tool call in step
        ctx.step.add("$ gh pr list\n#1 Initial structure OPEN")
        step_entry = ctx.step.entries[-1]

        # Remember it before compile clears step
        ctx.remember(step_entry)

        # Compile clears step
        ctx.compile(clear_volatile=True)

        # But convo still has it
        assert len(ctx.convo.entries) == 1
        result = ctx.compile()
        assert "gh pr list" in result

    def test_remember_returns_entry(self):
        ctx = Context("test")
        entry = ctx.remember("Something important")

        assert isinstance(entry, StringEntry)
        assert entry in ctx.convo.entries


class TestRedact:
    """Tests for Context.redact() - removing entries from context."""

    def test_redact_removes_entry(self):
        ctx = Context("test")
        entry = ctx.convo.add("Sensitive information")

        result = ctx.redact(entry)

        assert result is True
        assert entry not in ctx.convo.entries
        assert len(ctx.convo.entries) == 0

    def test_redact_with_tombstone(self):
        ctx = Context("test")
        entry = ctx.convo.add("Credit card: 4532-xxxx-1234")
        entry.name = "User message"

        ctx.redact(entry, tombstone="[REDACTED: contained PII]")

        # Entry replaced, not removed
        assert len(ctx.convo.entries) == 1
        result = ctx.compile()
        assert "[REDACTED: contained PII]" in result
        assert "4532" not in result

    def test_redact_preserves_entry_name_in_tombstone(self):
        ctx = Context("test")
        entry = ctx.foundation.add("Secret config")
        entry.name = "config.yaml"

        ctx.redact(entry, tombstone="[REDACTED]")

        # Tombstone should keep the name for context
        result = ctx.compile()
        assert "config.yaml" in result
        assert "[REDACTED]" in result

    def test_redact_searches_all_sections(self):
        ctx = Context("test")

        # Add to different sections
        e1 = ctx.foundation.add("Foundation entry")
        e2 = ctx.focus.add("Focus entry")
        e3 = ctx.topic.add("Topic entry")
        e4 = ctx.convo.add("Convo entry")
        e5 = ctx.step.add("Step entry")
        e6 = ctx.attention.add("Attention entry")

        # Redact from each
        assert ctx.redact(e1) is True
        assert ctx.redact(e2) is True
        assert ctx.redact(e3) is True
        assert ctx.redact(e4) is True
        assert ctx.redact(e5) is True
        assert ctx.redact(e6) is True

        # All gone
        result = ctx.compile()
        assert result == ""

    def test_redact_nonexistent_returns_false(self):
        ctx = Context("test")
        entry = StringEntry("Not in context")

        result = ctx.redact(entry)

        assert result is False
