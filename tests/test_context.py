"""Basic tests for Context."""

import pytest
from loom import Context, StringEntry


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
