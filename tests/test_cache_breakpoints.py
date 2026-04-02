"""Tests for cache breakpoints in to_messages()."""

import pytest

from loom import Context, StringEntry, reset_context_generator


@pytest.fixture(autouse=True)
def reset_ids():
    """Reset ID generator before each test for reproducibility."""
    reset_context_generator(seed=42)


class TestCacheBreakpoints:
    """Tests for Anthropic-style cache breakpoints."""

    def test_to_messages_simple_format(self):
        """Without cache_breakpoints, returns simple string content."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("You are helpful."))
        ctx.topic.add(StringEntry("Current task."))

        messages = ctx.to_messages()

        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert isinstance(messages[0]["content"], list)
        texts = " ".join(b["text"] for b in messages[0]["content"])
        assert "You are helpful." in texts
        assert "Current task." in texts

    def test_to_messages_with_breakpoints(self):
        """With cache_breakpoints, returns multi-block format."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("Foundation content."))
        ctx.topic.add(StringEntry("Topic content."))

        messages = ctx.to_messages(cache_breakpoints=["foundation", "topic"])

        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        content = messages[0]["content"]
        assert isinstance(content, list)

    def test_cache_control_on_breakpoint_sections(self):
        """Sections in cache_breakpoints get cache_control."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("Foundation."))
        ctx.focus.add(StringEntry("Focus."))
        ctx.topic.add(StringEntry("Topic."))

        messages = ctx.to_messages(cache_breakpoints=["foundation", "topic"])
        content = messages[0]["content"]

        # Find foundation block
        foundation_block = next(b for b in content if "Foundation." in b["text"])
        assert foundation_block.get("cache_control") == {"type": "ephemeral"}

        # Find focus block (no cache_control)
        focus_block = next(b for b in content if "Focus." in b["text"])
        assert "cache_control" not in focus_block

        # Find topic block
        topic_block = next(b for b in content if "Topic." in b["text"])
        assert topic_block.get("cache_control") == {"type": "ephemeral"}

    def test_empty_sections_not_included(self):
        """Empty sections don't create blocks."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("Foundation only."))
        # topic, focus, etc. are empty

        messages = ctx.to_messages(cache_breakpoints=["foundation"])
        content = messages[0]["content"]

        assert len(content) == 1
        assert "Foundation only." in content[0]["text"]

    def test_case_insensitive_breakpoints(self):
        """Breakpoint names are case-insensitive."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("Foundation."))

        messages = ctx.to_messages(cache_breakpoints=["FOUNDATION", "Topic"])
        content = messages[0]["content"]

        foundation_block = content[0]
        assert foundation_block.get("cache_control") == {"type": "ephemeral"}

    def test_all_sections_as_breakpoints(self):
        """Can set breakpoints on all sections."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("F"))
        ctx.focus.add(StringEntry("Fo"))
        ctx.topic.add(StringEntry("T"))
        ctx.convo.add(StringEntry("C"))
        ctx.step.add(StringEntry("S"))
        ctx.attention.add(StringEntry("A"))

        messages = ctx.to_messages(
            cache_breakpoints=["foundation", "focus", "topic", "convo", "step", "attention"],
            clear_volatile=False,  # Keep step for assertion
        )
        content = messages[0]["content"]

        # All 6 sections should have cache_control
        for block in content:
            assert block.get("cache_control") == {"type": "ephemeral"}

    def test_clear_volatile_with_breakpoints(self):
        """clear_volatile works with cache breakpoints."""
        ctx = Context("test")
        ctx.step.add(StringEntry("Volatile."))

        # First call with clear_volatile=True (default)
        messages = ctx.to_messages(cache_breakpoints=["step"])
        content = messages[0]["content"]
        assert len(content) == 1
        assert "Volatile." in content[0]["text"]

        # Second call - step should be empty
        messages = ctx.to_messages(cache_breakpoints=["step"])
        content = messages[0]["content"]
        assert len(content) == 0

    def test_visitors_included_in_blocks(self):
        """Visitor content is included in appropriate blocks."""
        main = Context("main")
        visitor = Context("helper")

        main.foundation.add(StringEntry("Main foundation."))
        visitor.foundation.add(StringEntry("Visitor foundation."))
        main.include(visitor)

        messages = main.to_messages(cache_breakpoints=["foundation"])
        content = messages[0]["content"]

        all_text = " ".join(b["text"] for b in content)
        assert "Main foundation." in all_text
        assert "Visitor foundation." in all_text
        # cache_control should be on the last foundation-related block
        assert any(b.get("cache_control") == {"type": "ephemeral"} for b in content)
