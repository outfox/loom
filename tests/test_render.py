"""Tests for Context.render() and Context.to_messages()."""

import pytest
from loom import Context, StringEntry, FileEntry
from loom.ids import reset_generator, reset_context_generator


@pytest.fixture(autouse=True)
def reset_ids():
    """Reset ID generators before each test for deterministic IDs."""
    reset_generator(seed=42, length=2)
    reset_context_generator(seed=42)
    yield


class TestRender:
    """Tests for Context.render()."""

    def test_render_empty_context(self):
        """Empty context renders to empty string."""
        ctx = Context("empty")
        assert ctx.render() == ""

    def test_render_single_entry(self):
        """Single entry renders correctly."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("You are a helpful assistant."))
        
        result = ctx.render()
        assert "You are a helpful assistant." in result
        assert "FOUNDATION" in result

    def test_render_multiple_sections(self):
        """Multiple sections render in order."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("Base identity"))
        ctx.topic.add(StringEntry("Current topic"))
        ctx.attention.add(StringEntry("Pay attention to this"))
        
        result = ctx.render()
        
        # All content present
        assert "Base identity" in result
        assert "Current topic" in result
        assert "Pay attention to this" in result
        
        # Order: foundation before topic before attention
        foundation_pos = result.find("Base identity")
        topic_pos = result.find("Current topic")
        attention_pos = result.find("Pay attention to this")
        
        assert foundation_pos < topic_pos < attention_pos

    def test_render_clears_volatile_by_default(self):
        """render() clears step section by default."""
        ctx = Context("test")
        ctx.step.add(StringEntry("Volatile content"))
        
        # First render includes step
        result1 = ctx.render()
        assert "Volatile content" in result1
        
        # Second render: step is cleared
        result2 = ctx.render()
        assert "Volatile content" not in result2

    def test_render_preserve_volatile(self):
        """render(clear_volatile=False) preserves step section."""
        ctx = Context("test")
        ctx.step.add(StringEntry("Volatile content"))
        
        # Render without clearing
        result1 = ctx.render(clear_volatile=False)
        assert "Volatile content" in result1
        
        # Still there
        result2 = ctx.render(clear_volatile=False)
        assert "Volatile content" in result2

    def test_render_with_named_entries(self):
        """Named entries get headers."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("Identity content", name="IDENTITY"))
        
        result = ctx.render()
        assert "# IDENTITY" in result
        assert "Identity content" in result


class TestToMessages:
    """Tests for Context.to_messages()."""

    def test_to_messages_returns_list(self):
        """to_messages() returns a list of message dicts."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("You are helpful."))
        
        messages = ctx.to_messages()
        
        assert isinstance(messages, list)
        assert len(messages) == 1

    def test_to_messages_system_role(self):
        """First message has system role."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("System prompt content"))
        
        messages = ctx.to_messages()
        
        assert messages[0]["role"] == "system"
        assert "System prompt content" in messages[0]["content"]

    def test_to_messages_empty_context(self):
        """Empty context returns system message with empty content."""
        ctx = Context("empty")
        
        messages = ctx.to_messages()
        
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == ""

    def test_to_messages_clears_volatile(self):
        """to_messages() clears step section by default."""
        ctx = Context("test")
        ctx.step.add(StringEntry("Volatile"))
        
        messages1 = ctx.to_messages()
        assert "Volatile" in messages1[0]["content"]
        
        messages2 = ctx.to_messages()
        assert "Volatile" not in messages2[0]["content"]

    def test_to_messages_openai_compatible(self):
        """Output is compatible with OpenAI chat API format."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("You are Blue."))
        ctx.topic.add(StringEntry("Help with Python."))
        
        messages = ctx.to_messages()
        
        # OpenAI expects: [{"role": "...", "content": "..."}]
        for msg in messages:
            assert "role" in msg
            assert "content" in msg
            assert isinstance(msg["role"], str)
            assert isinstance(msg["content"], str)
