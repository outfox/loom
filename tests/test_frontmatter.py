"""Tests for frontmatter-based role support in entries."""

import pytest
import tempfile
from pathlib import Path

from loom import Context, StringEntry, FileEntry
from loom.entry import _parse_frontmatter
from loom.ids import reset_generator, reset_context_generator


@pytest.fixture(autouse=True)
def reset_ids():
    """Reset ID generators before each test for deterministic IDs."""
    reset_generator(seed=42, length=2)
    reset_context_generator(seed=42)
    yield


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


class TestParseFrontmatter:
    """Tests for the _parse_frontmatter helper."""

    def test_no_frontmatter(self):
        meta, body = _parse_frontmatter("Hello, world!")
        assert meta == {}
        assert body == "Hello, world!"

    def test_simple_frontmatter(self):
        content = "---\nrole: assistant\n---\nHello, world!"
        meta, body = _parse_frontmatter(content)
        assert meta == {"role": "assistant"}
        assert body == "Hello, world!"

    def test_frontmatter_with_quotes(self):
        content = '---\nrole: "assistant"\n---\nBody text'
        meta, _ = _parse_frontmatter(content)
        assert meta["role"] == "assistant"

    def test_frontmatter_with_single_quotes(self):
        content = "---\nrole: 'assistant'\n---\nBody text"
        meta, _ = _parse_frontmatter(content)
        assert meta["role"] == "assistant"

    def test_multiple_fields(self):
        content = "---\nrole: assistant\ntitle: My Title\npriority: high\n---\nBody"
        meta, body = _parse_frontmatter(content)
        assert meta == {"role": "assistant", "title": "My Title", "priority": "high"}
        assert body == "Body"

    def test_no_closing_delimiter(self):
        content = "---\nrole: assistant\nNo closing"
        meta, body = _parse_frontmatter(content)
        assert meta == {}
        assert body == content

    def test_body_newlines_stripped(self):
        content = "---\nrole: assistant\n---\n\n\nBody with leading newlines"
        _, body = _parse_frontmatter(content)
        assert body == "Body with leading newlines"

    def test_empty_frontmatter(self):
        content = "---\n---\nBody"
        meta, body = _parse_frontmatter(content)
        assert meta == {}
        assert body == "Body"


class TestEntryRole:
    """Tests for role parameter on Entry subclasses."""

    def test_string_entry_default_role(self):
        entry = StringEntry("Hello")
        assert entry.role == "system"

    def test_string_entry_custom_role(self):
        entry = StringEntry("Hello", role="assistant")
        assert entry.role == "assistant"

    def test_string_entry_user_role(self):
        entry = StringEntry("Hello", role="user")
        assert entry.role == "user"

    def test_file_entry_default_role(self, tmp_dir):
        f = tmp_dir / "test.md"
        f.write_text("No frontmatter here")
        entry = FileEntry(f)
        assert entry.role == "system"

    def test_file_entry_frontmatter_role(self, tmp_dir):
        f = tmp_dir / "test.md"
        f.write_text("---\nrole: assistant\n---\nAssistant content")
        entry = FileEntry(f)
        assert entry.role == "assistant"

    def test_file_entry_explicit_role_overrides_frontmatter(self, tmp_dir):
        f = tmp_dir / "test.md"
        f.write_text("---\nrole: assistant\n---\nContent")
        entry = FileEntry(f, role="user")
        assert entry.role == "user"

    def test_file_entry_explicit_system_role(self, tmp_dir):
        """Explicitly passing role='system' should override frontmatter."""
        f = tmp_dir / "test.md"
        f.write_text("---\nrole: assistant\n---\nContent")
        entry = FileEntry(f, role="system")
        assert entry.role == "system"


class TestFileEntryCompile:
    """Tests for FileEntry.compile() stripping frontmatter."""

    def test_compile_strips_frontmatter(self, tmp_dir):
        f = tmp_dir / "test.md"
        f.write_text("---\nrole: assistant\n---\nThe actual body content")
        entry = FileEntry(f)
        assert entry.compile() == "The actual body content"

    def test_compile_no_frontmatter_unchanged(self, tmp_dir):
        f = tmp_dir / "test.md"
        f.write_text("Just plain content")
        entry = FileEntry(f)
        assert entry.compile() == "Just plain content"

    def test_compile_frontmatter_with_extra_fields(self, tmp_dir):
        f = tmp_dir / "test.md"
        f.write_text("---\nrole: assistant\ntitle: Example\n---\nBody text here")
        entry = FileEntry(f)
        assert entry.compile() == "Body text here"
        assert entry.role == "assistant"

    def test_compile_multiline_body(self, tmp_dir):
        f = tmp_dir / "test.md"
        f.write_text("---\nrole: assistant\n---\nLine 1\nLine 2\nLine 3")
        entry = FileEntry(f)
        assert entry.compile() == "Line 1\nLine 2\nLine 3"

    def test_compile_rereads_on_each_call(self, tmp_dir):
        """FileEntry re-reads the file on every compile()."""
        f = tmp_dir / "test.md"
        f.write_text("Version 1")
        entry = FileEntry(f)
        assert entry.compile() == "Version 1"

        f.write_text("Version 2")
        assert entry.compile() == "Version 2"

    def test_compile_rereads_frontmatter_role(self, tmp_dir):
        """Role from frontmatter is updated on re-read."""
        f = tmp_dir / "test.md"
        f.write_text("---\nrole: assistant\n---\nBody")
        entry = FileEntry(f)
        assert entry.role == "assistant"

        f.write_text("---\nrole: user\n---\nNew body")
        entry.compile()
        assert entry.role == "user"

    def test_compile_explicit_role_not_overridden_by_reread(self, tmp_dir):
        """Explicit role parameter is never overridden by frontmatter on re-read."""
        f = tmp_dir / "test.md"
        f.write_text("---\nrole: assistant\n---\nBody")
        entry = FileEntry(f, role="system")
        assert entry.role == "system"

        f.write_text("---\nrole: user\n---\nNew body")
        entry.compile()
        assert entry.role == "system"

    def test_compile_deleted_file(self, tmp_dir):
        """Deleted file returns a notice instead of raising."""
        f = tmp_dir / "test.md"
        f.write_text("Content")
        entry = FileEntry(f)
        f.unlink()

        result = entry.compile()
        assert "File removed" in result
        assert "test.md" in result

    def test_compile_deleted_then_recreated(self, tmp_dir):
        """File can be re-read after being deleted and recreated."""
        f = tmp_dir / "test.md"
        f.write_text("Original")
        entry = FileEntry(f)
        f.unlink()
        assert "File removed" in entry.compile()

        f.write_text("Recreated")
        assert entry.compile() == "Recreated"


class TestContextToMessagesWithRoles:
    """Tests for Context.to_messages() with non-system role entries."""

    def test_assistant_entry_emitted_as_separate_message(self):
        ctx = Context("test")
        ctx.foundation.add(StringEntry("You are a helpful assistant."))
        ctx.topic.add(StringEntry("Some assistant prefill", role="assistant"))

        messages = ctx.to_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "assistant"
        assert "Some assistant prefill" in messages[1]["content"]

    def test_assistant_entry_excluded_from_system_message(self):
        ctx = Context("test")
        ctx.foundation.add(StringEntry("System content"))
        ctx.topic.add(StringEntry("Assistant prefill", role="assistant"))

        messages = ctx.to_messages()

        # System message should NOT contain assistant content
        sys_texts = " ".join(b["text"] for b in messages[0]["content"])
        assert "Assistant prefill" not in sys_texts
        assert "System content" in sys_texts

    def test_multiple_non_system_entries(self):
        ctx = Context("test")
        ctx.foundation.add(StringEntry("System prompt"))
        ctx.topic.add(StringEntry("First assistant msg", role="assistant"))
        ctx.attention.add(StringEntry("Second assistant msg", role="assistant"))

        messages = ctx.to_messages()

        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "assistant"
        assert "First assistant msg" in messages[1]["content"]
        assert "Second assistant msg" in messages[2]["content"]

    def test_no_non_system_entries_backward_compat(self):
        """When no non-system entries exist, behavior is unchanged."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("You are helpful."))
        ctx.topic.add(StringEntry("Current task"))

        messages = ctx.to_messages()

        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        texts = " ".join(b["text"] for b in messages[0]["content"])
        assert "You are helpful." in texts
        assert "Current task" in texts

    def test_render_includes_all_entries_regardless_of_role(self):
        """render() should include ALL entries, even non-system ones (backward compat)."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("System content"))
        ctx.topic.add(StringEntry("Assistant content", role="assistant"))

        result = ctx.render()

        assert "System content" in result
        assert "Assistant content" in result

    def test_compile_includes_all_entries_regardless_of_role(self):
        """compile() should include ALL entries, even non-system ones."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("System content"))
        ctx.topic.add(StringEntry("Assistant content", role="assistant"))

        result = ctx.compile()

        assert "System content" in result
        assert "Assistant content" in result

    def test_file_entry_with_frontmatter_role_in_context(self, tmp_dir):
        """FileEntry with frontmatter role works end-to-end in Context."""
        f = tmp_dir / "prefill.md"
        f.write_text("---\nrole: assistant\n---\nI'll help you with that.")

        ctx = Context("test")
        ctx.foundation.add(StringEntry("You are a helpful assistant."))
        ctx.topic.add(FileEntry(f))

        messages = ctx.to_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "assistant"
        assert "I'll help you with that." in messages[1]["content"]
        # Frontmatter should be stripped from compiled content
        assert "---" not in messages[1]["content"]

    def test_named_non_system_entry_gets_header(self):
        """Non-system entries with names get # headers in their message."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("System"))
        ctx.topic.add(StringEntry("Prefill content", name="Prefill", role="assistant"))

        messages = ctx.to_messages()

        assert len(messages) == 2
        assert "# Prefill" in messages[1]["content"]
        assert "Prefill content" in messages[1]["content"]

    def test_to_messages_with_cache_breakpoints_and_roles(self):
        """Cache breakpoints work correctly with non-system role entries."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("Foundation content"))
        ctx.topic.add(StringEntry("Topic content"))
        ctx.attention.add(StringEntry("Assistant msg", role="assistant"))

        messages = ctx.to_messages(cache_breakpoints=["foundation"])

        # System message is first, with block format
        assert messages[0]["role"] == "system"
        assert isinstance(messages[0]["content"], list)

        # Assistant message comes after
        assert messages[-1]["role"] == "assistant"
        assert "Assistant msg" in messages[-1]["content"]

        # System blocks should NOT contain assistant content
        for block in messages[0]["content"]:
            if isinstance(block, dict) and "text" in block:
                assert "Assistant msg" not in block["text"]

    def test_user_role_entry(self):
        """User role entries are also emitted as separate messages."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("System prompt"))
        ctx.topic.add(StringEntry("User message", role="user"))

        messages = ctx.to_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_mixed_roles(self):
        """Multiple different roles are handled correctly."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("System prompt"))
        ctx.topic.add(StringEntry("Assistant prefill", role="assistant"))
        ctx.attention.add(StringEntry("User query", role="user"))

        messages = ctx.to_messages()

        assert len(messages) == 3
        roles = [m["role"] for m in messages]
        assert roles == ["system", "assistant", "user"]

    def test_visitor_non_system_entries(self):
        """Non-system entries from visitors are also collected."""
        main = Context("main")
        visitor = Context("visitor")

        main.foundation.add(StringEntry("Main system"))
        visitor.topic.add(StringEntry("Visitor assistant msg", role="assistant"))
        main.include(visitor)

        messages = main.to_messages()

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "assistant"
        assert "Visitor assistant msg" in messages[1]["content"]

    def test_clear_volatile_with_roles(self):
        """Step section clearing works correctly with role entries."""
        ctx = Context("test")
        ctx.foundation.add(StringEntry("System"))
        ctx.step.add(StringEntry("Volatile assistant", role="assistant"))

        messages1 = ctx.to_messages()
        assert len(messages1) == 2
        assert messages1[1]["role"] == "assistant"

        # Step should be cleared
        messages2 = ctx.to_messages()
        assert len(messages2) == 1  # Only system message now
